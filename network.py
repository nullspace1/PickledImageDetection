import tensorflow as tf
import numpy as np
import os
from keras import layers

class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    def build(self, input_shape):
        self.conv1 = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')
        
    def multilayer_convolve(self, image):
        x1 = self.conv1(image)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x1, x2, x3
        
    def call(self, inputs):
        screenshot, crop = inputs
        
        sfeat1, sfeat2, sfeat3 = self.multilayer_convolve(screenshot)
        cfeat1, cfeat2, cfeat3 = self.multilayer_convolve(crop)
        
        return (sfeat1, sfeat2, sfeat3), (cfeat1, cfeat2, cfeat3)

class CrossCorrelation(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
        
    ## TODO :: this is kind of ass, but it didn't bork the model so i'm taking the W. might need to fix later
    ## does a manual convolution since tf.nn.depthwise_conv2d doesn't between featuresets of my own
    def cross_correlate_batch(self, screenshots, templates):
        
        batch_size = tf.shape(screenshots)[0]
        in_channels = screenshots.shape[-1]
        template_channels = templates.shape[-1]

        if in_channels != template_channels:
            raise ValueError(f"Channel mismatch: screenshots {in_channels} vs templates {template_channels}")

        templates = tf.image.flip_left_right(templates)
        templates = tf.image.flip_up_down(templates)

        templates = tf.expand_dims(templates, axis=-1)

        screenshots = tf.transpose(screenshots, [1,2,0,3])
        screenshots = tf.reshape(screenshots, [1, tf.shape(screenshots)[0], tf.shape(screenshots)[1], batch_size * in_channels])

        templates = tf.reshape(templates, [tf.shape(templates)[1], tf.shape(templates)[2], batch_size * in_channels, 1])

        correlation = tf.nn.depthwise_conv2d(
            screenshots,
            templates,
            strides=[1,1,1,1],
            padding='VALID'
        )
        
        correlation = tf.reshape(correlation, [tf.shape(correlation)[1], tf.shape(correlation)[2], batch_size, 1])
        correlation = tf.transpose(correlation, [2, 0, 1, 3]) 

        return correlation
    
    def call(self, inputs):
        screenshot_feats, template_feats = inputs
        (sfeat1, sfeat2, sfeat3), (cfeat1, cfeat2, cfeat3) = screenshot_feats, template_feats
        
        correlation_1 = self.cross_correlate_batch(sfeat1, cfeat1)
        correlation_2 = self.cross_correlate_batch(sfeat2, cfeat2)
        correlation_3 = self.cross_correlate_batch(sfeat3, cfeat3)
        
        correlation_1 = tf.image.resize(correlation_1, (128, 128))
        correlation_2 = tf.image.resize(correlation_2, (128, 128))
        correlation_3 = tf.image.resize(correlation_3, (128, 128))
        
        correlation = tf.concat([correlation_1, correlation_2, correlation_3], axis=-1)
        
        return correlation
        
      
class DetectionHead(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()


    def build(self, input_shape):
        self.conv2d = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')
        self.pooling = layers.GlobalAveragePooling2D()
        self.dense_box = layers.Dense(4)
        self.dense_found = layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        correlation = inputs
        correlation = self.conv2d(correlation)
        correlation = self.pooling(correlation)
        correlation = self.dense_box(correlation)
        found = self.dense_found(correlation)
        
        return correlation, found

def __smooth_l1_loss__(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = (less_than_one * 0.5 * tf.square(diff)) + (1.0 - less_than_one) * (diff - 0.5)
    return tf.reduce_sum(loss)

def __custom_loss__(y_true_bbox, y_pred_bbox, y_true_found, y_pred_found, lambda_coord=1.0, lambda_found=1.0):
    y_true_bbox = tf.cast(y_true_bbox, tf.float32)
    y_pred_bbox = tf.cast(y_pred_bbox, tf.float32)
    y_true_found = tf.cast(y_true_found, tf.float32)
    y_pred_found = tf.cast(y_pred_found, tf.float32)

    y_true_found = tf.expand_dims(y_true_found, axis=-1)

    bbox_loss = __smooth_l1_loss__(y_true_bbox, y_pred_bbox)
    bbox_loss = bbox_loss * tf.squeeze(y_true_found, axis=-1)
    bbox_loss = tf.reduce_mean(bbox_loss)
    
    found_loss = tf.keras.losses.binary_crossentropy(y_true_found, y_pred_found)
    found_loss = tf.reduce_mean(found_loss)
    
    loss = lambda_coord * bbox_loss + lambda_found * found_loss
    return loss

def __preprocess_sample__(screenshot,template,bbox,found):
    screenshot = tf.image.convert_image_dtype(screenshot, tf.float32)
    template = tf.image.convert_image_dtype(template, tf.float32)
    
    return (screenshot,template), (bbox,found)

def __load_training_data__(path):
    data = np.load(path, allow_pickle=True)
    screenshots = []
    templates = []
    bboxes = []
    founds = []
    for screenshot, template, bbox, found in data:
        screenshots.append(screenshot)
        templates.append(template)
        bboxes.append(bbox)
        founds.append(found)
    return np.array(screenshots), np.array(templates), np.array(bboxes), np.array(founds)

def __create_dataset__(screenshots, templates, bboxes, founds, batch_size = 16, shuffle = True):
    dataset = tf.data.Dataset.from_tensor_slices((screenshots, templates, bboxes, founds))
    dataset = dataset.map(__preprocess_sample__, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def __build_full_model__():
    feature_extractor = FeatureExtractor()
    correlation = CrossCorrelation()
    detection_head = DetectionHead()
    return tf.keras.Sequential([
        feature_extractor,
        correlation,
        detection_head
    ])


class FullDetectionModel(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.backbone = model
        
    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        
    def train_step(self, data):
        (screenshots, templates), (bboxes, founds) = data
        
        with tf.GradientTape() as tape:
            pred_bboxes, pred_founds = self.backbone([screenshots, templates], training=True)
            loss = __custom_loss__(bboxes, pred_bboxes, founds, pred_founds)
            
        gradients = tape.gradient(loss, self.backbone.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables))
        return {'loss': loss}

def __get_model__():
    full_model = __build_full_model__()
    detection_model = FullDetectionModel(full_model)
    detection_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    return detection_model

def get_model_graph_sample():
    model = __build_full_model__()
    model.build(input_shape=[(1600, 900, 3), (300, 300, 3)])
    return tf.keras.utils.plot_model(model, to_file='./data/model.png', show_shapes=True)  

def train_pickled_model(dataset_file_path,train = True):
    detection_model = __get_model__()

    if train or not os.path.exists('./data/pickled_model.h5'):
        screenshots, templates, bboxes, founds = __load_training_data__(dataset_file_path)
        dataset = __create_dataset__(screenshots, templates, bboxes, founds)
        detection_model.fit(dataset, epochs=50)
        detection_model.save_weights('./data/pickled_model.h5')
    else:
        detection_model.load_weights('./data/pickled_model.h5')
        
    return detection_model
        
    


