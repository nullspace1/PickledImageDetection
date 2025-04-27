import tensorflow as tf
import numpy as np
from keras import layers, Model

class CrossCorrelation(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    def call(self, screenshot_feat, template_feat):
        screenshot_feat = tf.transpose(screenshot_feat, [0,3,1,2])
        template_feat = tf.transpose(template_feat, [0,3,1,2])
        
        template_feat = tf.image.flip_left_right(template_feat)
        template_feat = tf.image.flip_up_down(template_feat)
        
        correlation = tf.nn.conv2d(
            screenshot_feat,
            template_feat,
            strides=1,
            padding='VALID',
            data_format='NCHW'
        )
        
        correlation = tf.transpose(correlation, [0,2,3,1])
        return correlation

def feature_extractor():
    inputs = layers.Input(shape=(None,None,3))
    
    x = layers.Conv2D(32,3,strides=2, padding='same', activation='relu')(inputs)
    feat1 = x
    
    x = layers.Conv2D(64,3,strides=2, padding='same', activation='relu')(x)
    feat2 = x
    
    x = layers.Conv2D(128,3,strides=2, padding='same', activation='relu')(x)
    feat3 = x
    
    model = Model(inputs=inputs, outputs=[feat1,feat2,feat3])
    return model

def build_detection_head(input):
    x = layers.Conv2D(64,3,strides=2, padding='same', activation='relu')(input)
    x = layers.GlobalAveragePooling2D()(x)
    
    bbox_output = layers.Dense(4)(x)
    found_output = layers.Dense(1, activation='sigmoid')(x)
    return bbox_output, found_output
    
def build_full_model():
    
    cross_correlation_layer = CrossCorrelation()
    
    screenshot_input = layers.Input(shape=(None,None,3))
    template_input = layers.Input(shape=(None,None,3))
    
    extractor = feature_extractor()
    
    screenshot_features = extractor(screenshot_input)
    template_features = extractor(template_input)
    
    correlations = []
    for screenshot_feat, template_feat in zip(screenshot_features, template_features):
        correlation = cross_correlation_layer(screenshot_feat, template_feat)
        correlations.append(correlation)
        
    base_size = (128,128)
    resized_correlations = []
    for correlation in correlations:
        correlation = tf.image.resize(correlation, base_size)
        resized_correlations.append(correlation)
        
    fused = layers.Concatenate()(resized_correlations)
    bbox_output, found_output = build_detection_head(fused)
    
    model = Model(inputs=[screenshot_input, template_input], outputs=[bbox_output, found_output])
    return model
    
def smooth_l1_loss(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), 'float32')
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return tf.reduce_sum(loss)
    
def custom_loss(y_true_bbox, y_pred_bbox, y_true_found, y_pred_found, lambda_coord=1.0, lambda_found=1.0):
    bbox_loss = smooth_l1_loss(y_true_bbox, y_pred_bbox)
    bbox_loss = bbox_loss *  tf.squeeze(y_true_found, -1)
    bbox_loss = tf.reduce_mean(bbox_loss)
    
    found_loss = tf.keras.losses.binary_crossentropy(y_true_found, y_pred_found)
    found_loss = tf.reduce_mean(found_loss)
    
    loss = lambda_coord * bbox_loss + lambda_found * found_loss
    return loss

def preprocess_sample(screenshot,template,bbox,found):
    screenshot = tf.image.convert_image_dtype(screenshot, tf.float32)
    template = tf.image.convert_image_dtype(template, tf.float32)
    
    return (screenshot,template), (bbox,found)

def load_training_data(path):
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

def create_dataset(screenshots, templates, bboxes, founds, batch_size = 16, shuffle = True):
    dataset = tf.data.Dataset.from_tensor_slices((screenshots, templates, bboxes, founds))
    dataset = dataset.map(preprocess_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


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
            loss = custom_loss(bboxes, pred_bboxes, founds, pred_founds)
            
        gradients = tape.gradient(loss, self.backbone.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables))
        return {'loss': loss}
    
    
full_model = build_full_model()
detection_model = FullDetectionModel(full_model)
detection_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))


screenshots, templates, bboxes, founds = load_training_data('./data/data.npy')
dataset = create_dataset(screenshots, templates, bboxes, founds)
detection_model.fit(dataset, epochs=50)

detection_model.save('./model.keras')