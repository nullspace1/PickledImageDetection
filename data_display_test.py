import capturer as cap
import network as net


settings = cap.Settings(
    negative_sample_probability=0.1,
    max_crop_percentage=0.4,
    min_scale_x=0.8,
    max_scale_x=1.5,
    min_scale_y=0.8,
    max_scale_y=1.5,
    iterations=2
)

cap.load_training_data('screenshots', settings, 'training_data.npy')
net.train_pickled_model('training_data.npy')

