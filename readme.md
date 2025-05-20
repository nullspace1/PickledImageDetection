# Pickled Image Detection

This project implements a hypernetwork-based approach for detecting pickled images in screenshots. The model uses a combination of image processing and template matching to identify specific patterns in images.

## Project Structure

```
.
├── data/
│   ├── screenshots/     # Directory containing input screenshots
│   ├── templates/       # Directory containing template images
│   ├── generated_data/  # Directory for generated training data
│   ├── model.pth       # Saved model weights
│   └── loss_plot.png   # Training loss visualization
├── Hypernetwork.py     # Main model architecture
├── ImageProcessor.py   # Image processing module
├── TemplateProcessor.py # Template processing module
├── DataLoader.py       # Data loading utilities
├── DataCreator.py      # Training data generation
├── Trainer.py          # Training loop implementation
└── main.py            # Training script
```

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install requirements.txt
```

## Usage

### Training the Model

The model can be trained using the `main.py` script with various command-line arguments:

```bash
python main.py --batch_size 1 --samples 1000 --epochs 10 --model_path "data/model.pth"
```

#### Command Line Arguments

- `--batch_size`: Number of samples per training batch (default: 1)
- `--samples`: Number of training samples to generate (default: 1000)
- `--epochs`: Number of training epochs (default: 10)
- `--model_path`: Path to save/load the model (default: "data/model.pth")
- `--generated_data_path`: Path for generated training data (default: "data/generated_data")
- `--screenshots_path`: Path to screenshot images (default: "data/screenshots")
- `--templates_path`: Path to template images (default: "data/templates")
- `--training_data_path`: Path to training data file (default: "data/training_data.npy")
- `--validation_data_path`: Path to validation data file (default: "data/validation_data.npy")

### Data Organization

1. Place your screenshot images in the `data/screenshots/` directory
2. Place your template images in the `data/templates/` directory
3. The model will automatically generate training data in the `data/generated_data/` directory

### Training Process

The training process includes:
- Automatic data generation from screenshots and templates
- Training with early stopping (patience=10 epochs)
- Validation after each epoch
- Automatic model saving when validation loss improves
- Loss visualization saved as `data/loss_plot.png`

### Model Architecture

The model consists of three main components:
1. `ImageProcessor`: Processes input screenshots
2. `TemplateProcessor`: Processes template images
3. `HyperNetwork`: Combines both processors to generate detection heatmaps

### Output

The model outputs a heatmap where:
- Higher values (closer to 1) indicate higher probability of template match
- Lower values (closer to 0) indicate lower probability of template match


## Troubleshooting

1. If you get CUDA out of memory errors:
   - Reduce the batch size
   - Reduce the cache size
   - Process smaller images

2. If training is too slow:
   - Increase the batch size if memory allows
   - Reduce the number of templates per screenshot
   - Use a smaller number of samples

3. If the model isn't learning:
   - Check that your templates are clear and distinctive
   - Verify that your screenshots contain the templates
   - Try adjusting the learning rate
   - Increase the number of epochs

## License

If you're here, you're probably fine doing whatever you want with this repository. Good luck!