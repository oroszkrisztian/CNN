# Traffic Sign Recognition System Documentation

## Overview
This project implements a Convolutional Neural Network (CNN) based traffic sign recognition system. The system consists of three main components:
1. Data Preparation (`prepare_data.py`)
2. Model Training (`train_models.py`)
3. Model Evaluation (`evaluate_models.py`, `test_model.py`, `gui.py`)

## Data Preparation (prepare_data.py)
### Purpose
Processes raw traffic sign images and prepares them for model training by:
- Organizing images into train/validation/test splits
- Resizing images to uniform dimensions (32x32)
- Normalizing pixel values (0-1)
- Creating class-based directory structure

### Key Functions
- `create_directory_structure()`: Creates necessary directories for processed data
- `process_image()`: Resizes and normalizes individual images
- `split_data()`: Splits data into train (70%), validation (15%), and test (15%) sets
- `process_dataset()`: Main function orchestrating the entire preparation process

### Usage
```python
python prepare_data.py
```

## Model Training (train_models.py)
### Architecture
Implements two CNN variants:
1. **CNNMaxPool**:
   - 3 Convolutional layers (16->32->64 filters)
   - Batch Normalization after each conv layer
   - MaxPooling layers
   - Dropout for regularization
   - Two fully connected layers (256->128->num_classes)

2. **CNNAvgPool**:
   - Similar architecture but uses Average Pooling instead of Max Pooling

### Dataset Handling
- `TrafficSignsDataset` class: Custom PyTorch Dataset implementation
- `load_processed_data()`: Loads and organizes processed data for training

### Training Process
- Uses Adam optimizer with learning rate 0.00005 and weight decay 0.01
- Cross-Entropy Loss function
- Early stopping with patience=10
- Batch size of 32
- Saves best performing model

### Key Functions
- `train_model()`: Handles the training loop and early stopping
- `evaluate_model()`: Evaluates model performance on test set
- `plot_training_history()`: Visualizes training and validation metrics

### Usage
```python
python train_models.py
```

## Model Evaluation

### evaluate_models.py
#### Purpose
Evaluates the saved model's performance on validation data.

#### Key Functions
- `calculate_metrics_per_class()`: Calculates detailed metrics per class
- `plot_confusion_matrix()`: Generates confusion matrix visualization
- `save_metrics_to_excel()`: Exports detailed metrics to Excel

### test_model.py
#### Purpose
Tests the model on individual images.

#### Key Functions
- `preprocess_image()`: Prepares single images for prediction
- `load_model()`: Loads trained model and configurations

### gui.py
#### Purpose
Provides a graphical interface for model testing.

#### Features
- Image upload and preview
- Confidence threshold adjustment
- Real-time predictions
- Clear visual feedback

#### Key Components
- `TrafficSignGUI`: Main GUI class
- Image preprocessing and prediction functionality
- Error handling and logging

### Metrics Generated
- Accuracy (ACC)
- Precision (PPV)
- Recall (TPR)
- Specificity (TNR)
- Negative Predictive Value (NPV)
- Dice Score (DS)

### Usage
```python
# For basic evaluation
python evaluate_models.py

# For testing single images
python test_model.py

# For GUI interface
python gui.py
```

## Model Files
- `best_model.pth`: Saved model weights
- `model_info.pth`: Model configuration and metadata

## Requirements
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PIL
- tkinter (for GUI)