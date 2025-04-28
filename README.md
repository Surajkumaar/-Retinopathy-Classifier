# Retinopathy-Classifier

# Diabetic Retinopathy Detection Project

This project implements a deep learning model for detecting and classifying Diabetic Retinopathy (DR) using PyTorch and ONNX runtime for efficient inference.

## Project Overview

This application uses a DenseNet121 model to classify retinal images into five DR severity grades: No DR, Mild, Moderate, Severe, and Proliferative DR. The model is trained using PyTorch and then converted to ONNX format for deployment.

## Files and Folders

This project contains the following files and folders:

### Main Files

- `app.py`: The main application file with web interface for DR detection.

- `train.py`: Script for training the DenseNet121 model on retinal images.

### Folders

- `Data Scripts/`: Contains utility scripts for data preprocessing:
  - `crop_square.py`: Script for cropping images to square format.
  - `csvname.py`: Utility for CSV file handling.
  - `filter_images.py`: Script to filter images based on criteria.
  - `filter_labels.py`: Script to filter labels in the dataset.
  - `normalization.py`: Script for normalizing image data.
  - `refine.py`: Script for refining the dataset.
  - `rename.py`: Utility for renaming files.
  - `resize.py`: Script for resizing images.
  - `split.py`: Script for splitting data into train/validation sets.

- `Model/`: Contains the trained model files:
  - `best_densenet121.ckpt`: The trained PyTorch model checkpoint.
  - `model_.onnx`: The converted ONNX model file used for inference.

- `Onnx conversion/`: Contains scripts for ONNX conversion:
  - `convert.py`: Script to convert the PyTorch model to ONNX format.

- `csv data/`: Contains CSV files with dataset information:
  - `train.csv`: Training dataset labels and metadata.
  - `val.csv`: Validation dataset labels and metadata.

## Usage

### Running the Application

To start the web application:

```bash
python app.py
```

### Training the Model

To train the model from scratch:

```bash
python train.py
```

### Converting to ONNX

To convert the trained PyTorch model to ONNX format:

```bash
python "Onnx conversion/convert.py"
```

## Model Information

The model is a DenseNet121 architecture pretrained on ImageNet and fine-tuned for DR classification. It was converted to ONNX format for faster inference and better deployment options.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- onnxruntime
- numpy
- PIL
- opencv-python (cv2)
- pandas
- scikit-learn
