import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from train_models import CNNMaxPool, CNNAvgPool, TrafficSignsDataset

def load_test_data(test_path='processed_data/test'):
    """Load test images and their labels"""
    images = []
    labels = []
    
    # Load label mapping from saved model info
    model_info = torch.load('model_info.pth')
    label_to_idx = model_info['label_mapping']
    
    print("\nLoading test data...")
    class_folders = os.listdir(test_path)
    for class_folder in tqdm(class_folders):
        class_path = os.path.join(test_path, class_folder)
        if not os.path.isdir(class_path):
            continue
            
        try:
            original_label = int(class_folder)
            class_label = label_to_idx[original_label]
        except (ValueError, KeyError) as e:
            print(f"Skipping folder {class_folder}: {str(e)}")
            continue
        
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = np.array(Image.open(img_path)) / 255.0
            images.append(img)
            labels.append(class_label)
    
    return np.array(images), np.array(labels)

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model info and state
    model_info = torch.load('model_info.pth')
    model_state = torch.load('best_model.pth')
    
    # Create appropriate model instance
    if model_info['model_type'] == 'maxpool':
        model = CNNMaxPool(model_info['input_shape'], model_info['num_classes'])
    else:
        model = CNNAvgPool(model_info['input_shape'], model_info['num_classes'])
    
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    # Load and prepare test data
    test_images, test_labels = load_test_data()
    test_images = np.transpose(test_images, (0, 3, 1, 2))
    test_dataset = TrafficSignsDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Test the model
    correct = 0
    total = 0
    predictions = []
    
    print("\nTesting model...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
    
    accuracy = 100. * correct / total
    print(f"\nTest Results:")
    print(f"Model type: {model_info['model_type']}")
    print(f"Total test images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Print per-class results
    class_correct = {}
    class_total = {}
    for pred, label in zip(predictions, test_labels):
        if label not in class_total:
            class_total[label] = 0
            class_correct[label] = 0
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    print("\nPer-class accuracy:")
    for label in sorted(class_total.keys()):
        class_acc = 100. * class_correct[label] / class_total[label]
        original_label = [k for k, v in model_info['label_mapping'].items() if v == label][0]
        print(f"Original class {original_label} (mapped to {label}): {class_acc:.2f}% ({class_correct[label]}/{class_total[label]})")

if __name__ == "__main__":
    test_model()