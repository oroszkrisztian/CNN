import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrafficSignsDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class OverfittingModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(OverfittingModel, self).__init__()
        # Excessive number of layers and parameters
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.flat_features = self._get_flat_features(input_shape)
        
        # Very large fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def _get_flat_features(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        output = self.conv_layers(dummy_input)
        return int(np.prod(output.shape[1:]))
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.flat_features)
        x = self.fc_layers(x)
        return x

def load_data(base_path='processed_data'):
    """Load and prepare the processed dataset"""
    splits = ['train', 'val']
    data = {}
    
    # First, find all unique labels across all splits
    all_labels = set()
    for split in splits:
        split_path = os.path.join(base_path, split)
        for class_folder in os.listdir(split_path):
            try:
                label = int(class_folder)
                all_labels.add(label)
            except ValueError:
                continue
    
    # Create label mapping
    label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
    print("\nLabel mapping:")
    for original, mapped in label_to_idx.items():
        print(f"Original label {original} -> New label {mapped}")
    
    print("\nLoading processed data...")
    for split in tqdm(splits, desc='Loading datasets'):
        images = []
        labels = []
        split_path = os.path.join(base_path, split)
        
        class_folders = os.listdir(split_path)
        for class_folder in tqdm(class_folders, desc=f'Loading {split} classes', leave=False):
            class_path = os.path.join(split_path, class_folder)
            try:
                original_label = int(class_folder)
                class_label = label_to_idx[original_label]  # Remap the label
            except ValueError as e:
                print(f"Error processing class folder {class_folder}: {str(e)}")
                continue
            
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                img = np.array(Image.open(img_path)) / 255.0
                images.append(img)
                labels.append(class_label)
        
        if not images:
            raise ValueError(f"No valid images found in {split} split")
        
        data[split] = {
            'images': np.array(images),
            'labels': np.array(labels)
        }
        
        # Validate labels
        unique_labels = np.unique(data[split]['labels'])
        print(f"\n{split} set - Unique labels found: {unique_labels}")
        print(f"Min label: {unique_labels.min()}, Max label: {unique_labels.max()}")
    
    return data

def train_model(model, train_loader, val_loader, num_epochs=100):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in tqdm(range(num_epochs), desc='Training epochs'):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Training batch', leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        val_pbar = tqdm(val_loader, desc=f'Validation batch', leave=False)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }

def plot_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('overfitting_curves.png')
    plt.show()

def main():
    print("Loading data...")
    data = load_data()
    
    # Prepare datasets
    batch_size = 32
    train_dataset = TrafficSignsDataset(
        np.transpose(data['train']['images'], (0, 3, 1, 2)),
        data['train']['labels']
    )
    val_dataset = TrafficSignsDataset(
        np.transpose(data['val']['images'], (0, 3, 1, 2)),
        data['val']['labels']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create and train the model
    input_shape = (3, data['train']['images'].shape[1], data['train']['images'].shape[2])
    num_classes = len(np.unique(data['train']['labels']))
    
    print("Creating model...")
    model = OverfittingModel(input_shape, num_classes)
    
    print("Training model...")
    history = train_model(model, train_loader, val_loader)
    
    print("Plotting results...")
    plot_training_curves(history)

if __name__ == "__main__":
    main()