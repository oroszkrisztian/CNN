import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import joblib
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# GPU Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TrafficSignsDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def load_processed_data(base_path='processed_data'):
    """Load and prepare the processed and split dataset"""
    splits = ['train', 'val', 'test']
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

class CNNMaxPool(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNMaxPool, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, 3, padding=1),  # Reduced from 16
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Increased dropout
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1),  # Reduced from 32
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),  # Reduced from 64
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2)
        )
        
        self.flat_features = self._get_flat_features(input_shape)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 128),  # Reduced from 256
            nn.ReLU(),
            nn.Dropout(0.6),  # Increased dropout
            nn.Linear(128, 64),  # Reduced from 128
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def _get_flat_features(self, input_shape):
        # Helper function to calculate flattened size
        dummy_input = torch.zeros(1, *input_shape)
        output = self.conv_layers(dummy_input)
        return int(np.prod(output.shape[1:]))
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.flat_features)
        x = self.fc_layers(x)
        return x

class CNNAvgPool(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNAvgPool, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.AvgPool2d(2, 2)
        )
        
        self.flat_features = self._get_flat_features(input_shape)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, patience=10):
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
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
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                model.load_state_dict(best_state)
                break
    
    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }

def plot_training_history(history1, history2):
    """Plot training histories for comparison"""
    metrics = [('loss', 'Loss'), ('acc', 'Accuracy')]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for idx, (metric, title) in enumerate(metrics):
        axes[idx].plot(history1[f'train_{metric}'], label=f'MaxPool train')
        axes[idx].plot(history1[f'val_{metric}'], label=f'MaxPool val')
        axes[idx].plot(history2[f'train_{metric}'], label=f'AvgPool train')
        axes[idx].plot(history2[f'val_{metric}'], label=f'AvgPool val')
        axes[idx].set_title(f'Model {title}')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(title)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return accuracy, conf_matrix

def main():
    # Load data
    print("Loading processed data...")
    data = load_processed_data()
    
    # Prepare datasets and dataloaders
    batch_size = 32
    train_dataset = TrafficSignsDataset(
        np.transpose(data['train']['images'], (0, 3, 1, 2)),
        data['train']['labels']
    )
    val_dataset = TrafficSignsDataset(
        np.transpose(data['val']['images'], (0, 3, 1, 2)),
        data['val']['labels']
    )
    test_dataset = TrafficSignsDataset(
        np.transpose(data['test']['images'], (0, 3, 1, 2)),
        data['test']['labels']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create models
    input_shape = (3, data['train']['images'].shape[1], data['train']['images'].shape[2])
    num_classes = len(np.unique(data['train']['labels']))
    
    print("Creating models...")
    model_maxpool = CNNMaxPool(input_shape, num_classes)
    model_avgpool = CNNAvgPool(input_shape, num_classes)
    
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer_maxpool = optim.Adam(model_maxpool.parameters(), lr=0.00005, weight_decay=0.01)  # Reduced lr, increased weight decay
    optimizer_avgpool = optim.Adam(model_avgpool.parameters(), lr=0.00005, weight_decay=0.01)
    
    # Train models
    print("\nTraining MaxPooling model...")
    history_maxpool = train_model(
        model_maxpool, train_loader, val_loader,
        criterion, optimizer_maxpool,
        num_epochs=30
    )
    
    print("\nTraining AveragePooling model...")
    history_avgpool = train_model(
        model_avgpool, train_loader, val_loader,
        criterion, optimizer_avgpool,
        num_epochs=30
    )
    
    # Plot training histories
    plot_training_history(history_maxpool, history_avgpool)
    
    # Evaluate models
    print("\nModel Evaluation:")
    acc_maxpool, conf_maxpool = evaluate_model(model_maxpool, test_loader)
    print(f"MaxPooling Model - Test accuracy: {acc_maxpool:.2f}%")
    
    acc_avgpool, conf_avgpool = evaluate_model(model_avgpool, test_loader)
    print(f"AveragePooling Model - Test accuracy: {acc_avgpool:.2f}%")
    
    # Save only the better model
    if acc_maxpool > acc_avgpool:
        better_model = model_maxpool
        model_type = 'maxpool'
        conf_matrix = conf_maxpool
        accuracy = acc_maxpool
    else:
        better_model = model_avgpool
        model_type = 'avgpool'
        conf_matrix = conf_avgpool
        accuracy = acc_avgpool
    
    print(f"\nSaving the better model ({model_type})...")
    torch.save(better_model.state_dict(), 'best_model.pth')
    
    # Get original label mapping
    all_labels = set()
    label_to_idx = {}
    for split in ['train', 'val', 'test']:
        split_path = os.path.join('processed_data', split)
        for class_folder in os.listdir(split_path):
            try:
                label = int(class_folder)
                all_labels.add(label)
            except ValueError:
                continue
    
    # Create mapping from original to model indices
    original_labels = sorted(all_labels)
    label_to_idx = {label: idx for idx, label in enumerate(original_labels)}
    
    # Save model info with both original labels and their mapping
    model_info = {
        'model_type': model_type,
        'input_shape': (3, data['train']['images'].shape[1], data['train']['images'].shape[2]),
        'num_classes': len(np.unique(data['train']['labels'])),
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'original_labels': original_labels,
        'label_mapping': label_to_idx
    }
    torch.save(model_info, 'model_info.pth')
    print("Model and info saved successfully!")

if __name__ == "__main__":
    main()