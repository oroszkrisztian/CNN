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
    splits = ['train', 'val']  # Remove 'test' from splits
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
            # First conv block
            nn.Conv2d(input_shape[0], 16, 3, padding=1),  # Increased filters from 8 to 16
            nn.BatchNorm2d(16),  # Added BatchNorm
            nn.LeakyReLU(0.1),  # Replaced ReLU with LeakyReLU
            nn.Dropout2d(0.6),  # Reduced dropout from 0.5
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(16, 32, 3, padding=1),  # Increased filters from 16 to 32
            nn.BatchNorm2d(32),  # Added BatchNorm
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(2, 2)
        )
        
        self.flat_features = self._get_flat_features(input_shape)
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_features, 128),
            nn.BatchNorm1d(128),  # Added BatchNorm
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),  # Reduced dropout from 0.5
            nn.Linear(128, num_classes)
        )
    
    def _get_flat_features(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        output = self.conv_layers(dummy_input)
        return int(np.prod(output.shape[1:]))
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CNNAvgPool(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNAvgPool, self).__init__()
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_shape[0], 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.6),
            nn.AvgPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.4),
            nn.AvgPool2d(2, 2)
        )
        
        self.flat_features = self._get_flat_features(input_shape)
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def _get_flat_features(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        output = self.conv_layers(dummy_input)
        return int(np.prod(output.shape[1:]))
    
    def forward(self, x):
        x = self.conv_layers(x)
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
    """Plot training histories for comparison and save as PNG"""
    # Plot Training and Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history1['train_acc'], label='MaxPool Training Acc')
    plt.plot(history1['val_acc'], label='MaxPool Validation Acc')
    plt.plot(history2['train_acc'], label='AvgPool Training Acc')
    plt.plot(history2['val_acc'], label='AvgPool Validation Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('accuracy_history.png')
    plt.close()

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history1['train_loss'], label='MaxPool Training Loss')
    plt.plot(history1['val_loss'], label='MaxPool Validation Loss')
    plt.plot(history2['train_loss'], label='AvgPool Training Loss')
    plt.plot(history2['val_loss'], label='AvgPool Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_history.png')
    plt.close()

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    print("\nDetailed evaluation:")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            batch_correct = predicted.eq(labels).sum().item()
            batch_total = labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += batch_total
            correct += batch_correct
            
            print(f"Batch {batch_idx + 1}:")
            print(f"  Correct predictions: {batch_correct}/{batch_total}")
            print(f"  Running accuracy: {100. * correct / total:.2f}%")
    
    accuracy = 100. * correct / total
    print(f"\nFinal test set size: {total} images")
    print(f"Total correct predictions: {correct}")
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return accuracy, conf_matrix

def save_model_h5(model, path):
    """Save PyTorch model in H5 format if h5py is available"""
    try:
        import h5py
        # Create H5 file
        with h5py.File(path, 'w') as f:
            for name, param in model.state_dict().items():
                f.create_dataset(name, data=param.cpu().numpy())
        print(f"Model saved in H5 format: {path}")
    except ImportError:
        print("Warning: h5py module not found. Model will only be saved in PyTorch format.")
        print("To save in H5 format, install h5py using: pip install h5py")

def plot_dataset_distribution(data):
    """Plot the number of samples in each dataset split and per class"""
    splits = ['train', 'val']  # Remove 'test' from splits
    
    # First plot: total samples per split
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sample_counts = [len(data[split]['images']) for split in splits]
    plt.bar(splits, sample_counts)
    plt.title('Total Samples in Each Dataset Split')
    plt.ylabel('Number of Samples')
    for i, count in enumerate(sample_counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    # Second plot: samples per class for each split
    plt.subplot(1, 2, 2)
    num_classes = len(np.unique(np.concatenate([data[split]['labels'] for split in splits])))
    x = np.arange(num_classes)
    width = 0.3  # Increased width since we only have 2 splits now
    
    for i, split in enumerate(splits):
        class_counts = np.bincount(data[split]['labels'], minlength=num_classes)
        plt.bar(x + i*width, class_counts, width, label=split)
    
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Samples per Class for Each Split')
    plt.legend()
    plt.xticks(x + width/2, range(num_classes))
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png')
    plt.close()
    
    print("\nDataset distribution:")
    for split in splits:
        print(f"\n{split.capitalize()} set:")
        class_counts = np.bincount(data[split]['labels'])
        for class_idx, count in enumerate(class_counts):
            print(f"  Class {class_idx}: {count} samples")

def plot_model_architecture(model, input_shape, filename):
    """Display text-based model architecture summary"""
    print(f"\nModel Architecture: {filename}")
    print("=" * 50)
    
    # Print model structure
    print(model)
    
    # Calculate and print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Summary:")
    print(f"Input shape: {input_shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 50)

def train_and_save_models():
    # Load data
    print("Loading processed data...")
    data = load_processed_data()
    
    # Plot dataset distribution for train and validation only
    plot_dataset_distribution({k: data[k] for k in ['train', 'val']})
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create models
    input_shape = (3, data['train']['images'].shape[1], data['train']['images'].shape[2])
    num_classes = len(np.unique(data['train']['labels']))
    
    print("Creating models...")
    model_maxpool = CNNMaxPool(input_shape, num_classes)
    model_avgpool = CNNAvgPool(input_shape, num_classes)
    
    # Visualize model architectures
    plot_model_architecture(model_maxpool, input_shape, "maxpool_architecture")
    plot_model_architecture(model_avgpool, input_shape, "avgpool_architecture")
    
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer_maxpool = optim.AdamW(model_maxpool.parameters(), lr=0.0001)
    optimizer_avgpool = optim.AdamW(model_avgpool.parameters(), lr=0.0001)
    
    # Train models
    print("\nTraining MaxPooling model...")
    history_maxpool = train_model(
        model_maxpool, train_loader, val_loader,
        criterion, optimizer_maxpool,
        num_epochs=30,
        patience=10
    )
    
    print("\nTraining AveragePooling model...")
    history_avgpool = train_model(
        model_avgpool, train_loader, val_loader,
        criterion, optimizer_avgpool,
        num_epochs=30,
        patience=10
    )
    
    # Plot training histories
    plot_training_history(history_maxpool, history_avgpool)
    
    # Compare validation accuracies
    maxpool_val_acc = max(history_maxpool['val_acc'])
    avgpool_val_acc = max(history_avgpool['val_acc'])
    
    # Save only the better model based on validation accuracy
    if (maxpool_val_acc > avgpool_val_acc):
        better_model = model_maxpool
        model_type = 'maxpool'
        best_val_acc = maxpool_val_acc
    else:
        better_model = model_avgpool
        model_type = 'avgpool'
        best_val_acc = avgpool_val_acc
    
    print(f"\nSaving the better model ({model_type}) with validation accuracy: {best_val_acc:.2f}%")
    # Always save in PyTorch format
    torch.save(better_model.state_dict(), 'best_model.pth')
    print("Model saved in PyTorch format: best_model.pth")
    
    # Try to save in H5 format
    try:
        save_model_h5(better_model, 'best_model.h5')
    except Exception as e:
        print(f"Failed to save in H5 format: {str(e)}")
    
    # Get original label mapping
    all_labels = set()
    label_to_idx = {}
    for split in ['train', 'val']:
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
    
    # Save model info
    model_info = {
        'model_type': model_type,
        'input_shape': input_shape,
        'num_classes': num_classes,
        'validation_accuracy': best_val_acc,
        'original_labels': original_labels,
        'label_mapping': label_to_idx
    }
    torch.save(model_info, 'model_info.pth')
    print("Model and info saved successfully!")
    
    return {
        'model': better_model,
        'data': {'train': data['train'], 'val': data['val']},
        'model_type': model_type,
        'validation_accuracy': best_val_acc
    }

if __name__ == "__main__":
    train_and_save_models()