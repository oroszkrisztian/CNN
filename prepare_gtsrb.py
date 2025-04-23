import os
import shutil
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import random

def get_available_classes(base_path='gtsrb/Final_Training/Images'):
    """Get all available class IDs from the dataset"""
    return [int(d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

# Randomly select 10 classes
available_classes = get_available_classes()
SELECTED_CLASSES = sorted(random.sample(available_classes, 10))
print(f"Randomly selected classes: {SELECTED_CLASSES}")
IMAGES_PER_CLASS = 5000

def augment_image(img):
    """Apply random augmentation to the image"""
    augmentations = [
        lambda x: ImageEnhance.Brightness(x).enhance(np.random.uniform(0.8, 1.2)),
        lambda x: ImageEnhance.Contrast(x).enhance(np.random.uniform(0.8, 1.2)),
        lambda x: x.rotate(np.random.uniform(-15, 15)),
        lambda x: ImageOps.mirror(x) if np.random.random() > 0.5 else x
    ]
    
    img = img.copy()
    for aug in np.random.choice(augmentations, size=np.random.randint(1, 3)):
        img = aug(img)
    return img

def load_gtsrb_data(base_path='gtsrb/Final_Training/Images'):
    images = []
    labels = []
    
    # Loop through only selected classes
    for class_folder in sorted(os.listdir(base_path)):
        class_label = int(class_folder)
        if class_label not in SELECTED_CLASSES:
            continue
            
        class_path = os.path.join(base_path, class_folder)
        if not os.path.isdir(class_path):
            continue
        
        # Read all images in the class folder
        for image_file in os.listdir(class_path):
            if image_file.endswith('.ppm'):
                image_path = os.path.join(class_path, image_file)
                images.append(image_path)
                labels.append(class_label)
    
    return images, labels

def create_directory_structure(base_dir='processed_data'):
    """Create train/val/test directory structure"""
    splits = ['train', 'val', 'test']
    for split in splits:
        for class_id in SELECTED_CLASSES:
            path = os.path.join(base_dir, split, str(class_id))
            os.makedirs(path, exist_ok=True)

def preprocess_image(image_path, target_size=(64, 64)):
    """Preprocess a single image"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize image
            img = img.resize(target_size, Image.LANCZOS)
            return img
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def copy_and_preprocess_image(src_path, dst_path, target_size=(64, 64)):
    """Preprocess and save image to destination"""
    img = preprocess_image(src_path, target_size)
    if img is not None:
        img.save(dst_path, 'PNG')

def visualize_dataset_stats(images, labels):
    """Visualize sample images and dataset statistics"""
    # Get image dimensions
    dimensions = []
    for img_path in images:
        with Image.open(img_path) as img:
            dimensions.append(img.size)
    
    # Create figure for samples
    plt.figure(figsize=(15, 8))
    plt.suptitle("Random Samples from Each Class", fontsize=16)
    
    # Group images by class
    class_images = {}
    for img_path, label in zip(images, labels):
        if label not in class_images:
            class_images[label] = []
        class_images[label].append(img_path)
    
    # Show one random sample from each class
    for i, class_id in enumerate(SELECTED_CLASSES):
        if class_id in class_images:
            sample_path = random.choice(class_images[class_id])
            plt.subplot(2, 5, i + 1)
            img = Image.open(sample_path)
            plt.imshow(img)
            plt.title(f"Class {class_id}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Plot dimension distribution
    plt.figure(figsize=(10, 5))
    widths, heights = zip(*dimensions)
    
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=30)
    plt.title('Image Width Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=30)
    plt.title('Image Height Distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    # Print class distribution
    print("\nClass Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"Class {class_id}: {count} images")

def visualize_split_distribution(splits):
    """Visualize distribution of images across splits and show sample images"""
    plt.figure(figsize=(15, 5))
    
    # Create subplots for each split
    for idx, (split_name, (X, y)) in enumerate(splits.items(), 1):
        plt.subplot(1, 3, idx)
        unique, counts = np.unique(y, return_counts=True)
        plt.bar([str(c) for c in unique], counts)
        plt.title(f'{split_name} Split Distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Show sample images from each split
    plt.figure(figsize=(15, 15))
    plt.suptitle("Random Samples from Each Class in Different Splits", fontsize=16)
    
    for split_idx, (split_name, (X, y)) in enumerate(splits.items()):
        # Group images by class
        class_images = {}
        for img_path, label in zip(X, y):
            if label not in class_images:
                class_images[label] = []
            class_images[label].append(img_path)
        
        # Show one random sample from each class
        for class_idx, class_id in enumerate(SELECTED_CLASSES):
            if class_id in class_images:
                sample_path = random.choice(class_images[class_id])
                plt.subplot(3, 10, split_idx * 10 + class_idx + 1)
                img = Image.open(sample_path)
                plt.imshow(img)
                plt.title(f"{split_name}\nClass {class_id}")
                plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def log_split_stats(processed_splits, processed_labels):
    """Log statistics about the processed data splits"""
    print("\nProcessed data statistics:")
    for split_name in processed_splits.keys():
        print(f"\n{split_name.upper()} split:")
        unique, counts = np.unique(processed_labels[split_name], return_counts=True)
        for class_id, count in zip(unique, counts):
            print(f"Class {class_id}: {count} images")

def augment_class_to_target(images, labels, class_id, target_count):
    """Augment images of a specific class until reaching target count"""
    class_images = [img for img, lbl in zip(images, labels) if lbl == class_id]
    augmented_images = []
    augmented_labels = []
    
    # First add all original images
    augmented_images.extend(class_images)
    augmented_labels.extend([class_id] * len(class_images))
    
    # Calculate how many augmented images we need per original image
    needed = target_count - len(class_images)
    if needed <= 0:
        return augmented_images, augmented_labels
    
    augs_per_image = needed // len(class_images) + 1
    
    # Generate augmented images
    for img_path in class_images:
        img = preprocess_image(img_path)
        if img is not None:
            for _ in range(augs_per_image):
                if len(augmented_images) >= target_count:
                    break
                aug_img = augment_image(img)
                # Save augmented image
                base_filename = os.path.splitext(os.path.basename(img_path))[0]
                aug_path = os.path.join('augmented_data', str(class_id), f"{base_filename}_aug_{len(augmented_images)}.png")
                os.makedirs(os.path.dirname(aug_path), exist_ok=True)
                aug_img.save(aug_path)
                augmented_images.append(aug_path)
                augmented_labels.append(class_id)
                
    return augmented_images[:target_count], augmented_labels[:target_count]

def main():
    # Load all image paths and labels
    print("Loading GTSRB dataset...")
    images, labels = load_gtsrb_data()
    
    # Show raw data distribution
    print("\nRaw data distribution before augmentation:")
    initial_data = {'raw': (images, labels)}
    visualize_split_distribution(initial_data)
    
    # Augment data to reach target count for each class
    print("\nAugmenting data to reach 5000 images per class...")
    balanced_images = []
    balanced_labels = []
    
    for class_id in SELECTED_CLASSES:
        print(f"Processing class {class_id}...")
        class_images, class_labels = augment_class_to_target(
            images, labels, class_id, IMAGES_PER_CLASS
        )
        balanced_images.extend(class_images)
        balanced_labels.extend(class_labels)
    
    # Show balanced data distribution
    print("\nData distribution after balancing:")
    balanced_data = {'balanced': (balanced_images, balanced_labels)}
    visualize_split_distribution(balanced_data)
    
    # Split balanced data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        balanced_images, balanced_labels, test_size=0.3, random_state=42, stratify=balanced_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    splits = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    
    # Create directory structure
    print("\nCreating directory structure...")
    create_directory_structure()
    
    # Process and copy images to final locations
    processed_splits = {'train': [], 'val': [], 'test': []}
    processed_labels = {'train': [], 'val': [], 'test': []}
    
    for split_name, (X, y) in splits.items():
        print(f"Processing {split_name} split...")
        for img_path, label in zip(X, y):
            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join('processed_data', split_name, str(label), f"{base_filename}.png")
            
            # If it's an original image, process it; if augmented, just copy
            if 'augmented_data' not in img_path:
                img = preprocess_image(img_path)
                if img is not None:
                    img.save(save_path, 'PNG')
            else:
                shutil.copy2(img_path, save_path)
                
            processed_splits[split_name].append(save_path)
            processed_labels[split_name].append(label)
    
    # Show final distribution and log statistics
    print("\nFinal data distribution after processing:")
    final_splits = {
        name: (paths, labels) 
        for name, (paths, labels) in zip(
            processed_splits.keys(),
            [(processed_splits[k], processed_labels[k]) for k in processed_splits.keys()]
        )
    }
    
    log_split_stats(processed_splits, processed_labels)
    visualize_split_distribution(final_splits)
    
    print("Data preparation completed!")

if __name__ == "__main__":
    main()