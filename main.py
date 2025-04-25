from train_models import train_and_save_models, evaluate_model
from test import test_model
import torch
from torch.utils.data import DataLoader
import numpy as np
from train_models import TrafficSignsDataset

def main():
    print("Step 1: Training models...")
    best_model_info = train_and_save_models()  # This will save the better performing model
    
    print("\nStep 2: Evaluating best model on validation set...")
    model = best_model_info['model']
    val_dataset = TrafficSignsDataset(
        np.transpose(best_model_info['data']['val']['images'], (0, 3, 1, 2)),
        best_model_info['data']['val']['labels']
    )
    val_loader = DataLoader(val_dataset, batch_size=32)
    val_accuracy, val_conf_matrix = evaluate_model(model, val_loader)
    print(f"\nValidation Results:")
    print(f"Model type: {best_model_info['model_type']}")
    print(f"Validation accuracy: {val_accuracy:.2f}%")
    
    print("\nStep 3: Testing best model on test dataset...")
    test_model()  # This will load the saved model and test it

if __name__ == "__main__":
    main()