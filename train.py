"""
Training script for food classification model
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

import config
from data_loader import DataLoader
from model import FoodClassifier


def plot_training_history(history, save_path: str = None):
    """
    Plot training history
    
    Args:
        history: Training history object
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Top-3 Accuracy
    if 'top_3_accuracy' in history.history:
        axes[1, 0].plot(history.history['top_3_accuracy'], label='Train Top-3')
        axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Val Top-3')
        axes[1, 0].set_title('Top-3 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def save_training_info(data_info: dict, model_type: str, history, save_dir: str):
    """
    Save training information to JSON
    
    Args:
        data_info: Dictionary with data information
        model_type: Type of model used
        history: Training history
        save_dir: Directory to save info
    """
    info = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': model_type,
        'num_classes': data_info['num_classes'],
        'class_names': data_info['class_names'],
        'train_samples': len(data_info['X_train']),
        'val_samples': len(data_info['X_val']),
        'image_size': config.IMAGE_SIZE,
        'epochs_trained': len(history.history['loss']),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'epochs': config.EPOCHS
        }
    }
    
    filepath = os.path.join(save_dir, 'training_info.json')
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"Training info saved to {filepath}")
    return info


def main(args):
    """
    Main training function
    """
    print("="*60)
    print("Food Classification Training")
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    
    # Load data
    print("\n[1/5] Loading dataset...")
    data_loader = DataLoader()
    
    try:
        data_info = data_loader.prepare_data(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            test_split=args.val_split
        )
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure your data is organized as follows:")
        print("data/")
        print("  train/")
        print("    class1/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("    class2/")
        print("      image1.jpg")
        print("      ...")
        return
    
    # Create model
    print(f"\n[2/5] Building {args.model_type} model...")
    classifier = FoodClassifier(
        num_classes=data_info['num_classes'],
        model_type=args.model_type
    )
    classifier.build_model(trainable_base=False)
    
    # Train model
    print(f"\n[3/5] Training model...")
    history = classifier.train(
        X_train=data_info['X_train'],
        y_train=data_info['y_train'],
        X_val=data_info['X_val'],
        y_val=data_info['y_val'],
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Fine-tune if requested
    if args.fine_tune:
        print(f"\n[4/5] Fine-tuning model...")
        history_fine = classifier.fine_tune(
            X_train=data_info['X_train'],
            y_train=data_info['y_train'],
            X_val=data_info['X_val'],
            y_val=data_info['y_val'],
            epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            unfreeze_layers=args.unfreeze_layers
        )
    else:
        print(f"\n[4/5] Skipping fine-tuning...")
    
    # Save model
    print(f"\n[5/5] Saving model and results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.model_type}_{timestamp}"
    
    # Save model
    model_path = os.path.join(config.MODEL_DIR, f"{model_name}.h5")
    classifier.save_model(model_path)
    
    # Save class names
    class_names_path = os.path.join(config.MODEL_DIR, f"{model_name}_classes.json")
    with open(class_names_path, 'w') as f:
        json.dump(data_info['class_names'], f, indent=4)
    print(f"Class names saved to {class_names_path}")
    
    # Plot and save training history
    plot_path = os.path.join(config.RESULTS_DIR, f"{model_name}_history.png")
    plot_training_history(history, save_path=plot_path)
    
    # Save training info
    save_training_info(data_info, args.model_type, history, config.RESULTS_DIR)
    
    # Print final results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {model_path}")
    print(f"Final validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final validation loss: {min(history.history['val_loss']):.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train food classification model')
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, default=config.TRAIN_DIR,
                       help='Training data directory')
    parser.add_argument('--val_dir', type=str, default=None,
                       help='Validation data directory (optional)')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio if val_dir not provided')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default=config.MODEL_TYPE,
                       choices=['mobilenet', 'resnet50'],
                       help='Model architecture to use')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    
    # Fine-tuning arguments
    parser.add_argument('--fine_tune', action='store_true',
                       help='Perform fine-tuning after initial training')
    parser.add_argument('--fine_tune_epochs', type=int, default=20,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--unfreeze_layers', type=int, default=20,
                       help='Number of layers to unfreeze for fine-tuning')
    
    args = parser.parse_args()
    
    main(args)
