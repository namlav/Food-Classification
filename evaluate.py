"""
Model evaluation and validation script
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score
)
import json
from tensorflow import keras

import config
from data_loader import DataLoader
from preprocessing import ImagePreprocessor


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_class_accuracy(y_true, y_pred, class_names, save_path=None):
    """
    Plot per-class accuracy
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), class_accuracy)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    
    # Color bars based on accuracy
    for i, bar in enumerate(bars):
        if class_accuracy[i] >= 0.9:
            bar.set_color('green')
        elif class_accuracy[i] >= 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class accuracy plot saved to {save_path}")
    
    plt.show()


def predict_in_batches(model, data_loader, test_dir, batch_size=32):
    """
    Make predictions in batches to avoid memory issues
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Create data generator for test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class names from generator
    class_names = [k for k, v in sorted(test_generator.class_indices.items(), key=lambda x: x[1])]
    
    # Make predictions
    print(f"Making predictions on {test_generator.samples} images...")
    y_pred_proba = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    return y_true, y_pred, y_pred_proba, class_names

def evaluate_model(model_path, test_dir=None, class_names_path=None, batch_size=32):
    """
    Evaluate trained model using batch processing
    
    Args:
        model_path: Path to saved model
        test_dir: Directory containing test data
        class_names_path: Path to class names JSON file
        batch_size: Batch size for prediction
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Load model
    print(f"\n[1/4] Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Load class names
    if class_names_path and os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        print(f"Loaded {len(class_names)} class names from file")
    else:
        class_names = None
        print("Class names not provided, will be extracted from test data")
    
    # Make predictions in batches
    print(f"\n[2/4] Loading test data from {test_dir}...")
    data_loader = DataLoader()
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found")
        return None
    
    # Make predictions using batch processing
    print(f"\n[3/4] Making predictions...")
    y_true, y_pred, y_pred_proba, loaded_class_names = predict_in_batches(
        model, data_loader, test_dir, batch_size=batch_size
    )
    
    # Use loaded class names if available, otherwise use from generator
    if class_names is None:
        class_names = loaded_class_names
    
    # Calculate metrics
    print(f"\n[4/4] Calculating metrics...")
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Top-3 and Top-5 accuracy
    from tensorflow.keras.utils import to_categorical
    y_test_cat = to_categorical(y_true, len(class_names))
    top3_acc = top_k_accuracy_score(y_true, y_pred_proba, k=min(3, len(class_names)))
    top5_acc = top_k_accuracy_score(y_true, y_pred_proba, k=min(5, len(class_names)))
    
    # Classification report
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Test Samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    
    # Generate classification report
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    print("\nPer-Class Metrics:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Create results directory if it doesn't exist
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Save confusion matrix (only for a subset of classes if too many)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    cm_path = os.path.join(config.RESULTS_DIR, f"{model_name}_confusion_matrix.png")
    
    # For large number of classes, only plot a subset for the confusion matrix
    if len(class_names) > 30:
        print("\nToo many classes for full confusion matrix. Plotting top 30 classes...")
        # Get top 30 classes by sample count
        from collections import Counter
        class_counts = Counter(y_true)
        top_classes = [c[0] for c in class_counts.most_common(30)]
        
        # Filter data for top classes
        mask = np.isin(y_true, top_classes)
        y_true_subset = y_true[mask]
        y_pred_subset = y_pred[mask]
        
        # Map to 0-29 for plotting
        class_to_idx = {c: i for i, c in enumerate(top_classes)}
        y_true_mapped = np.array([class_to_idx[c] for c in y_true_subset])
        y_pred_mapped = np.array([class_to_idx.get(c, 0) for c in y_pred_subset])
        
        # Use only the class names that are in top_classes
        class_names_subset = [class_names[i] for i in top_classes]
        
        # Plot subset confusion matrix
        plot_confusion_matrix(
            y_true_mapped, 
            y_pred_mapped, 
            class_names_subset, 
            save_path=cm_path
        )
    else:
        # Plot full confusion matrix
        plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path)
    
    # Plot class accuracy (only if not too many classes)
    if len(class_names) <= 50:
        acc_path = os.path.join(config.RESULTS_DIR, f"{model_name}_class_accuracy.png")
        plot_class_accuracy(y_true, y_pred, class_names, save_path=acc_path)
    else:
        print("Skipping class accuracy plot due to large number of classes")
    
    # Save results
    results = {
        'model_path': model_path,
        'test_samples': int(len(y_true)),
        'accuracy': float(accuracy),
        'top3_accuracy': float(top3_acc),
        'top5_accuracy': float(top5_acc),
        'classification_report': report
    }
    
    results_path = os.path.join(config.RESULTS_DIR, f"{model_name}_evaluation.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nEvaluation results saved to {results_path}")
    
    return results


def main(args):
    """
    Main evaluation function
    """
    # Find class names file if not provided
    if args.class_names is None:
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        class_names_path = os.path.join(config.MODEL_DIR, f"{model_name}_classes.json")
        if not os.path.exists(class_names_path):
            # Try to find any class names file
            class_files = [f for f in os.listdir(config.MODEL_DIR) if f.endswith('_classes.json')]
            if class_files:
                class_names_path = os.path.join(config.MODEL_DIR, class_files[0])
            else:
                class_names_path = None
    else:
        class_names_path = args.class_names
    
    # Evaluate model
    results = evaluate_model(
        model_path=args.model,
        test_dir=args.test_dir,
        class_names_path=class_names_path
    )
    
    if results:
        print("\n" + "="*60)
        print("Evaluation Complete!")
        print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate food classification model')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.h5 file)')
    parser.add_argument('--test_dir', type=str, default=config.TEST_DIR,
                       help='Test data directory')
    parser.add_argument('--class_names', type=str, default=None,
                       help='Path to class names JSON file')
    
    args = parser.parse_args()
    
    main(args)
