"""
Training script using data generators (memory efficient)
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint,
    TensorBoard,
    CSVLogger
)

import config


def create_data_generators(train_dir, val_dir=None, val_split=0.2, batch_size=32):
    """
    Create data generators with augmentation
    
    Args:
        train_dir: Training directory
        val_dir: Validation directory (optional)
        val_split: Validation split if val_dir not provided
        batch_size: Batch size
        
    Returns:
        train_generator, val_generator, num_classes
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=val_split if val_dir is None else 0
    )
    
    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create training generator
    if val_dir is None:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=config.IMAGE_SIZE,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=config.IMAGE_SIZE,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
    else:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=config.IMAGE_SIZE,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=config.IMAGE_SIZE,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    
    num_classes = len(train_generator.class_indices)
    
    return train_generator, val_generator, num_classes


def build_model(num_classes, model_type='mobilenet'):
    """
    Build model with transfer learning
    
    Args:
        num_classes: Number of output classes
        model_type: 'mobilenet' or 'resnet50'
        
    Returns:
        Compiled model
    """
    input_shape = (*config.IMAGE_SIZE, 3)
    
    # Load pre-trained base model
    if model_type == 'mobilenet':
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        print("Using MobileNetV2 as base model")
    elif model_type == 'resnet50':
        base_model = ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        print("Using ResNet50 as base model")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Freeze base model
    base_model.trainable = False
    
    # Build complete model
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    print(f"\nModel Summary:")
    print(f"  Total parameters: {model.count_params():,}")
    
    return model


def get_callbacks(model_name):
    """
    Get training callbacks
    
    Args:
        model_name: Name for saving model
        
    Returns:
        List of callbacks
    """
    checkpoint_dir = os.path.join(config.MODEL_DIR, 'checkpoints')
    log_dir = os.path.join(config.RESULTS_DIR, 'logs', model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1
        ),
        # Save full model every epoch to enable resume (includes optimizer state)
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_epoch{{epoch:02d}}_val{{val_accuracy:.4f}}.keras"),
            monitor='val_accuracy',
            save_best_only=False,
            save_weights_only=False,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        ),
        # CSV logger to keep epoch-by-epoch log
        CSVLogger(os.path.join(log_dir, 'training.csv'), append=True)
    ]
    
    return callbacks


def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history: Training history
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
    
    # Learning rate
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
    
    plt.close()


def _find_latest_checkpoint():
    """
    Find latest .keras checkpoint in the checkpoints directory.
    """
    checkpoint_dir = os.path.join(config.MODEL_DIR, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return None
    files = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith('.keras') or f.endswith('.h5')
    ]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _parse_initial_epoch_from_filename(filepath: str) -> int:
    """
    Parse epoch number from checkpoint filename pattern *_epochXX_val*.keras
    Returns 0 if not parsed.
    """
    base = os.path.basename(filepath)
    try:
        # e.g., model_epoch07_val0.9123.keras
        part = base.split('_epoch')[-1]
        epoch_str = part.split('_')[0]
        return int(epoch_str)
    except Exception:
        return 0


def main(args):
    """
    Main training function
    """
    print("="*60)
    print("Food Classification Training (Memory Efficient)")
    print("="*60)
    
    # Create data generators
    print("\n[1/4] Creating data generators...")
    train_generator, val_generator, num_classes = create_data_generators(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        val_split=args.val_split,
        batch_size=args.batch_size
    )
    
    print(f"\nDataset Info:")
    print(f"  Classes: {num_classes}")
    print(f"  Training samples: {train_generator.samples}")
    print(f"  Validation samples: {val_generator.samples}")
    print(f"  Batch size: {args.batch_size}")
    
    # Build or resume model
    model = None
    model_name = None
    initial_epoch = 0
    if args.resume or args.resume_from is not None:
        # Determine checkpoint path
        ckpt_path = args.resume_from if args.resume_from is not None else _find_latest_checkpoint()
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"\n[2/4] Resuming from checkpoint: {ckpt_path}")
            try:
                if ckpt_path.endswith('.keras'):
                    # New-style checkpoint: load full model (includes optimizer state)
                    model = keras.models.load_model(ckpt_path)
                else:
                    # Legacy .h5: build fresh model and load weights only
                    print("Detected legacy .h5 checkpoint. Building model and loading weights...")
                    tmp_model = build_model(num_classes, model_type=args.model_type)
                    tmp_model.load_weights(ckpt_path)
                    model = tmp_model
            except Exception as e:
                print(f"Warning: Could not load compiled model due to: {e}\nBuilding fresh model and loading weights as fallback...")
                tmp_model = build_model(num_classes, model_type=args.model_type)
                try:
                    tmp_model.load_weights(ckpt_path)
                    model = tmp_model
                except Exception as e2:
                    print(f"Fallback load_weights also failed: {e2}\nStarting fresh training.")
                    model = None
            # Derive model_name from checkpoint filename (exclude epoch/val suffix)
            base = os.path.splitext(os.path.basename(ckpt_path))[0]
            # strip trailing _epochXX_valYYYY pattern
            if '_epoch' in base:
                model_name = base.split('_epoch')[0]
            else:
                model_name = base
            # Continue from next epoch
            parsed_epoch = _parse_initial_epoch_from_filename(ckpt_path)
            initial_epoch = parsed_epoch if parsed_epoch > 0 else args.initial_epoch
            print(f"Resuming at initial_epoch={initial_epoch}")
        else:
            print("No valid checkpoint found to resume. Starting fresh training.")
    if model is None:
        print(f"\n[2/4] Building {args.model_type} model...")
        model = build_model(num_classes, model_type=args.model_type)
    
    # Get callbacks
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{args.model_type}_{timestamp}"
    callbacks = get_callbacks(model_name)
    
    # Train model
    print(f"\n[3/4] Training model...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Steps per epoch: {train_generator.samples // args.batch_size}")
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model and results
    print(f"\n[4/4] Saving model and results...")
    
    # Save model
    model_path = os.path.join(config.MODEL_DIR, f"{model_name}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save class names
    class_names = list(train_generator.class_indices.keys())
    class_names_path = os.path.join(config.MODEL_DIR, f"{model_name}_classes.json")
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=4)
    print(f"Class names saved to {class_names_path}")
    
    # Plot training history
    plot_path = os.path.join(config.RESULTS_DIR, f"{model_name}_history.png")
    plot_training_history(history, save_path=plot_path)
    
    # Save training info
    info = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': args.model_type,
        'num_classes': num_classes,
        'train_samples': train_generator.samples,
        'val_samples': val_generator.samples,
        'epochs_trained': len(history.history['loss']),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'config': {
            'batch_size': args.batch_size,
            'learning_rate': config.LEARNING_RATE,
            'epochs': args.epochs
        }
    }
    
    info_path = os.path.join(config.RESULTS_DIR, f"{model_name}_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    print(f"Training info saved to {info_path}")
    
    # Print final results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Classes: {num_classes}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train food classification model (memory efficient)')
    
    parser.add_argument('--train_dir', type=str, default=config.TRAIN_DIR,
                       help='Training data directory')
    parser.add_argument('--val_dir', type=str, default=config.VAL_DIR,
                       help='Validation data directory')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split if val_dir not provided')
    parser.add_argument('--model_type', type=str, default='mobilenet',
                       choices=['mobilenet', 'resnet50'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a specific checkpoint (.keras) to resume from')
    parser.add_argument('--initial_epoch', type=int, default=0,
                        help='Override initial epoch when resuming')
    
    args = parser.parse_args()
    main(args)
