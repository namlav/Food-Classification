"""
Model architecture using Transfer Learning
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint,
    TensorBoard
)
import os
from datetime import datetime
import config


class FoodClassifier:
    """
    Food classifier using transfer learning
    """
    
    def __init__(
        self, 
        num_classes: int,
        model_type: str = config.MODEL_TYPE,
        input_shape: tuple = (*config.IMAGE_SIZE, 3)
    ):
        """
        Initialize classifier
        
        Args:
            num_classes: Number of output classes
            model_type: 'mobilenet' or 'resnet50'
            input_shape: Input image shape (height, width, channels)
        """
        self.num_classes = num_classes
        self.model_type = model_type.lower()
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self, trainable_base: bool = False) -> keras.Model:
        """
        Build model with transfer learning
        
        Args:
            trainable_base: Whether to make base model trainable
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained base model
        if self.model_type == 'mobilenet':
            base_model = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
            print("Using MobileNetV2 as base model")
        elif self.model_type == 'resnet50':
            base_model = ResNet50(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
            print("Using ResNet50 as base model")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Freeze base model
        base_model.trainable = trainable_base
        
        # Build complete model
        inputs = keras.Input(shape=self.input_shape)
        
        # Preprocessing for the base model
        x = inputs
        
        # Base model
        x = base_model(x, training=False)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        self.model = model
        
        print(f"\nModel Summary:")
        print(f"  Total parameters: {model.count_params():,}")
        print(f"  Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return model
    
    def get_callbacks(self, model_name: str = None) -> list:
        """
        Get training callbacks
        
        Args:
            model_name: Name for saving model checkpoints
            
        Returns:
            List of callbacks
        """
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model_type}_{timestamp}"
        
        # Create directories
        checkpoint_dir = os.path.join(config.MODEL_DIR, 'checkpoints')
        log_dir = os.path.join(config.RESULTS_DIR, 'logs', model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=config.MIN_LR,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f'{model_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs: int = config.EPOCHS,
        batch_size: int = config.BATCH_SIZE,
        callbacks: list = None
    ):
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            X_val: Validation images
            y_val: Validation labels (one-hot encoded)
            epochs: Number of epochs
            batch_size: Batch size
            callbacks: List of callbacks (if None, default callbacks are used)
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        print(f"\nStarting training...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def fine_tune(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs: int = 20,
        batch_size: int = config.BATCH_SIZE,
        unfreeze_layers: int = 20
    ):
        """
        Fine-tune the model by unfreezing some layers
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of fine-tuning epochs
            batch_size: Batch size
            unfreeze_layers: Number of layers to unfreeze from the end
        """
        if self.model is None:
            raise ValueError("Model must be trained before fine-tuning")
        
        # Unfreeze the base model
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        # Freeze all layers except the last unfreeze_layers
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        print(f"\nFine-tuning model...")
        print(f"  Unfrozen layers: {unfreeze_layers}")
        print(f"  Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
        # Continue training
        callbacks = self.get_callbacks(model_name=f"{self.model_type}_finetuned")
        
        history_fine = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history_fine
    
    def save_model(self, filepath: str = None):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(config.MODEL_DIR, f"{self.model_type}_{timestamp}.h5")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath: Path to model file
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict(self, images):
        """
        Make predictions
        
        Args:
            images: Input images (preprocessed)
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        return self.model.predict(images)


if __name__ == "__main__":
    # Test model creation
    classifier = FoodClassifier(num_classes=10)
    model = classifier.build_model()
    print("\nModel created successfully!")
    model.summary()
