"""
Simple prediction script for single images
"""
import argparse
import os
import json
import numpy as np
from tensorflow import keras
import cv2

from preprocessing import ImagePreprocessor


def predict_single_image(image_path, model_path, class_names_path, top_k=5):
    """
    Predict class for a single image
    
    Args:
        image_path: Path to image file
        model_path: Path to trained model
        class_names_path: Path to class names JSON
        top_k: Number of top predictions to show
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Load class names
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    print(f"Model loaded with {len(class_names)} classes")
    
    # Load and preprocess image
    print(f"\nProcessing image: {image_path}")
    preprocessor = ImagePreprocessor()
    
    processed_image = preprocessor.preprocess_from_path(
        image_path,
        apply_clahe=True,
        normalize=True
    )
    
    if processed_image is None:
        print("Error: Could not load or process image")
        return
    
    # Add batch dimension
    input_image = np.expand_dims(processed_image, axis=0)
    
    # Predict
    print("Making prediction...")
    predictions = model.predict(input_image, verbose=0)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    for i, idx in enumerate(top_indices, 1):
        class_name = class_names[idx]
        confidence = predictions[idx] * 100
        
        # Create bar visualization
        bar_length = int(confidence / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        print(f"\n{i}. {class_name}")
        print(f"   Confidence: {confidence:.2f}%")
        print(f"   {bar}")
    
    print("\n" + "="*60)
    print(f"Top prediction: {class_names[top_indices[0]]} ({predictions[top_indices[0]]*100:.2f}%)")
    print("="*60)
    
    # Display image (optional)
    try:
        image = cv2.imread(image_path)
        if image is not None:
            # Resize for display
            display_image = cv2.resize(image, (400, 400))
            
            # Add prediction text
            top_class = class_names[top_indices[0]]
            top_conf = predictions[top_indices[0]] * 100
            text = f"{top_class}: {top_conf:.1f}%"
            
            cv2.putText(
                display_image, 
                text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            cv2.imshow("Prediction", display_image)
            print("\nPress any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Could not display image: {e}")


def main():
    parser = argparse.ArgumentParser(description='Predict food class for a single image')
    
    parser.add_argument('image', type=str,
                       help='Path to image file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.h5 file)')
    parser.add_argument('--classes', type=str, default=None,
                       help='Path to class names JSON file')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Find class names file if not provided
    if args.classes is None:
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        model_dir = os.path.dirname(args.model)
        class_names_path = os.path.join(model_dir, f"{model_name}_classes.json")
        
        if not os.path.exists(class_names_path):
            print(f"Error: Class names file not found: {class_names_path}")
            print("Please specify --classes argument")
            return
    else:
        class_names_path = args.classes
    
    # Make prediction
    predict_single_image(
        args.image,
        args.model,
        class_names_path,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
