"""
Enhanced Food Classification App with Object Detection and Counting
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO
import torch
from tensorflow import keras

import config
from preprocessing import ImagePreprocessor

# Page config
st.set_page_config(
    page_title="Enhanced Food Classification",
    page_icon="🍎",
    layout="wide"
)

# Initialize YOLO model for object detection
@st.cache_resource
def load_yolo_model():
    try:
        # Load a pretrained YOLO model (you can replace with your own trained model)
        model = YOLO('yolov8n.pt')  # Using nano version for speed, can use larger models for better accuracy
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

@st.cache_resource
def load_model_and_classes(model_path, class_names_path):
    """
    Load model and class names (cached)
    """
    try:
        model = keras.models.load_model(model_path)
        
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def detect_food_items(image, yolo_model, confidence_threshold=0.5):
    """
    Detect food items in the image using YOLO
    """
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run YOLO detection
    results = yolo_model(image_cv, conf=confidence_threshold, verbose=False)
    
    # Get detections
    detections = []
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': yolo_model.names[int(cls)]
            })
    
    return detections

def predict_food(image, model, class_names, preprocessor, top_k=5):
    """
    Predict food class for an image
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    # Preprocess
    processed = preprocessor.preprocess(image_bgr, apply_clahe=True, normalize=True)
    
    # Add batch dimension
    input_image = np.expand_dims(processed, axis=0)
    
    # Predict
    predictions = model.predict(input_image, verbose=0)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    results = {
        'top_class': class_names[top_indices[0]],
        'top_confidence': float(predictions[top_indices[0]]),
        'top_k_predictions': [
            {
                'class': class_names[idx],
                'confidence': float(predictions[idx])
            }
            for idx in top_indices
        ]
    }
    
    return results

def draw_detections(image, detections, class_colors=None):
    """
    Draw detection boxes and labels on the image
    """
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    draw = ImageDraw.Draw(image)
    
    # Default colors for different classes
    if class_colors is None:
        class_colors = {
            'apple': 'red',
            'banana': 'yellow',
            'orange': 'orange',
            # Add more class-color mappings as needed
        }
    
    # Count detections by class
    class_counts = defaultdict(int)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class_name']
        confidence = det['confidence']
        
        # Get color for this class
        color = class_colors.get(class_name.lower(), 'white')
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label background
        label = f"{class_name} {confidence:.2f}"
        text_bbox = draw.textbbox((x1, y1 - 20), label)
        draw.rectangle(text_bbox, fill=color)
        
        # Draw text
        draw.text((x1, y1 - 20), label, fill='black')
        
        # Update class count
        class_counts[class_name] += 1
    
    return image, dict(class_counts)

def main():
    """
    Main Streamlit app
    """
    # Title
    st.title("🍎 Enhanced Food Classification System")
    st.markdown("---")
    
    # Load YOLO model
    with st.spinner("Loading object detection model..."):
        yolo_model = load_yolo_model()
    
    if yolo_model is None:
        st.error("Failed to load object detection model. Please check the logs.")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_dir = config.MODEL_DIR
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
        
        if model_files:
            selected_model = st.sidebar.selectbox(
                "Select Classification Model",
                model_files,
                index=0
            )
            
            model_path = os.path.join(model_dir, selected_model)
            
            # Find corresponding class names file
            model_name = os.path.splitext(selected_model)[0]
            class_names_path = os.path.join(model_dir, f"{model_name}_classes.json")
            
            if not os.path.exists(class_names_path):
                # Try to find any class names file
                class_files = [f for f in os.listdir(model_dir) if f.endswith('_classes.json')]
                if class_files:
                    class_names_path = os.path.join(model_dir, class_files[0])
                else:
                    st.sidebar.error("Class names file not found!")
                    return
        else:
            st.sidebar.error("No trained models found!")
            st.info("Please train a model first using `train.py`")
            return
    else:
        st.sidebar.error(f"Model directory not found: {model_dir}")
        return
    
    # Load classification model
    with st.spinner("Loading classification model..."):
        model, class_names = load_model_and_classes(model_path, class_names_path)
    
    if model is None or class_names is None:
        return
    
    st.sidebar.success(f"✅ Model loaded: {selected_model}")
    st.sidebar.info(f"📊 Classes: {len(class_names)}")
    
    # Detection settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.05,
        help="Adjust the confidence threshold for object detection"
    )
    
    # Preprocessing options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Preprocessing")
    apply_clahe = st.sidebar.checkbox("Apply CLAHE Enhancement", value=True)
    top_k = st.sidebar.slider("Top-K Predictions", 1, min(10, len(class_names)), 5)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a food/fruit image for classification"
        )
        
        # Camera input
        camera_image = st.camera_input("Or take a photo")
        
        # Use camera image if available, otherwise uploaded file
        input_image = camera_image if camera_image else uploaded_file
        
        if input_image is not None:
            # Load image
            image = Image.open(input_image).convert('RGB')
            
            # Display original image
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Predict button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Detect food items
                    detections = detect_food_items(
                        image, 
                        yolo_model,
                        confidence_threshold=confidence_threshold
                    )
                    
                    # Classify the main image
                    classification = predict_food(
                        image,
                        model,
                        class_names,
                        preprocessor,
                        top_k=top_k
                    )
                    
                    # Draw detections on the image
                    if detections:
                        # Convert to RGB for display
                        image_rgb = image.convert('RGB')
                        annotated_image, class_counts = draw_detections(
                            image_rgb, 
                            detections
                        )
                        
                        # Display detected image with bounding boxes
                        st.image(
                            annotated_image, 
                            caption=f"Detected {len(detections)} food items",
                            use_container_width=True
                        )
                        
                        # Display detection results
                        with st.expander("🔍 Detection Results", expanded=True):
                            st.subheader("Detected Food Items")
                            
                            # Show count for each detected class
                            if class_counts:
                                st.write("### 🍽️ Food Item Counts")
                                for class_name, count in class_counts.items():
                                    st.write(f"- **{class_name.capitalize()}**: {count}")
                            else:
                                st.info("No food items detected. Try adjusting the confidence threshold.")
                    else:
                        st.warning("No food items detected. Try adjusting the confidence threshold.")
                    
                    # Store results in session state
                    st.session_state['classification'] = classification
                    st.session_state['detections'] = detections
                    st.session_state['image'] = image
    
    with col2:
        st.subheader("📊 Classification Results")
        
        if 'classification' in st.session_state:
            classification = st.session_state['classification']
            
            # Top prediction
            st.markdown("### 🏆 Top Prediction")
            st.success(f"**{classification['top_class']}**")
            st.metric(
                "Confidence",
                f"{classification['top_confidence']*100:.2f}%"
            )
            
            # Progress bar for confidence
            st.progress(classification['top_confidence'])
            
            st.markdown("---")
            
            # Top-K predictions
            st.markdown(f"### 📈 Top-{top_k} Predictions")
            
            for i, pred in enumerate(classification['top_k_predictions'], 1):
                with st.expander(
                    f"{i}. {pred['class']} - {pred['confidence']*100:.2f}%",
                    expanded=(i == 1)
                ):
                    st.progress(pred['confidence'])
            
            # Visualization
            st.markdown("---")
            st.markdown("### 📊 Confidence Distribution")
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            classes = [p['class'] for p in classification['top_k_predictions']]
            confidences = [p['confidence'] for p in classification['top_k_predictions']]
            
            bars = ax.barh(classes, confidences, color='skyblue')
            bars[0].set_color('green')
            
            ax.set_xlabel('Confidence')
            ax.set_xlim([0, 1])
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show detection results if available
            if 'detections' in st.session_state and st.session_state['detections']:
                st.markdown("---")
                st.markdown("### 🎯 Detection Details")
                
                for i, det in enumerate(st.session_state['detections'], 1):
                    with st.expander(f"Detection {i}: {det['class_name']} ({det['confidence']:.2f})"):
                        st.json({
                            'Bounding Box': det['bbox'],
                            'Confidence': f"{det['confidence']:.4f}",
                            'Class': det['class_name']
                        })
        else:
            st.info("👈 Upload an image and click 'Analyze Image' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit, TensorFlow, YOLO, and OpenCV</p>
            <p>Features: Object Detection, Food Classification, Real-time Analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 About")
    st.sidebar.info(
        """
        This enhanced app uses:
        - YOLO for object detection
        - Transfer learning with MobileNet/ResNet 
        - Advanced OpenCV preprocessing
        
        **Features:**
        - Real-time food detection & counting
        - Multi-food item recognition
        - Confidence scoring
        - Interactive visualization
        """
    )
    
    # Display class list
    with st.sidebar.expander("📋 Available Classes"):
        for i, class_name in enumerate(class_names, 1):
            st.text(f"{i}. {class_name}")


if __name__ == "__main__":
    main()
