"""
Streamlit web application for food classification
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os
from tensorflow import keras
import matplotlib.pyplot as plt

import config
from preprocessing import ImagePreprocessor


# Page config
st.set_page_config(
    page_title="Food Classification",
    page_icon="🍎",
    layout="wide"
)


@st.cache_resource
def load_model_and_classes(model_path, class_names_path):
    """
    Load model and class names (cached)
    
    Args:
        model_path: Path to model file
        class_names_path: Path to class names JSON
        
    Returns:
        Tuple of (model, class_names)
    """
    try:
        model = keras.models.load_model(model_path)
        
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def predict_image(image, model, class_names, preprocessor, top_k=5):
    """
    Predict image class
    
    Args:
        image: PIL Image or numpy array
        model: Keras model
        class_names: List of class names
        preprocessor: ImagePreprocessor instance
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with predictions
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


def main():
    """
    Main Streamlit app
    """
    # Title
    st.title("🍎 Food Classification System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_dir = config.MODEL_DIR
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
        
        if model_files:
            selected_model = st.sidebar.selectbox(
                "Select Model",
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
    
    # Load model
    with st.spinner("Loading model..."):
        model, class_names = load_model_and_classes(model_path, class_names_path)
    
    if model is None or class_names is None:
        return
    
    st.sidebar.success(f"✅ Model loaded: {selected_model}")
    st.sidebar.info(f"📊 Classes: {len(class_names)}")
    
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
            image = Image.open(input_image)
            
            # Display original image
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Predict button
            if st.button("🔍 Classify", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Predict
                    results = predict_image(
                        image, 
                        model, 
                        class_names, 
                        preprocessor,
                        top_k=top_k
                    )
                    
                    # Store results in session state
                    st.session_state['results'] = results
                    st.session_state['image'] = image
    
    with col2:
        st.subheader("📊 Results")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            # Top prediction
            st.markdown("### 🏆 Top Prediction")
            st.success(f"**{results['top_class']}**")
            st.metric(
                "Confidence",
                f"{results['top_confidence']*100:.2f}%"
            )
            
            # Progress bar for confidence
            st.progress(results['top_confidence'])
            
            st.markdown("---")
            
            # Top-K predictions
            st.markdown(f"### 📈 Top-{top_k} Predictions")
            
            for i, pred in enumerate(results['top_k_predictions'], 1):
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
            classes = [p['class'] for p in results['top_k_predictions']]
            confidences = [p['confidence'] for p in results['top_k_predictions']]
            
            bars = ax.barh(classes, confidences, color='skyblue')
            bars[0].set_color('green')
            
            ax.set_xlabel('Confidence')
            ax.set_xlim([0, 1])
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("👈 Upload an image and click 'Classify' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit, TensorFlow, and OpenCV</p>
            <p>Transfer Learning: MobileNet/ResNet | Preprocessing: OpenCV CLAHE</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 About")
    st.sidebar.info(
        """
        This app uses transfer learning with MobileNet/ResNet 
        and OpenCV preprocessing for food classification.
        
        **Features:**
        - Real-time classification
        - CLAHE enhancement
        - Top-K predictions
        - Confidence visualization
        """
    )
    
    # Display class list
    with st.sidebar.expander("📋 Available Classes"):
        for i, class_name in enumerate(class_names, 1):
            st.text(f"{i}. {class_name}")


if __name__ == "__main__":
    main()
