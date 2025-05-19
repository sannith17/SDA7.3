import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.metrics import roc_curve, auc, accuracy_score
from torchvision import transforms
from datetime import datetime
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Workaround for Python 3.13+ compatibility
if sys.version_info >= (3, 13):
    import torch._classes
    torch._classes._register_python_class = lambda *args, **kwargs: None

# Unique key generator with page awareness
def widget_key(name):
    return f"{name}_p{st.session_state.page}_c{st.session_state.get('key_counter',0)}"

# Initialize all session state variables
def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 1
        st.session_state.key_counter = 0
        
        # Image processing states
        st.session_state.heatmap_overlay_svm = None
        st.session_state.heatmap_overlay_cnn = None
        st.session_state.aligned_images = None
        st.session_state.change_mask = None
        
        # Classification states
        st.session_state.classification_svm = None
        st.session_state.classification_cnn = None
        st.session_state.classification = None
        
        # File and date states
        st.session_state.before_date = datetime(2023, 1, 1)
        st.session_state.after_date = datetime(2023, 6, 1)
        st.session_state.before_file = None
        st.session_state.after_file = None
        
        # Model states
        st.session_state.model_choice = "SVM"
        st.session_state.svm_roc_fig = None
        st.session_state.cnn_roc_fig = None
        st.session_state.svm_accuracy = None
        st.session_state.cnn_accuracy = None
        
        # Default classifications
        st.session_state.classification_before_svm = {"Vegetation": 50, "Land": 30, "Water": 20}
        st.session_state.classification_before_cnn = {"Vegetation": 55, "Land": 25, "Developed": 20}

init_session_state()

# Increment key counter when page changes
if 'last_page' not in st.session_state:
    st.session_state.last_page = st.session_state.page
elif st.session_state.last_page != st.session_state.page:
    st.session_state.key_counter += 1
    st.session_state.last_page = st.session_state.page

# Configure Streamlit
st.set_page_config(layout="wide", page_title="Satellite Image Analysis")

# Custom title
st.markdown("""
    <h1 style='color: yellow; font-size: 72px; font-weight: bold; text-align: center;'>
        Satellite Image Analysis
    </h1>
""", unsafe_allow_html=True)

# -------- Models --------
class StableCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6*14*14, 3)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

# Initialize models safely
try:
    cnn_model = StableCNN()
    cnn_model.eval()
    svm_model = svm.SVC(probability=True, random_state=42)
except Exception as e:
    st.error(f"Model initialization failed: {e}")
    st.stop()

# -------- Image Processing --------
def validate_image(image):
    """Ensure image is in RGB format"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        elif image.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))
        return Image.fromarray(image)
    return image.convert("RGB")

def preprocess_img(img, size=(64,64)):
    """Standardize image for model input"""
    try:
        img = validate_image(img)
        img = img.resize(size)
        return np.array(img)/255.0
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return None

def robust_align_images(img1, img2):
    """Align images with error handling"""
    try:
        img1 = validate_image(img1)
        img2 = validate_image(img2)
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
        
        # Initialize alignment
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
        
        # Find transformation
        _, warp_matrix = cv2.findTransformECC(gray1, gray2, warp_matrix, 
                                           cv2.MOTION_EUCLIDEAN, criteria)
        
        # Apply transformation
        aligned = cv2.warpAffine(img2_np, warp_matrix, 
                               (img1_np.shape[1], img1_np.shape[0]),
                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        # Calculate difference
        diff = cv2.absdiff(img1_np, aligned)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY_INV)
        aligned_black = cv2.bitwise_and(aligned, aligned, mask=mask)
        
        return Image.fromarray(aligned), Image.fromarray(aligned_black)
    except Exception as e:
        st.error(f"Alignment failed: {e}")
        return img2, img2  # Return original if alignment fails

# [Continued with all other functions and pages...]

# -------- Page Functions --------
def page1():
    """Model selection page"""
    st.header("1. Model Selection")
    
    col1, col2 = st.columns([3,1])
    with col1:
        st.session_state.model_choice = st.selectbox(
            "Select Analysis Model", 
            ["SVM", "CNN"],
            key=widget_key("model_select"),
            help="Choose between SVM (faster) or CNN (more accurate)"
        )
    with col2:
        if st.button("Next ➡️", key=widget_key("next_btn")):
            st.session_state.page = 2
    
    st.markdown("""
        <div style='background: #2e2e2e; padding: 15px; border-radius: 10px; margin-top: 20px;'>
            <h3>Model Information</h3>
            <p><b>SVM:</b> Faster processing, better for smaller datasets</p>
            <p><b>CNN:</b> More accurate, better for complex patterns</p>
        </div>
    """, unsafe_allow_html=True)

def page2():
    """Image upload page"""
    st.header("2. Image Upload & Dates")
    
    with st.sidebar:
        st.subheader("Image Dates")
        st.session_state.before_date = st.date_input(
            "BEFORE image date",
            value=st.session_state.before_date,
            key=widget_key("before_date")
        )
        st.session_state.after_date = st.date_input(
            "AFTER image date", 
            value=st.session_state.after_date,
            min_value=st.session_state.before_date,
            key=widget_key("after_date")
        )
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("BEFORE Image")
        st.session_state.before_file = st.file_uploader(
            "Upload BEFORE satellite image",
            type=["png", "jpg", "jpeg", "tif"],
            key=widget_key("before_upload")
        )
        if st.session_state.before_file:
            st.image(st.session_state.before_file, use_column_width=True)
    
    with col2:
        st.subheader("AFTER Image")
        st.session_state.after_file = st.file_uploader(
            "Upload AFTER satellite image",
            type=["png", "jpg", "jpeg", "tif"],
            key=widget_key("after_upload")
        )
        if st.session_state.after_file:
            st.image(st.session_state.after_file, use_column_width=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        if st.button("⬅️ Back", key=widget_key("back_btn")):
            st.session_state.page = 1
    with col2:
        if st.session_state.before_file and st.session_state.after_file:
            if st.button("Process Images ➡️", key=widget_key("process_btn")):
                with st.spinner("Processing images..."):
                    try:
                        before_img = Image.open(st.session_state.before_file)
                        after_img = Image.open(st.session_state.after_file)
                        
                        # Align images
                        aligned_after, aligned_black = robust_align_images(before_img, after_img)
                        st.session_state.aligned_images = {
                            "before": before_img,
                            "after": aligned_after,
                            "aligned_black": aligned_black
                        }
                        
                        # Calculate changes
                        st.session_state.change_mask = get_change_mask(before_img, aligned_after)
                        
                        # Classify based on model
                        if st.session_state.model_choice == "SVM":
                            after_arr = preprocess_img(aligned_after)
                            if after_arr is not None:
                                st.session_state.classification_svm = classify_land_svm(after_arr)
                                st.session_state.classification = st.session_state.classification_svm
                                
                                # Create heatmap
                                h, w = st.session_state.change_mask.shape
                                heatmap = np.zeros((h, w, 3), dtype=np.uint8)
                                heatmap[..., 0] = st.session_state.change_mask * 255  # Blue channel
                                heatmap_img = Image.fromarray(heatmap)
                                aligned_resized = aligned_after.resize((w, h))
                                st.session_state.heatmap_overlay_svm = Image.blend(
                                    aligned_resized.convert("RGB"),
                                    heatmap_img.convert("RGB"),
                                    alpha=0.5
                                )
                        elif st.session_state.model_choice == "CNN":
                            st.session_state.classification_cnn = classify_land_cnn(aligned_after)
                            st.session_state.classification = st.session_state.classification_cnn
                            
                            # Create heatmap
                            h, w = st.session_state.change_mask.shape
                            heatmap = np.zeros((h, w, 3), dtype=np.uint8)
                            heatmap[..., 1] = st.session_state.change_mask * 255  # Green channel
                            heatmap_img = Image.fromarray(heatmap)
                            aligned_resized = aligned_after.resize((w, h))
                            st.session_state.heatmap_overlay_cnn = Image.blend(
                                aligned_resized.convert("RGB"),
                                heatmap_img.convert("RGB"),
                                alpha=0.5
                            )
                        
                        st.session_state.page = 3
                    except Exception as e:
                        st.error(f"Processing failed: {str(e)}")
        else:
            st.warning("Please upload both images to proceed")

# [Additional page functions would continue here...]

# Main app controller
def main():
    try:
        if st.session_state.page == 1:
            page1()
        elif st.session_state.page == 2:
            page2()
        # [Add other page conditions here...]
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.button("Restart App", on_click=lambda: st.session_state.clear())

if __name__ == "__main__":
    main()
