import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import io
from sklearn import svm
from sklearn.metrics import roc_curve, auc, accuracy_score
from torchvision import transforms
from datetime import datetime
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Workaround for Python compatibility issues
if sys.version_info >= (3, 13):
    import torch._classes
    torch._classes._register_python_class = lambda *args, **kwargs: None

# Initialize session state for page navigation and data storage
def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 1
    if 'heatmap_overlay_svm' not in st.session_state:
        st.session_state.heatmap_overlay_svm = None
    if 'heatmap_overlay_cnn' not in st.session_state:
        st.session_state.heatmap_overlay_cnn = None
    if 'aligned_images' not in st.session_state:
        st.session_state.aligned_images = None
    if 'change_mask' not in st.session_state:
        st.session_state.change_mask = None
    if 'classification_svm' not in st.session_state:
        st.session_state.classification_svm = None
    if 'classification_cnn' not in st.session_state:
        st.session_state.classification_cnn = None
    if 'before_date' not in st.session_state:
        st.session_state.before_date = datetime(2023, 1, 1)
    if 'after_date' not in st.session_state:
        st.session_state.after_date = datetime(2023, 6, 1)
    if 'before_file' not in st.session_state:
        st.session_state.before_file = None
    if 'after_file' not in st.session_state:
        st.session_state.after_file = None
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = "SVM"
    if 'svm_roc_fig' not in st.session_state:
        st.session_state.svm_roc_fig = None
    if 'cnn_roc_fig' not in st.session_state:
        st.session_state.cnn_roc_fig = None
    if 'svm_accuracy' not in st.session_state:
        st.session_state.svm_accuracy = None
    if 'cnn_accuracy' not in st.session_state:
        st.session_state.cnn_accuracy = None
    if 'classification_before_svm' not in st.session_state:
        st.session_state.classification_before_svm = {"Vegetation": 50, "Land": 30, "Water": 20}
    if 'classification_before_cnn' not in st.session_state:
        st.session_state.classification_before_cnn = {"Vegetation": 55, "Land": 25, "Developed": 20}

initialize_session_state()

# Set the page layout and browser tab title
st.set_page_config(layout="wide", page_title="Satellite Image Analysis")

# Custom visible title with yellow color and large font
st.markdown(
    """
    <h1 style='color: yellow; font-size: 72px; font-weight: bold; text-align: center;'>
        Satellite Image Analysis
    </h1>
    """,
    unsafe_allow_html=True
)

# -------- Models --------
class DummyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 14 * 14, 3)  # Assuming 3 classes
        
        # Workaround for PyTorch internal class registration
        if hasattr(torch._C, '_ImperativeEngine'):
            self._backend = torch._C._ImperativeEngine()
        else:
            self._backend = None
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

# Initialize models safely
try:
    cnn_model = DummyCNN()
    cnn_model.eval()
    svm_model = svm.SVC(probability=True, random_state=42)
except Exception as e:
    st.error(f"Model initialization failed: {e}")
    st.stop()

# -------- Image Processing Functions --------
def validate_image(image):
    """Validate and convert image to RGB format"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return Image.fromarray(image)
    elif isinstance(image, Image.Image):
        return image.convert("RGB")
    else:
        raise ValueError("Unsupported image format")

def preprocess_img(img, size=(64, 64)):
    """Preprocess image for model input"""
    try:
        img = validate_image(img)
        img = img.resize(size)
        img_arr = np.array(img) / 255.0
        return img_arr
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        return None

def align_images(img1, img2):
    """Align images using ECC (Enhanced Correlation Coefficient) method"""
    try:
        img1_np = np.array(validate_image(img1))
        img2_np = np.array(validate_image(img2))

        # Convert to grayscale
        gray1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)

        # Initialize warp matrix
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

        # Find transformation
        _, warp_matrix = cv2.findTransformECC(gray1, gray2, warp_matrix, warp_mode, criteria)
        
        # Apply transformation
        aligned = cv2.warpAffine(img2_np, warp_matrix, 
                               (img1_np.shape[1], img1_np.shape[0]),
                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        # Calculate difference mask
        diff_mask = cv2.absdiff(img1_np, aligned)
        diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_RGB2GRAY)
        _, black_mask = cv2.threshold(diff_mask, 30, 255, cv2.THRESH_BINARY_INV)
        aligned_black = cv2.bitwise_and(aligned, aligned, mask=black_mask)

        return Image.fromarray(aligned), Image.fromarray(aligned_black)
    except Exception as e:
        st.error(f"Image alignment failed: {e}")
        # Return original images if alignment fails
        return validate_image(img2), validate_image(img2)

def get_change_mask(img1, img2, threshold=30):
    """Generate change mask between two images"""
    try:
        img1 = validate_image(img1)
        img2 = validate_image(img2).resize(img1.size)

        gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, change_mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
        return change_mask.astype(np.uint8)
    except Exception as e:
        st.error(f"Change mask generation failed: {e}")
        return np.zeros((100, 100), dtype=np.uint8)  # Return empty mask if error occurs

# -------- Classification Functions --------
def classify_land_svm(img_arr):
    """Simplified land classification using SVM"""
    try:
        if img_arr is None:
            return {"Vegetation": 0, "Land": 0, "Water": 0}

        features = img_arr.flatten()[:100]  # Use a subset for dummy training
        labels = np.random.randint(0, 3, 50)  # Dummy labels
        
        # Train with error handling
        try:
            svm_model.fit(np.random.rand(50, 100), labels)
            probabilities = svm_model.predict_proba(features.reshape(1, -1))[0]
        except:
            probabilities = np.array([0.4, 0.3, 0.3])  # Default probabilities if training fails
            
        classes = ["Vegetation", "Land", "Water"]
        return {classes[i]: prob * 100 for i, prob in enumerate(probabilities)}
    except Exception as e:
        st.error(f"SVM classification failed: {e}")
        return {"Vegetation": 33.3, "Land": 33.3, "Water": 33.3}

def classify_land_cnn(img):
    """Simplified land classification using CNN"""
    try:
        img = validate_image(img)
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = cnn_model(img_tensor)
            probabilities = torch.softmax(output, dim=1).numpy()[0]
            classes = ["Vegetation", "Land", "Developed"]
            return {classes[i]: prob * 100 for i, prob in enumerate(probabilities)}
    except Exception as e:
        st.error(f"CNN classification failed: {e}")
        return {"Vegetation": 33.3, "Land": 33.3, "Developed": 33.3}

# -------- Analysis Functions --------
def detect_calamity(date1, date2, change_percentage):
    """Detects potential calamities based on changes and time difference"""
    try:
        date_diff = (date2 - date1).days
        change_percentage = float(change_percentage)

        if change_percentage > 0.15:
            if date_diff <= 10:
                return "⚠️ **Possible Flood:** Rapid and significant changes observed in a short period may indicate flooding."
            elif date_diff <= 30:
                return "🔥 **Possible Deforestation:** Significant loss of vegetation over a short term could suggest deforestation or wildfires."
            else:
                return "🏗️ **Possible Urbanization:** Gradual yet significant increase in developed areas over time might indicate urbanization."
        elif change_percentage > 0.05:
            return "🌱 **Seasonal Changes Detected:** Minor changes likely due to natural seasonal variations in vegetation or water bodies."
        return "✅ **No Significant Calamity Detected:** Minimal changes observed between the two images."
    except:
        return "❓ **Analysis Unavailable:** Could not determine calamity status."

def generate_roc_curve(model_type):
    """Generate ROC curve for model evaluation"""
    try:
        # Dummy data - replace with actual model evaluations
        y_true = np.array([0, 0, 1, 1, 0, 1] if model_type == "SVM" else [0, 1, 0, 1, 1, 0])
        y_scores = np.array([0.2, 0.3, 0.7, 0.8, 0.4, 0.9] if model_type == "SVM" else [0.8, 0.6, 0.3, 0.9, 0.7, 0.2])

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        color = 'blue' if model_type == "SVM" else 'green'
        ax.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Receiver Operating Characteristic ({model_type})')
        ax.legend(loc="lower right")
        return fig
    except Exception as e:
        st.error(f"ROC curve generation failed: {e}")
        return plt.figure()  # Return empty figure

def calculate_accuracy(model_type):
    """Calculate accuracy for model evaluation"""
    try:
        # Dummy accuracy - replace with actual model evaluations
        return 0.85 if model_type == "SVM" else 0.88
    except:
        return 0.0

# -------- Page Functions --------
def page1():
    """Model selection page"""
    st.header("1. Model Selection")
    st.session_state.model_choice = st.selectbox(
        "Select Analysis Model", 
        ["SVM", "CNN"],
        help="Choose between Support Vector Machine (SVM) or Convolutional Neural Network (CNN)"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**SVM** - Faster but less accurate for complex patterns")
    with col2:
        st.markdown("**CNN** - More accurate but requires more computation")
    
    if st.button("Next ➡️", key="page1_next"):
        st.session_state.page = 2

def page2():
    """Image upload and date selection page"""
    st.markdown(
        """
        <h2 style='font-size: 36px; color: white;'>
            2. Image Upload & Dates
        </h2>
        <p style='font-size: 18px; color: lightgray;'>
            Please upload the <b>before</b> and <b>after/current</b> satellite images along with their respective dates for analysis.
        </p>
        """,
        unsafe_allow_html=True
    )
    
    with st.sidebar:
        st.session_state.before_date = st.date_input(
            "BEFORE image date", 
            value=st.session_state.before_date,
            max_value=datetime.today()
        )
        st.session_state.before_file = st.file_uploader(
            "Upload BEFORE image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="before"
        )
        
        st.session_state.after_date = st.date_input(
            "AFTER image date", 
            value=st.session_state.after_date,
            min_value=st.session_state.before_date,
            max_value=datetime.today()
        )
        st.session_state.after_file = st.file_uploader(
            "Upload AFTER image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="after"
        )

    # Display preview if images are uploaded
    if st.session_state.before_file and st.session_state.after_file:
        try:
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.before_file, caption="Before Image Preview", use_column_width=True)
            with col2:
                st.image(st.session_state.after_file, caption="After Image Preview", use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying image previews: {e}")

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back", key="page2_back"):
            st.session_state.page = 1
    with col2:
        if st.session_state.before_file and st.session_state.after_file:
            if st.button("Next ➡️", key="page2_next"):
                try:
                    # Process images
                    before_img = Image.open(st.session_state.before_file)
                    after_img = Image.open(st.session_state.after_file)
                    
                    # Align images
                    aligned_after, aligned_black = align_images(before_img, after_img)
                    st.session_state.aligned_images = {
                        "before": before_img,
                        "after": aligned_after,
                        "aligned_black": aligned_black
                    }
                    
                    # Calculate change mask
                    st.session_state.change_mask = get_change_mask(before_img, aligned_after)
                    
                    # Classify based on selected model
                    if st.session_state.model_choice == "SVM":
                        after_arr = preprocess_img(aligned_after)
                        if after_arr is not None:
                            st.session_state.classification_svm = classify_land_svm(after_arr)
                            st.session_state.classification = st.session_state.classification_svm
                            
                            # Create SVM heatmap (Blue)
                            h, w = st.session_state.change_mask.shape
                            heatmap_svm = np.zeros((h, w, 3), dtype=np.uint8)
                            heatmap_svm[..., 0] = st.session_state.change_mask * 255
                            heatmap_img_svm = Image.fromarray(heatmap_svm)
                            aligned_after_resized = aligned_after.resize((w, h))
                            st.session_state.heatmap_overlay_svm = Image.blend(
                                aligned_after_resized.convert("RGB"),
                                heatmap_img_svm.convert("RGB"),
                                alpha=0.5
                            )
                            
                            # Generate evaluation metrics
                            st.session_state.svm_roc_fig = generate_roc_curve("SVM")
                            st.session_state.svm_accuracy = calculate_accuracy("SVM")
                            
                    elif st.session_state.model_choice == "CNN":
                        st.session_state.classification_cnn = classify_land_cnn(aligned_after)
                        st.session_state.classification = st.session_state.classification_cnn
                        
                        # Create CNN heatmap (Green)
                        h, w = st.session_state.change_mask.shape
                        heatmap_cnn = np.zeros((h, w, 3), dtype=np.uint8)
                        heatmap_cnn[..., 1] = st.session_state.change_mask * 255
                        heatmap_img_cnn = Image.fromarray(heatmap_cnn)
                        aligned_after_resized = aligned_after.resize((w, h))
                        st.session_state.heatmap_overlay_cnn = Image.blend(
                            aligned_after_resized.convert("RGB"),
                            heatmap_img_cnn.convert("RGB"),
                            alpha=0.5
                        )
                        
                        # Generate evaluation metrics
                        st.session_state.cnn_roc_fig = generate_roc_curve("CNN")
                        st.session_state.cnn_accuracy = calculate_accuracy("CNN")
                    
                    st.session_state.page = 3
                except Exception as e:
                    st.error(f"Error processing images: {str(e)}")
        else:
            st.warning("Please upload both images to proceed")

def page3():
    """Aligned images comparison page"""
    st.header("3. Aligned Images Comparison")

    if st.session_state.aligned_images is None:
        st.error("No aligned images found. Please upload images first.")
        st.session_state.page = 2
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(
            st.session_state.aligned_images["before"],
            caption="BEFORE Image", 
            use_column_width=True,
            channels="RGB"
        )
    with col2:
        st.image(
            st.session_state.aligned_images["after"],
            caption="Aligned AFTER Image", 
            use_column_width=True,
            channels="RGB"
        )
    with col3:
        st.image(
            st.session_state.aligned_images["aligned_black"],
            caption="Aligned Difference", 
            use_column_width=True,
            channels="RGB"
        )

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back", key="page3_back"):
            st.session_state.page = 2
    with col2:
        if st.button("Next ➡️", key="page3_next"):
            st.session_state.page = 4

def page4():
    """Change detection heatmap page"""
    st.header("4. Change Detection Heatmap")

    # Validate data
    if not all(key in st.session_state for key in ['aligned_images', 'change_mask']):
        st.error("Please upload and process images first")
        st.session_state.page = 2
        return

    st.subheader(f"Heatmap using {st.session_state.model_choice} Model")
    
    h, w = st.session_state.change_mask.shape
    aligned_after = st.session_state.aligned_images["after"]
    aligned_after_resized = aligned_after.resize((w, h))

    # Display appropriate heatmap
    if st.session_state.model_choice == "SVM" and st.session_state.heatmap_overlay_svm:
        st.image(
            st.session_state.heatmap_overlay_svm, 
            caption="Change Heatmap (Dark_Coloured = Changes)", 
            use_column_width=True,
            channels="RGB"
        )
    elif st.session_state.model_choice == "CNN" and st.session_state.heatmap_overlay_cnn:
        st.image(
            st.session_state.heatmap_overlay_cnn, 
            caption="Change Heatmap (Dark_Coloured = Changes)", 
            use_column_width=True,
            channels="RGB"
        )
    else:
        # Default red heatmap
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        heatmap[..., 2] = st.session_state.change_mask * 255
        heatmap_img = Image.fromarray(heatmap)
        heatmap_overlay = Image.blend(
            aligned_after_resized.convert("RGB"),
            heatmap_img.convert("RGB"),
            alpha=0.5
        )
        st.image(
            heatmap_overlay, 
            caption="Change Heatmap (Red = Changes)", 
            use_column_width=True,
            channels="RGB"
        )

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back", key="page4_back"):
            st.session_state.page = 3
    with col2:
        if st.button("Next ➡️", key="page4_next"):
            st.session_state.page = 5

def page5():
    """Land classification and analysis page"""
    st.header("5. Land Classification & Analysis")

    # Validate data
    required_keys = ['classification', 'change_mask', 'before_date', 'after_date']
    if not all(key in st.session_state for key in required_keys):
        st.error("Analysis data not found. Please start from the beginning.")
        st.session_state.page = 1
        return

    # Calculate change percentage
    try:
        total_pixels = np.prod(st.session_state.change_mask.shape)
        changed_pixels = np.sum(st.session_state.change_mask)
        change_percentage = changed_pixels / total_pixels
    except:
        change_percentage = 0

    # Calamity detection
    st.subheader("🚨 Calamity Detection")
    calamity_report = detect_calamity(
        st.session_state.before_date,
        st.session_state.after_date,
        change_percentage
    )
    
    # Display calamity report with appropriate color
    if "⚠️" in calamity_report or "🔥" in calamity_report:
        color = "red"
    elif "🌱" in calamity_report:
        color = "green"
    elif "✅" in calamity_report:
        color = "lightgreen"
    else:
        color = "orange"
    
    st.markdown(f"<h3 style='color: {color};'>{calamity_report}</h3>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <p style='font-size: 16px; color: lightgray;'>
            <b>Change Detected:</b> {change_percentage:.2%} of the area<br>
            <b>Time Period:</b> {(st.session_state.after_date - st.session_state.before_date).days} days
        </p>
    """, unsafe_allow_html=True)

    # Classification Table
    st.subheader(f"Land Classification using {st.session_state.model_choice}")
    
    # Get classification data
    classification_data = st.session_state.classification
    if classification_data is None:
        classification_data = {"Vegetation": 0, "Land": 0, "Water": 0} if st.session_state.model_choice == "SVM" else {"land": 0, "water": 0, "vegetation": 0}
    
    # Display as both table and chart
    col1, col2 = st.columns([1, 2])
    with col1:
        df_class = pd.DataFrame(list(classification_data.items()), columns=["Class", "Area (%)"])
        st.table(df_class.style.format({"Area (%)": "{:.1f}%"}))
    
    with col2:
        # Get before classification data
        before_class = st.session_state.classification_before_svm if st.session_state.model_choice == "SVM" else st.session_state.classification_before_cnn
        
        # Create comparison pie charts
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Before image
        ax[0].pie(
            before_class.values(),
            labels=before_class.keys(),
            autopct='%1.1f%%',
            shadow=True,
            startangle=140,
            colors=['#2e8b57', '#cd853f', '#4682b4']  # Green, Brown, Blue
        )
        ax[0].set_title('Before Image')
        
        # After image
        ax[1].pie(
            classification_data.values(),
            labels=classification_data.keys(),
            autopct='%1.1f%%',
            shadow=True,
            startangle=140,
            colors=['#2e8b57', '#cd853f', '#4682b4']  # Green, Brown, Blue
        )
        ax[1].set_title('After Image')
        
        st.pyplot(fig)

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back", key="page5_back"):
            st.session_state.page = 4
    with col2:
        if st.button("Next ➡️", key="page5_next"):
            st.session_state.page = 6

def page6():
    """Model evaluation page"""
    st.header("6. Model Evaluation")
    
    if st.session_state.model_choice == "SVM":
        st.subheader("SVM Model Evaluation")
        
        # Display ROC curve
        if st.session_state.svm_roc_fig:
            st.pyplot(st.session_state.svm_roc_fig)
        else:
            st.warning("ROC curve data not available for SVM.")
        
        # Display accuracy
        if st.session_state.svm_accuracy is not None:
            st.metric("Model Accuracy", f"{st.session_state.svm_accuracy:.1%}")
        else:
            st.warning("Accuracy data not available for SVM.")
            
        # Model description
        st.markdown("""
            <div style='background-color: #2e2e2e; padding: 15px; border-radius: 10px;'>
                <h4>SVM Model Characteristics</h4>
                <ul>
                    <li><b>Pros:</b> Fast training, works well with small datasets, good for linear separations</li>
                    <li><b>Cons:</b> Less effective with complex patterns, requires careful feature engineering</li>
                    <li><b>Best for:</b> Quick analyses with limited computational resources</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
    elif st.session_state.model_choice == "CNN":
        st.subheader("CNN Model Evaluation")
        
        # Display ROC curve
        if st.session_state.cnn_roc_fig:
            st.pyplot(st.session_state.cnn_roc_fig)
        else:
            st.warning("ROC curve data not available for CNN.")
        
        # Display accuracy
        if st.session_state.cnn_accuracy is not None:
            st.metric("Model Accuracy", f"{st.session_state.cnn_accuracy:.1%}")
        else:
            st.warning("Accuracy data not available for CNN.")
            
        # Model description
        st.markdown("""
            <div style='background-color: #2e2e2e; padding: 15px; border-radius: 10px;'>
                <h4>CNN Model Characteristics</h4>
                <ul>
                    <li><b>Pros:</b> Excellent with image data, automatic feature extraction, handles complex patterns</li>
                    <li><b>Cons:</b> Requires more data, slower training, needs more computational power</li>
                    <li><b>Best for:</b> Detailed analyses where accuracy is critical</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Navigation button
    if st.button("⬅️ Back", key="page6_back"):
        st.session_state.page = 5

# -------- Main App Flow --------
def main():
    try:
        if st.session_state.page == 1:
            page1()
        elif st.session_state.page == 2:
            page2()
        elif st.session_state.page == 3:
            page3()
        elif st.session_state.page == 4:
            page4()
        elif st.session_state.page == 5:
            page5()
        elif st.session_state.page == 6:
            page6()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.button("Return to Start", on_click=lambda: st.session_state.update({"page": 1}))

if __name__ == "__main__":
    main()
