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
from sklearn.model_selection import train_test_split
from torchvision import transforms
from datetime import datetime
import seaborn as sns

# Initialize session state for page navigation and data storage
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
if 'svm_model' not in st.session_state:
    st.session_state.svm_model = None
if 'cnn_model' not in st.session_state:
    st.session_state.cnn_model = None
if 'svm_metrics' not in st.session_state:
    st.session_state.svm_metrics = {}
if 'cnn_metrics' not in st.session_state:
    st.session_state.cnn_metrics = {}
if 'before_classification' not in st.session_state:
    st.session_state.before_classification = {"Water": 0, "Vegetation": 0, "Barren Land": 0}
if 'after_classification' not in st.session_state:
    st.session_state.after_classification = {"Water": 0, "Vegetation": 0, "Barren Land": 0}


# Set the page layout and browser tab title
st.set_page_config(layout="wide", page_title="Satellite Image Analysis")

# Custom visible title with yellow color and large font
st.markdown(
    """
    <h1 style='color: yellow; font-size: 72px; font-weight: bold;'>
        Satellite Image Analysis
    </h1>
    """,
    unsafe_allow_html=True
)


# -------- Models --------
class DummyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DummyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 6 * 14 * 14)
        x = self.fc1(x)
        return x

def train_dummy_svm(features, labels):
    """Trains a dummy SVM model."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = svm.SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    return model, y_test, y_prob, accuracy

def train_dummy_cnn(images, labels, num_classes=2, epochs=10):
    """Trains a dummy CNN model."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Simulate dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = Image.fromarray(self.images[idx])
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            return image, label

    dataset = DummyDataset(images, labels, transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    model = DummyCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Simulate validation
    val_dataset = DummyDataset(images[:len(images)//5], labels[:len(labels)//5], transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return model, np.array(all_labels), np.array(all_probs), accuracy

# -------- Image Processing Functions --------
def preprocess_img(img, size=(64,64)):
    img = img.convert("RGB").resize(size)
    img_arr = np.array(img)/255.0
    return img_arr

def align_images(img1, img2):
    """Align images using ECC (Enhanced Correlation Coefficient) method"""
    # Convert PIL Images to numpy arrays
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    gray1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

    try:
        cc, warp_matrix = cv2.findTransformECC(gray1, gray2, warp_matrix,
                                             cv2.MOTION_EUCLIDEAN, criteria)
        aligned = cv2.warpAffine(img2_np, warp_matrix,
                                 (img1_np.shape[1], img1_np.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        diff_mask = cv2.absdiff(img1_np, aligned)
        diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_RGB2GRAY)
        _, black_mask = cv2.threshold(diff_mask, 30, 255, cv2.THRESH_BINARY_INV)
        aligned_black = cv2.bitwise_and(aligned, aligned, mask=black_mask)

        return Image.fromarray(aligned), Image.fromarray(aligned_black)
    except Exception as e:
        st.error(f"Image alignment failed: {e}")
        return img2, img2

def get_change_mask(img1, img2, threshold=30):
    # Ensure images are the same size
    img2 = img2.resize(img1.size)

    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, change_mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
    return change_mask.astype(np.uint8)

def classify_pixels(img_array, land_classes):
    """Assigns land classes to pixels based on a simplified rule."""
    classification = np.zeros(img_array.shape[:2], dtype=np.uint8)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            r, g, b = img_array[i, j]
            if b > 100 and r < 50 and g < 50:  # Simplified water detection
                classification[i, j] = 0  # Water
            elif g > r and g > b:  # Simplified vegetation detection
                classification[i, j] = 1  # Vegetation
            else:
                classification[i, j] = 2  # Barren Land
    counts = np.bincount(classification.flatten(), minlength=len(land_classes))
    total_pixels = classification.size
    percentages = {land_classes[i]: (counts[i] / total_pixels) * 100 for i in range(len(land_classes))}
    return percentages, classification

def detect_calamity(date1, date2, change_percentage):
    """Detects potential calamities based on changes and time difference"""
    date_diff = (date2 - date1).days

    if change_percentage > 0.15:
        if date_diff <= 10:
            return "⚠️ **Possible Flood:** Rapid and significant changes observed in a short period may indicate flooding."
        elif date_diff <= 30:
            return "🔥 **Possible Deforestation/Fire:** Significant loss of vegetation over a short term could suggest deforestation or wildfires."
        else:
            return "🏗️ **Possible Urbanization/Land Development:** Gradual yet significant increase in non-vegetated areas over time might indicate urbanization or land development."
    elif change_percentage > 0.05:
        return "🌱 **Noticeable Environmental Changes:** Changes beyond typical seasonal variations are observed, requiring further investigation."
    return "✅ **No Significant Environmental Calamity Detected:** Minimal changes observed between the two images."

def get_csv_bytes(data_dict):
    df = pd.DataFrame(list(data_dict.items()), columns=["Class", "Area (%)"])
    return df.to_csv(index=False).encode()

# -------- Pages --------
def page1():
    st.header("1. Model Selection")
    st.session_state.model_choice = st.selectbox("Select Analysis Model", ["SVM", "CNN"])
    if st.button("Next ➡️"):
        st.session_state.page = 2

def page2():
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
        st.session_state.before_date = st.date_input("BEFORE image date", value=st.session_state.before_date)
        st.session_state.before_file = st.file_uploader("Upload BEFORE image",
                                                        type=["png", "jpg", "tif"],
                                                        key="before")
        st.session_state.after_date = st.date_input("AFTER image date", value=st.session_state.after_date)
        st.session_state.after_file = st.file_uploader("Upload AFTER image",
                                                       type=["png", "jpg", "tif"],
                                                       key="after")

    if st.button("⬅️ Back"):
        st.session_state.page = 1
    if st.session_state.before_file and st.session_state.after_file:
        if st.button("Next ➡️"):
            try:
                before_img = Image.open(st.session_state.before_file).convert("RGB")
                after_img = Image.open(st.session_state.after_file).convert("RGB")

                # Align images using ECC method
                aligned_after, aligned_black = align_images(before_img, after_img)
                st.session_state.aligned_images = {
                    "before": before_img,
                    "after": aligned_after,
                    "aligned_black": aligned_black
                }

                # Calculate change mask
                st.session_state.change_mask = get_change_mask(before_img, aligned_after)

                # Simplified pixel-based classification for before and after images
                land_classes = ["Water", "Vegetation", "Barren Land"]
                before_arr = np.array(before_img)
                after_arr = np.array(aligned_after)
                st.session_state.before_classification, _ = classify_pixels(before_arr, land_classes)
                st.session_state.after_classification, _ = classify_pixels(after_arr, land_classes)

                # Train dummy models for demonstration
                dummy_features = np.random.rand(100, 10) # Replace with actual features
                dummy_labels = np.random.randint(0, 2, 100) # Replace with actual labels
                st.session_state.svm_model, svm_y_test, svm_y_prob, svm_accuracy = train_dummy_svm(dummy_features, dummy_labels)
                fpr_svm, tpr_svm, _ = roc_curve(svm_y_test, svm_y_prob)
                roc_auc_svm = auc(fpr_svm, tpr_svm)
                st.session_state.svm_metrics = {"fpr": fpr_svm, "tpr": tpr_svm, "roc_auc": roc_auc_svm, "accuracy": svm_accuracy}

                dummy_images = np.random.randint(0, 256, size=(50, 64, 64, 3), dtype=np.uint8) # Replace with actual image data
                dummy_cnn_labels = np.random.randint(0, 2, 50) # Replace with actual labels
                st.session_state.cnn_model, cnn_y_test, cnn_y_prob, cnn_accuracy = train_dummy_cnn(dummy_images, dummy_cnn_labels)
                fpr_cnn, tpr_cnn, _ = roc_curve(cnn_y_test, cnn_y_prob)
                roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
                st.session_state.cnn_metrics = {"fpr": fpr_cnn, "tpr": tpr_cnn, "roc_auc": roc_auc_cnn, "accuracy": cnn_accuracy}


                st.session_state.page = 3
            except Exception as e:
                st.error(f"Error processing images: {e}")

def page3():
    st.header("3. Aligned Images Comparison")

    if st.session_state.aligned_images is None:
        st.error("No aligned images found. Please upload images first.")
        st.session_state.page = 2
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(st.session_state.aligned_images["before"],
                 caption="
