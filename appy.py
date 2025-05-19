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

# Unique key generator
def widget_key(name):
    return f"{name}_{st.session_state.get('key_counter', 0)}"

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 1
    st.session_state.key_counter = 0
    st.session_state.heatmap_overlay_svm = None
    st.session_state.heatmap_overlay_cnn = None
    st.session_state.aligned_images = None
    st.session_state.change_mask = None
    st.session_state.classification_svm = None
    st.session_state.classification_cnn = None
    st.session_state.before_date = datetime(2023, 1, 1)
    st.session_state.after_date = datetime(2023, 6, 1)
    st.session_state.before_file = None
    st.session_state.after_file = None
    st.session_state.model_choice = "SVM"
    st.session_state.svm_roc_fig = None
    st.session_state.cnn_roc_fig = None
    st.session_state.svm_accuracy = None
    st.session_state.cnn_accuracy = None
    st.session_state.classification_before_svm = {"Vegetation": 50, "Land": 30, "Water": 20}
    st.session_state.classification_before_cnn = {"Vegetation": 55, "Land": 25, "Developed": 20}

# Increment key counter on page change
if st.session_state.get('last_page') != st.session_state.page:
    st.session_state.key_counter += 1
    st.session_state.last_page = st.session_state.page

# Page config
st.set_page_config(layout="wide", page_title="Satellite Image Analysis")

# Title
st.markdown("""
    <h1 style='color: yellow; font-size: 72px; font-weight: bold; text-align: center;'>
        Satellite Image Analysis
    </h1>
""", unsafe_allow_html=True)

# Models
class DummyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6*14*14, 3)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 6*14*14)
        x = self.fc1(x)
        return x

cnn_model = DummyCNN()
cnn_model.eval()
svm_model = svm.SVC(probability=True)

# Image processing functions
def preprocess_img(img, size=(64,64)):
    img = img.convert("RGB").resize(size)
    return np.array(img)/255.0

def align_images(img1, img2):
    try:
        img1_np = np.array(img1.convert("RGB"))
        img2_np = np.array(img2.convert("RGB"))
        gray1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
        _, warp_matrix = cv2.findTransformECC(gray1, gray2, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
        aligned = cv2.warpAffine(img2_np, warp_matrix, (img1_np.shape[1], img1_np.shape[0]), 
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        diff_mask = cv2.absdiff(img1_np, aligned)
        diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_RGB2GRAY)
        _, black_mask = cv2.threshold(diff_mask, 30, 255, cv2.THRESH_BINARY_INV)
        aligned_black = cv2.bitwise_and(aligned, aligned, mask=black_mask)
        return Image.fromarray(aligned), Image.fromarray(aligned_black)
    except:
        return img2, img2

def get_change_mask(img1, img2, threshold=30):
    img2 = img2.resize(img1.size)
    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, change_mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
    return change_mask.astype(np.uint8)

# Classification functions
def classify_land_svm(img_arr):
    features = img_arr.flatten()[:100]
    labels = np.random.randint(0, 3, 50)
    svm_model.fit(np.random.rand(50, 100), labels)
    probabilities = svm_model.predict_proba(features.reshape(1, -1))[0]
    classes = ["Vegetation", "Land", "Water"]
    return {classes[i]: prob*100 for i, prob in enumerate(probabilities)}

def classify_land_cnn(img):
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = cnn_model(img_tensor)
        probabilities = torch.softmax(output, dim=1).numpy()[0]
        classes = ["Vegetation", "Land", "Developed"]
        return {classes[i]: prob*100 for i, prob in enumerate(probabilities)}

# Pages
def page1():
    st.header("1. Model Selection")
    st.session_state.model_choice = st.selectbox(
        "Select Analysis Model", 
        ["SVM", "CNN"],
        key=widget_key("model_select")
    )
    if st.button("Next ➡️", key=widget_key("page1_next")):
        st.session_state.page = 2

def page2():
    st.markdown("""
        <h2 style='font-size: 36px; color: white;'>
            2. Image Upload & Dates
        </h2>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.session_state.before_date = st.date_input(
            "BEFORE image date", 
            value=st.session_state.before_date,
            key=widget_key("before_date")
        )
        st.session_state.before_file = st.file_uploader(
            "Upload BEFORE image",
            type=["png", "jpg", "tif"],
            key=widget_key("before_upload")
        )
        st.session_state.after_date = st.date_input(
            "AFTER image date", 
            value=st.session_state.after_date,
            key=widget_key("after_date")
        )
        st.session_state.after_file = st.file_uploader(
            "Upload AFTER image",
            type=["png", "jpg", "tif"],
            key=widget_key("after_upload")
        )

    if st.button("⬅️ Back", key=widget_key("page2_back")):
        st.session_state.page = 1
    if st.session_state.before_file and st.session_state.after_file:
        if st.button("Next ➡️", key=widget_key("page2_next")):
            try:
                before_img = Image.open(st.session_state.before_file)
                after_img = Image.open(st.session_state.after_file)
                aligned_after, aligned_black = align_images(before_img, after_img)
                st.session_state.aligned_images = {
                    "before": before_img,
                    "after": aligned_after,
                    "aligned_black": aligned_black
                }
                st.session_state.change_mask = get_change_mask(before_img, aligned_after)
                
                if st.session_state.model_choice == "SVM":
                    after_arr = preprocess_img(aligned_after)
                    st.session_state.classification_svm = classify_land_svm(after_arr)
                    st.session_state.classification = st.session_state.classification_svm
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
                elif st.session_state.model_choice == "CNN":
                    st.session_state.classification_cnn = classify_land_cnn(aligned_after)
                    st.session_state.classification = st.session_state.classification_cnn
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
                st.session_state.page = 3
            except Exception as e:
                st.error(f"Error processing images: {e}")

# [Rest of your page functions (page3-page6) with similar key updates...]

# Main app flow
if st.session_state.page == 1:
    page1()
elif st.session_state.page == 2:
    page2()
# [Add similar conditions for other pages...]

if __name__ == "__main__":
    st.runtime.legacy_caching.clear_cache()
