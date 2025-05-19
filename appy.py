import streamlit as st
import numpy as np
import cv2
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from io import BytesIO
import base64

# --- Dummy CNN Model Definition ---
def DummyCNN():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        Flatten(),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Image Preprocessing Functions ---
def load_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def align_images(before_img, after_img):
    # Convert images to grayscale
    before_gray = cv2.cvtColor(before_img, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after_img, cv2.COLOR_RGB2GRAY)

    # Use ORB for keypoint detection
    orb = cv2.ORB_create(5000)
    keypoints1, descriptors1 = orb.detectAndCompute(before_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(after_gray, None)

    # Match descriptors using BFMatcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, _ = cv2.findHomography(points2, points1, cv2.RANSAC)
    height, width, _ = before_img.shape
    aligned_after = cv2.warpPerspective(after_img, h, (width, height))
    aligned_black = cv2.absdiff(before_img, aligned_after)
    return aligned_after, aligned_black

# --- Classification Functions ---
def dummy_classification(image, model_type="cnn"):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64)) / 255.0
    features = resized.reshape(-1, 64, 64, 1) if model_type == "cnn" else resized.flatten().reshape(1, -1)

    if model_type == "cnn":
        model = DummyCNN()
        labels = np.random.randint(0, 2, size=(100,))
        dummy_images = np.random.rand(100, 64, 64, 1)
        model.fit(dummy_images, to_categorical(labels), epochs=1, verbose=0)
        prediction = model.predict(features)[0]
        return prediction
    else:
        X = np.random.rand(100, 4096)
        y = np.random.randint(0, 2, size=(100,))
        clf = svm.SVC(probability=True)
        clf.fit(X, y)
        pred = clf.predict_proba(features)[0]
        return pred

# --- Utility Functions ---
def get_csv_bytes(data):
    csv = "\n".join([",".join(map(str, row)) for row in data])
    return BytesIO(csv.encode())

# --- Streamlit Page Definitions ---
def page1():
    st.title("Satellite Image Analysis App")
    st.markdown("""
        This application helps in aligning satellite images, performing change detection, and comparing CNN and SVM models for land classification.
        Use the sidebar to navigate between pages.
    """)
    if st.button("Start ➡️"):
        st.session_state.page = 2

def page2():
    st.header("1. Upload Satellite Images")

    before_image = st.file_uploader("Upload BEFORE Image", type=["jpg", "png", "jpeg"], key="before")
    after_image = st.file_uploader("Upload AFTER Image", type=["jpg", "png", "jpeg"], key="after")

    if before_image and after_image:
        st.session_state.before_img = load_image(before_image)
        st.session_state.after_img = load_image(after_image)

        with st.spinner("Aligning images..."):
            aligned_after, aligned_black = align_images(st.session_state.before_img, st.session_state.after_img)
            st.session_state.aligned_images = {
                "before": st.session_state.before_img,
                "after": st.session_state.after_img,
                "aligned_after": aligned_after,
                "aligned_black": aligned_black
            }
        st.success("Images aligned successfully!")

    if st.button("⬅️ Back"):
        st.session_state.page = 1
    if st.button("Next ➡️"):
        if "aligned_images" in st.session_state:
            st.session_state.page = 3
        else:
            st.error("Please upload and align images first.")

def page3():
    st.header("2. Aligned Images Comparison")

    if st.session_state.aligned_images is None:
        st.error("No aligned images found. Please upload images first.")
        st.session_state.page = 2
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(st.session_state.aligned_images["before"], caption="BEFORE Image", use_column_width=True)
    with col2:
        st.image(st.session_state.aligned_images["after"], caption="Aligned AFTER Image", use_column_width=True)
    with col3:
        st.image(st.session_state.aligned_images["aligned_black"], caption="Aligned Difference", use_column_width=True)

    if st.button("⬅️ Back"):
        st.session_state.page = 2
    if st.button("Next ➡️"):
        st.session_state.page = 4

def page4():
    st.header("3. Land Classification Comparison (CNN vs SVM)")

    image = st.session_state.aligned_images["aligned_black"]
    cnn_pred = dummy_classification(image, model_type="cnn")
    svm_pred = dummy_classification(image, model_type="svm")

    st.subheader("CNN Prediction")
    st.write(f"Class 0: {cnn_pred[0]:.2f}")
    st.write(f"Class 1: {cnn_pred[1]:.2f}")

    st.subheader("SVM Prediction")
    st.write(f"Class 0: {svm_pred[0]:.2f}")
    st.write(f"Class 1: {svm_pred[1]:.2f}")

    # ROC Curve
    fpr_cnn, tpr_cnn, _ = roc_curve([0, 1], cnn_pred)
    fpr_svm, tpr_svm, _ = roc_curve([0, 1], svm_pred)
    auc_cnn = auc(fpr_cnn, tpr_cnn)
    auc_svm = auc(fpr_svm, tpr_svm)

    fig, ax = plt.subplots()
    ax.plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC = {auc_cnn:.2f})')
    ax.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc='lower right')
    st.pyplot(fig)

    if st.button("⬅️ Back"):
        st.session_state.page = 3

# --- App Controller ---
def main():
    if "page" not in st.session_state:
        st.session_state.page = 1
        st.session_state.before_img = None
        st.session_state.after_img = None
        st.session_state.aligned_images = None

    if st.session_state.page == 1:
        page1()
    elif st.session_state.page == 2:
        page2()
    elif st.session_state.page == 3:
        page3()
    elif st.session_state.page == 4:
        page4()

if __name__ == "__main__":
    main()
