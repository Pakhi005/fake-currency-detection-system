import streamlit as st
import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# --- Helper Functions (adapted from original script) ---
def detect_country(image, aspect_ratio=None):
    """Simple heuristic-based country detection using aspect ratio and average HSV color."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_color = np.mean(hsv, axis=(0, 1))
    hue, sat = avg_color[0], avg_color[1]
    
    if aspect_ratio is not None:
        if 2.30 <= aspect_ratio <= 2.40:
            return "United States Dollar (USD)"
        elif 1.90 <= aspect_ratio <= 2.29:
            return "Indian Rupee (INR)"
            
    if sat < 20:
        return "Unknown / Grayscale"
    if 10 <= hue <= 25 or 100 <= hue <= 170:
        return "Indian Rupee (INR)"
    elif 35 <= hue <= 70:
        return "United States Dollar (USD)"
    else:
        return "Indian Rupee (INR)"

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def perform_edge_detection(image):
    edges = cv2.Canny(image, threshold1=30, threshold2=100)
    return edges

def extract_features(image, original_image):
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = {}
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        aspect_ratio = float(w) / h
        features['aspect_ratio'] = aspect_ratio
        features['area'] = cv2.contourArea(largest_contour)
        features['bounding_box'] = (x, y, w, h)
        
    return features

def calculate_ssim(imageA, imageB):
    if not HAS_SKIMAGE: return None
    if imageA.shape != imageB.shape:
        imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))
    if len(imageA.shape) == 3: imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    if len(imageB.shape) == 3: imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(imageA, imageB, full=True)
    return score

# --- UI Layout ---
st.set_page_config(page_title="Fake Currency Detector", page_icon="💵", layout="wide")

st.title("💵 Fake Currency Detection System")
st.markdown("Upload or capture a currency note image to analyze its authenticity and detect its country using image processing techniques.")

st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Select Input Method", ["Upload Image", "Use Camera"])

test_file = None
if input_method == "Upload Image":
    test_file = st.sidebar.file_uploader("Upload Test Currency Image", type=["jpg", "png", "jpeg"])
else:
    test_file = st.camera_input("Take a picture of the currency")

st.sidebar.header("Reference Image")
ref_file = st.sidebar.file_uploader("Upload Real Reference Image (Optional)", type=["jpg", "png", "jpeg"])

if test_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(test_file.read()), dtype=np.uint8)
    test_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Process
    processed_test = preprocess_image(test_img)
    edges_test = perform_edge_detection(processed_test)
    features = extract_features(edges_test, test_img)
    
    # UI Layout: Show images in columns
    st.subheader("Image Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
    with col2:
        st.image(edges_test, caption="Edge Detection", use_container_width=True, clamp=True)
        
    annotated_img = test_img.copy()
    if 'bounding_box' in features:
        x, y, w, h = features['bounding_box']
        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 5)
        with col3:
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Detected Region", use_container_width=True)
    else:
        with col3:
            st.write("Could not detect currency region.")
            
    st.divider()
    
    # Prediction Logic
    st.subheader("Results")
    
    country = detect_country(test_img, features.get('aspect_ratio', None))
    st.info(f"🌍 **Detected Country / Currency Type:** {country}")
    
    is_fake = False
    reasons = []
    
    if 'aspect_ratio' in features:
        ar = features['aspect_ratio']
        # Check both horizontal and vertical standard orientations
        if not ((1.5 <= ar <= 3.0) or (0.33 <= ar <= 0.66)):
            is_fake = True
            reasons.append(f"Anomalous Aspect Ratio ({ar:.2f})")
            
    if ref_file is not None:
        ref_bytes = np.asarray(bytearray(ref_file.read()), dtype=np.uint8)
        ref_img = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)
        processed_ref = preprocess_image(ref_img)
        similarity = calculate_ssim(processed_test, processed_ref)
        
        if similarity is not None:
            st.metric("Structural Similarity Index (SSIM)", f"{similarity*100:.2f}%")
            if similarity < 0.60:
                is_fake = True
                reasons.append(f"Low Structural Similarity ({similarity:.2f})")
        else:
            st.warning("scikit-image is not installed. Skipping SSIM check.")
            
    # Final Verdict Display
    if is_fake:
        st.error("🚨 **Verdict: FAKE CURRENCY DETECTED**")
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.success("✅ **Verdict: REAL CURRENCY** (or unverified anomalies not found)")
else:
    st.info("Please upload a test image from the sidebar to begin.")
