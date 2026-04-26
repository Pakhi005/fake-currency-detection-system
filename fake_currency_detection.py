import cv2
import numpy as np
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: 'scikit-image' is not installed. Structural Similarity (SSIM) checks will be disabled.")
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    """Image acquisition."""
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return None
    image = cv2.imread(image_path)
    return image

def preprocess_image(image):
    """RGB to grey conversion and noise reduction."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return blurred

def detect_country(image, aspect_ratio=None):
    """Heuristic-based country detection using aspect ratio and average HSV color."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_color = np.mean(hsv, axis=(0, 1))
    hue, sat = avg_color[0], avg_color[1]
    
    # Heuristics based on Aspect Ratio if available
    # USD notes are uniformly 156 x 66.3 mm (Aspect Ratio ~ 2.35)
    # Indian notes vary but generally:
    # 500 INR is 150 x 66 mm (~2.27)
    # 200 INR is 146 x 66 mm (~2.21)
    # 100 INR is 142 x 66 mm (~2.15)
    # Smaller INR notes have aspect ratios around 1.95 - 2.05
    
    # If aspect ratio is close to 2.35 (e.g. > 2.30), it's highly likely USD.
    if aspect_ratio is not None:
        if 2.30 <= aspect_ratio <= 2.40:
            return "United States Dollar (USD)"
        elif 1.90 <= aspect_ratio <= 2.29:
            return "Indian Rupee (INR)"
            
    # Fallback to color if aspect ratio is not conclusive or missing
    if sat < 20:
        return "Unknown / Grayscale"
    if 10 <= hue <= 25 or 100 <= hue <= 170:
        return "Indian Rupee (INR)" # Covers purple, blue, brown notes
    elif 35 <= hue <= 70:
        return "United States Dollar (USD)"
    else:
        return "Indian Rupee (INR)" # Default guess for other colors based on user location

def perform_edge_detection(image):
    """Classification through edge detection."""
    # Apply Canny edge detection
    edges = cv2.Canny(image, threshold1=30, threshold2=100)
    return edges

def extract_features(image, original_image):
    """Feature management: Extract contours and aspect ratio."""
    # Find contours
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = {}
    
    if contours:
        # Find the largest contour which we assume is the note
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        features['aspect_ratio'] = float(w) / h
        features['area'] = cv2.contourArea(largest_contour)
        features['bounding_box'] = (x, y, w, h)
        
        # Extract the ROI (Region of Interest)
        roi = original_image[y:y+h, x:x+w]
        features['roi'] = roi
    
    return features

def calculate_ssim(imageA, imageB):
    """Calculate Structural Similarity Index between two images."""
    if not HAS_SKIMAGE:
        return None
        
    # Ensure images have the same dimensions for SSIM
    if imageA.shape != imageB.shape:
        imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))
        
    # Convert to grayscale for SSIM
    if len(imageA.shape) == 3:
        imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    if len(imageB.shape) == 3:
        imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        
    score, diff = ssim(imageA, imageB, full=True)
    return score

def detect_fake_currency(test_image_path=None, reference_image_path=None, test_img=None):
    """Main pipeline for fake currency detection."""
    if test_img is None:
        print(f"Processing test image: {test_image_path}")
        test_img = load_image(test_image_path)
    
    if test_img is None:
        return
        
    # 1. Preprocessing (RGB to Gray)
    processed_test = preprocess_image(test_img)
    
    # 2. Edge Detection
    edges_test = perform_edge_detection(processed_test)
    
    # 3. Feature Extraction
    features = extract_features(edges_test, test_img)
    
    print("\n--- Extracted Features ---")
    if 'aspect_ratio' in features:
        aspect_ratio = features['aspect_ratio']
        print(f"Aspect Ratio: {aspect_ratio:.2f}")
    else:
        aspect_ratio = None
    
    country = detect_country(test_img, aspect_ratio)
    print(f"Detected Country / Currency Type: {country}")
    
    # 4. Prediction logic
    is_fake = False
    prediction_reason = []
    
    # Example heuristic: Indian notes have specific aspect ratios
    # e.g., 2000 rupee note is 166mm x 66mm (aspect ratio ~2.51)
    # e.g., 500 rupee note is 150mm x 66mm (aspect ratio ~2.27)
    # This is a very simplified rule-based check. You can adjust these bounds.
    if 'aspect_ratio' in features:
        aspect_ratio = features['aspect_ratio']
        if aspect_ratio < 1.5 or aspect_ratio > 3.0:
            is_fake = True
            prediction_reason.append("Anomalous Aspect Ratio")
            
    # If a reference real image is provided, do SSIM comparison
    if reference_image_path:
        print(f"Comparing with reference image: {reference_image_path}")
        ref_img = load_image(reference_image_path)
        if ref_img is not None:
            processed_ref = preprocess_image(ref_img)
            similarity_score = calculate_ssim(processed_test, processed_ref)
            
            if similarity_score is not None:
                print(f"Similarity Score (SSIM): {similarity_score:.4f}")
                
                if similarity_score < 0.60: # Threshold for fake based on structural similarity
                    is_fake = True
                    prediction_reason.append(f"Low Structural Similarity: {similarity_score:.4f}")
            else:
                print("Skipping SSIM check since scikit-image is not installed.")
                
    # Display results
    print("\n--- Final Prediction ---")
    if is_fake:
        print("Verdict: FAKE CURRENCY DETECTED")
        print("Reasons:", ", ".join(prediction_reason))
    else:
        print("Verdict: REAL CURRENCY (or unverified without reference)")
        
    # Plotting results for visual analysis
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("Original Image (Image Acquisition)")
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 2, 2)
    plt.title("RGB to Grayscale & Blurred")
    plt.imshow(processed_test, cmap='gray')
    
    plt.subplot(2, 2, 3)
    plt.title("Classification through Edge Detection")
    plt.imshow(edges_test, cmap='gray')
    
    if 'bounding_box' in features:
        plt.subplot(2, 2, 4)
        plt.title("Feature Management: Detected Note Region")
        x, y, w, h = features['bounding_box']
        annotated_img = test_img.copy()
        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fake Currency Detection System")
    parser.add_argument("--test_image", "-t", help="Path to the test image of the currency note", default=None)
    parser.add_argument("--ref", "-r", dest="ref_image", help="Optional: Path to a real reference image for structural comparison", default=None)
    parser.add_argument("--camera", "-c", action="store_true", help="Use webcam to capture the test image")
    
    args = parser.parse_args()
    
    print("Fake Currency Detection System Initialized.")
    
    if args.camera:
        print("Opening webcam... Press 'q' to quit.")
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            # Real-time processing
            processed_test = preprocess_image(frame)
            edges_test = perform_edge_detection(processed_test)
            features = extract_features(edges_test, frame)
            
            is_fake = False
            reason = ""
            
            if 'aspect_ratio' in features:
                aspect_ratio = features['aspect_ratio']
                if aspect_ratio < 1.5 or aspect_ratio > 3.0:
                    is_fake = True
                    reason = "Anomaly"
                    
            # Annotate frame
            if 'bounding_box' in features and features['area'] > 5000: # Ensure minimum area so it doesn't pick up small noise
                x, y, w, h = features['bounding_box']
                
                # Use aspect ratio for better country detection
                ar = features.get('aspect_ratio', None)
                
                # Extract ROI for country detection to make it slightly more accurate
                roi = frame[y:y+h, x:x+w]
                country = detect_country(roi, ar)
                
                color = (0, 0, 255) if is_fake else (0, 255, 0)
                text = f"FAKE ({reason})" if is_fake else "REAL"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, f"{text} | {country}", (x, max(15, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(frame, "Show Currency Note...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
            cv2.imshow("Real-Time Currency Detection - Press 'q' to quit", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
            
    elif args.test_image:
        detect_fake_currency(test_image_path=args.test_image, reference_image_path=args.ref_image)
    else:
        print("Please provide either --test_image or --camera flag.")

