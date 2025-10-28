import streamlit as st
import torch
import cv2
import sys
import os
import numpy as np
from PIL import Image
from torchvision import transforms

# --- Import your custom modules ---
# This works because streamlit_app.py is in the root,
# and your 'models' and 'explainability' folders are also in the root.
try:
    from models.model import load_trained_model
    from explainability.gradcam import get_gradcam_heatmap
    from explainability.roi_extraction import extract_roi
    from explainability.clip_labelling import CLIPArtifactDetector
except ImportError as e:
    st.error(
        f"Error: Could not import custom modules (e.g., models, explainability). {e}"
        "\n\nPlease make sure this `streamlit_app.py` file is in your project's "
        "**root directory** (`D:\Projects\Adobe-Internal`), at the same level as your 'models' and "
        "'explainability' folders."
    )
    st.stop()


# --- 1. MODEL & DETECTOR LOADING (CACHED) ---
# This ensures your models only load ONCE, making the app fast.

@st.cache_resource
def load_app_model(model_weights="models/weights/best_resnet50.pth"):
    """
    Loads the trained ResNet-50 model.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_trained_model(weight_path=model_weights, num_classes=2)
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error(f"Error: Model weights not found at {model_weights}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_resource
def load_clip_detector():
    """Loads the CLIP artifact detector."""
    try:
        return CLIPArtifactDetector()
    except Exception as e:
        st.error(f"Error loading CLIP: {e}")
        return None

# --- 2. IMAGE ANALYSIS PIPELINE ---
# This is the logic from your 'analyze_image' function,
# adapted to return all intermediate steps for the UI.

def run_analysis_pipeline(pil_image, model, device, clip_detector):
    """
    Runs the full Task 1 (Detection) and Task 2 (Explanation) pipeline.
    Returns:
        - prediction_label (str)
        - confidence_percent (float)
        - overlayed_image (np.array)
        - heatmap_rgb (np.array)
        - artifact_results (list)
    """
    
    # 1. PREPARE IMAGE
    img_rgb = np.array(pil_image)
    # Convert RGB to BGR for cv2 functions that might need it (like extract_roi)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_resized_pil = pil_image.resize((224, 224))
    img_resized_rgb = np.array(img_resized_pil)
    
    tensor = transform(img_resized_pil).unsqueeze(0).to(device)

    # 2. TASK 1: AI IMAGE DETECTION
    class_names = ["Real", "AI-Generated"] # Assuming 0: Real, 1: AI
    prediction_label = "Error"
    confidence_percent = 0.0
    
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
        prediction_label = class_names[predicted_class.item()]
        confidence_percent = confidence.item() * 100

    # 3. TASK 2: ARTIFACT IDENTIFICATION
    target_layer = model.backbone.layer4[-1] 
    heatmap = get_gradcam_heatmap(model, tensor, target_layer)
    
    # Create pure heatmap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlayed_image = cv2.addWeighted(img_resized_rgb, 0.6, heatmap_rgb, 0.4, 0)

    # Extract ROIs
    rois, boxes = extract_roi(heatmap, img_bgr)

    # Label Artifacts
    artifact_results = []
    if not rois:
        print("No significant artifact regions found.")
    else:
        for i, roi_pil in enumerate(rois):
            artifacts = clip_detector.classify_artifacts(roi_pil, top_k=3)
            artifact_results.append({
                "roi_image": roi_pil,
                "box": boxes[i],
                "artifacts": artifacts
            })

    # 4. RETURN ALL RESULTS FOR STREAMLIT TO DISPLAY
    return prediction_label, confidence_percent, overlayed_image, heatmap_rgb, artifact_results


# --- 3. UI DRAWING FUNCTIONS ---

def draw_home_page():
    """Draws the Welcome/Introduction page."""
    
    st.title("🤖 Welcome to the AI Image Detector")
    st.header("AI-Generated Image Detection and Explanation")
    
    st.markdown("""
    This application is a demonstration of our project for the **Inter-IIT Mid Prep Internal Hackathon**.
    
    Our system is designed to solve two core problems:
    1.  **Detect** if an image is real or created by AI.
    2.  **Explain** *why* an image is flagged as AI-generated by identifying visual artifacts.
    """)

    # Real vs. Fake image placeholders
    col1, col2 = st.columns(2)
    with col1:
        # --- FIX: Added 'https://' ---
        st.image(
            "./images/real.png", 
            caption="A real photograph.", 
            use_container_width=True
        )
    with col2:
        # --- FIX: Added 'https://' ---
        st.image(
            "./images/Fake.png", 
            caption="An AI-generated image with subtle flaws.", 
            use_container_width=True
        )
        
    st.header("Our 'Semantic Explainer' Pipeline")
    st.markdown("""
    When an image is flagged as "AI-Generated," we don't just stop there. 
    We use a 3-step pipeline (based on **Grad-CAM++** and **CLIP**) to find and label the artifacts.
    """)
    
    # Workflow diagram from PPT
    with st.expander("See the 3-Step Explanation Workflow"):
        st.markdown("""
        1.  **Step 1: Grad-CAM++ Heatmap**
            -   We generate a "heatmap" to see which parts of the image our AI model found most suspicious.
            
        2.  **Step 2: Region of Interest (ROI) Extraction**
            -   Our algorithm identifies the "hottest" areas of the map and crops them out.
            
        3.  **Step 3: CLIP Semantic Labeling**
            -   Each cropped region is passed to **CLIP**, a model that understands both images and text.
            -   We ask CLIP: "Does this image patch look like 'deformed hands,' 'waxy skin,' or 'garbled text'?"
            -   The system then provides a human-readable label for the artifact.
        """)

    st.markdown("---")
    
    # Button to navigate to the app
    if st.button("Proceed to the Live Detector →", type="primary"):
        st.session_state.page = "app"
        st.rerun()


def draw_app_page():
    """Draws the main application page for analysis."""
    
    if st.button("‹ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("🤖 Live Detector & Explainer")
    st.markdown(
        "Upload an image to begin the analysis. The workflow will be shown below."
    )

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        # Load model and detector
        model, device = load_app_model()
        clip_detector = load_clip_detector()

        if model is None or clip_detector is None:
            st.error("Application is not configured correctly. Please check logs.")
        else:
            pil_image = Image.open(uploaded_file).convert("RGB")
            
            with st.spinner("🕵️ Analyzing image... This may take a moment."):
                try:
                    (
                        prediction, 
                        confidence, 
                        overlay, 
                        heatmap,
                        artifact_results
                    ) = run_analysis_pipeline(pil_image, model, device, clip_detector)
                
                    # --- TASK 1: DETECTION RESULT ---
                    st.markdown("---")
                    st.header("TASK 1: AI IMAGE DETECTION")
                    
                    if prediction == "AI-Generated":
                        st.error(f"**Prediction: AI-Generated**")
                    else:
                        st.success(f"**Prediction: Real**")
                    
                    st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
                    
                    # --- TASK 2: EXPLANATION WORKFLOW ---
                    if prediction == "AI-Generated":
                        st.markdown("---")
                        st.header("TASK 2: EXPLANATION WORKFLOW")

                        # Step 1 & 2: Original vs. Heatmap
                        st.subheader("Step 1: Grad-CAM Heatmap")
                        st.markdown("We generate a 'heatmap' to see which parts of the image our model found most suspicious.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(pil_image, caption="1. Uploaded Image", 
                                     use_container_width=True)
                        with col2:
                            st.image(heatmap, caption="2. Grad-CAM Heatmap", 
                                     use_container_width=True)

                        # Step 3: Overlay
                        st.subheader("Step 2: Heatmap Overlay")
                        st.image(overlay, caption="3. Heatmap Overlaid on Image", 
                                 use_container_width=True)

                        # Step 4: ROI + CLIP
                        st.subheader("Step 3: Artifact Analysis (ROI + CLIP)")
                        st.markdown("Finally, we extract each suspicious region (ROI) and use CLIP to label what the artifact might be.")
                        
                        if not artifact_results:
                            st.info("No significant artifacts were found to label.")
                        else:
                            st.subheader(f"Found {len(artifact_results)} Suspicious Region(s)")
                            for i, item in enumerate(artifact_results):
                                st.markdown(f"#### Region of Interest #{i+1}")
                                
                                r_col1, r_col2 = st.columns([1, 2])
                                
                                with r_col1:
                                    st.image(
                                        item["roi_image"], 
                                        caption=f"Cropped Patch",
                                        use_container_width=True
                                    )
                                with r_col2:
                                    st.markdown("**Potential Artifacts (from CLIP):**")
                                    for label, score in item['artifacts']:
                                        st.text(f"  - {label} (Confidence: {score:.2f})")
                                        
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.error(
                        "This might be due to an issue with the model or "
                        "one of the explainability modules. Check your terminal for logs."
                    )

# --- 4. MAIN APP LOGIC ---

# Set page config
st.set_page_config(
    layout="wide", 
    page_title="AI Image Detector",
    page_icon="🤖"
)

# --- Sidebar (Content from your PPT, visible on both pages) ---
with st.sidebar:
    st.title("INTER-IIT MID PREP")
    st.caption("Internal Hackathon")
    
    st.markdown("---")
    st.header("Problem Statement")
    st.info(
        "This application is a demo for the **Inter-IIT Mid Prep "
        "Internal Hackathon** problem statement: "
        "**AI-Generated Image Detection and Explanation**."
    )
    
    st.markdown("---")
    st.header("Sponsors")
    st.markdown("- Adobe\n- Veritas AI")
    
    st.markdown("---")
    st.header("Our Team")
    st.markdown(
        """
        - **Rishi Chauhan (Team Leader)**
        - **Dashpreet Singh**
        - **Daksh Singhal**
        - **Hemant Nagar**
        - **Abhay Mishra**
        - **Anag Mahipal** """
    ) # Note: 'Anag Mahipal' is based on the PPT image

# --- Page Navigation ---
# Use session_state to track the current "page"
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Draw the correct page
if st.session_state.page == 'home':
    draw_home_page()
else:
    draw_app_page()