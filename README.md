# AI Image Detector & Explainer

## Overview

This project is a comprehensive solution for **AI-Generated Image Detection and Explanation**, developed for the **Inter-IIT Mid Prep Internal Hackathon**. It addresses the challenge of distinguishing real photographs from AI-generated images and provides interpretable explanations for detections using advanced computer vision techniques.

The system combines a deep learning classifier with explainability tools to not only detect AI-generated content but also identify and label specific visual artifacts that give away the synthetic nature of the image.

## Features

### Core Functionality
- **AI Image Detection**: Classifies images as "Real" or "AI-Generated" using a fine-tuned ResNet-50 model
- **Explainability Pipeline**: When an image is flagged as AI-generated, provides detailed explanations using:
  - **Grad-CAM++ Heatmaps**: Visualizes which parts of the image the model found most suspicious
  - **Region of Interest (ROI) Extraction**: Identifies and crops suspicious areas
  - **CLIP Semantic Labeling**: Uses OpenAI's CLIP model to label artifacts (e.g., "deformed hands", "waxy skin", "garbled text")

### User Interface
- **Streamlit Web App**: Interactive web interface for easy image upload and analysis
- **Real-time Analysis**: Processes images on-the-fly with confidence scores
- **Visual Results**: Displays heatmaps, overlays, and artifact labels with confidence scores

### Technical Highlights
- **Model Architecture**: ResNet-50 backbone with custom classification head
- **Explainability Stack**: Grad-CAM++ for attribution, CLIP for semantic understanding
- **Preprocessing**: Robust image preprocessing pipeline with normalization
- **Performance**: Optimized for both accuracy and interpretability

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Adobe-Internal
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights**:
   - Place the trained model weights at `models/weights/best_resnet50.pth`
   - If not available, you'll need to train the model using the provided training scripts

5. **Prepare data** (optional, for training):
   - Place your dataset in the `dataset/` directory
   - Preprocessed data should go in `dataset_preprocessed/`

## Usage

### Running the Web App

Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

This will launch a web interface at `http://localhost:8501` where you can:
1. Upload images (JPG, JPEG, PNG)
2. View detection results with confidence scores
3. Explore explainability visualizations for AI-generated images

### Programmatic Usage

You can also use the analysis pipeline directly in Python:

```python
from pipeline.main import analyze_image

# Analyze a single image
results = analyze_image("path/to/your/image.jpg")
```

### Training the Model

Use the provided training script:
```python
# Example training command (adjust paths as needed)
python models/model_train.py
```

## Project Structure

```
Adobe-Internal/
├── streamlit_app.py          # Main Streamlit web application
├── requirements.txt          # Python dependencies
├── models/
│   ├── model.py             # ResNet-50 classifier implementation
│   ├── model_train.py       # Training script
│   └── weights/             # Model weights directory
├── explainability/
│   ├── gradcam.py           # Grad-CAM++ implementation
│   ├── roi_extraction.py    # ROI extraction algorithm
│   └── clip_labelling.py    # CLIP-based artifact labeling
├── pipeline/
│   └── main.py              # Main analysis pipeline
├── dataset/                 # Raw dataset
├── dataset_preprocessed/    # Preprocessed dataset
├── images/                  # Sample images (real/fake)
├── model.ipynb              # Model development notebook
├── data_modify.ipynb        # Data preprocessing notebook
├── .gitignore
├── LICENSE
└── README.md
```

## Methodology

### Detection Pipeline
1. **Image Preprocessing**: Resize to 224x224, normalize using ImageNet statistics
2. **Classification**: ResNet-50 predicts "Real" vs "AI-Generated"
3. **Confidence Scoring**: Softmax probabilities provide confidence levels

### Explanation Pipeline (for AI-Generated images)
1. **Grad-CAM++**: Generate attention heatmap highlighting suspicious regions
2. **ROI Extraction**: Identify and crop high-attention areas
3. **CLIP Classification**: Label each ROI with potential artifacts using zero-shot classification

## Team

- **Rishi Chauhan** (Team Leader)
- **Dashpreet Singh**
- **Daksh Singhal**
- **Hemant Nagar**
- **Abhay Mishra**
- **Anurag Mahipal**


## Hackathon Context

This project was developed as part of the **Inter-IIT Mid Prep Internal Hackathon**, focusing on advancing AI safety and interpretability in computer vision applications.

## Dependencies

Key libraries used:
- **PyTorch**: Deep learning framework
- **TorchVision**: Computer vision utilities
- **OpenCV**: Image processing
- **Streamlit**: Web app framework
- **Transformers**: CLIP model access
- **Grad-CAM**: Explainability library
- **TIMM**: Model zoo and utilities

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

For contributions or questions, please contact the team members listed above.

## Acknowledgments

- OpenAI for CLIP model
- PyTorch team for the deep learning framework
- Research community for Grad-CAM and related explainability techniques
