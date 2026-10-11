"""
app.py
======
Streamlit Web Application for:
"AI-Based Road Waterlogging Detection and Severity Estimation Using Deep Learning"

Features:
- Cached PyTorch model inference for low latency.
- Real-time road waterlogging detection and confidence assessment.
- Explainable visual localization using Grad-CAM / segmentation masks.
- Transparent, documented affected area calculation and severity estimation.
- Graceful error handling for missing models, invalid inputs, and uncorrupted uploads.
"""

import os
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Road Waterlogging Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A, #3B82F6, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
    }
    .badge-detected {
        color: #DC2626;
    }
    .badge-clear {
        color: #059669;
    }
    .callout-box {
        border-left: 4px solid #3B82F6;
        background: #EFF6FF;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor(model_path: str):
    """Load model once and cache in memory."""
    from src.predict import WaterloggingPredictor
    return WaterloggingPredictor(model_checkpoint_path=model_path)


def main():
    st.markdown('<div class="main-header">🌊 AI Road Waterlogging Detection & Severity Estimation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning-Powered Road Surface Inundation Monitoring & Affected Area Assessment</div>', unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ System Configuration")
        default_model = "models/best_waterlogging_model.pt"
        model_path = st.text_input("Model Checkpoint Path", value=default_model)
        detection_threshold = st.slider("Detection Decision Threshold", min_value=0.30, max_value=0.90, value=0.50, step=0.05)

        st.markdown("---")
        st.subheader("📋 Methodology & Thresholds")
        st.markdown("""
        **Severity Rules (Project-Defined):**
        - **None / Dry:** 0.0% coverage
        - **Low:** 0.1% – 15.0% coverage
        - **Moderate:** 15.1% – 40.0% coverage
        - **Severe:** > 40.0% coverage
        
        *Note: Severity is computed strictly from measurable water surface area.*
        """)

        st.markdown("---")
        st.subheader("📊 Dataset & Model Info")
        report_json_path = Path("reports/dataset_summary.json")
        if report_json_path.exists():
            import json
            try:
                with open(report_json_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                st.write(f"**Discovered Images:** {summary.get('total_images', 0):,}")
                st.write(f"**Classes:** {', '.join(summary.get('classes', {}).get('detected_classes', []))}")
            except Exception:
                pass
        else:
            st.info("Dataset analysis report will appear here once inspection runs.")

    # Check model file existence
    if not Path(model_path).exists():
        st.warning(f"⚠️ Model checkpoint `{model_path}` is not found yet. Please run the training pipeline first: `python -m src.train`")
        st.info("💡 You can inspect the dataset and review project documentation while the model trains.")
        return

    # Load predictor
    try:
        predictor = load_predictor(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return

    # Image upload section
    st.subheader("📤 Upload Road Image")
    uploaded_file = st.file_uploader(
        "Choose an image (JPEG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a street-level photograph to assess road inundation and severity."
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read image file: {e}")
            return

        # Run Prediction
        with st.spinner("Analyzing image features and calculating flood region..."):
            try:
                result = predictor.predict(image, detection_threshold=detection_threshold)
            except Exception as e:
                st.error(f"Inference error: {e}")
                return

        # Display Metrics in 4 columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status = result["detection_status"]
            badge_class = "badge-detected" if status == "DETECTED" else "badge-clear"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Waterlogging Status</div>
                <div class="metric-value {badge_class}">{status}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Model Confidence</div>
                <div class="metric-value">{result['confidence_pct']}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            area_val = f"{result['affected_area_pct']}%" if result.get('affected_area_pct') is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Affected Area</div>
                <div class="metric-value">{area_val}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            sev = result["severity"]
            sev_color = sev.get("color", "#64748B")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Estimated Severity</div>
                <div class="metric-value" style="color: {sev_color};">{sev['severity_level']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Comparison Section
        c_left, c_right = st.columns(2)
        with c_left:
            st.subheader("Original Input Image")
            st.image(image, use_container_width=True)

        with c_right:
            st.subheader("Detected / Localized Flood Region")
            st.image(result["overlay_image"], use_container_width=True)

        # Technical Details Expander
        with st.expander("🔍 Detailed Technical Breakdown & Explainability", expanded=True):
            st.write(f"**Severity Rationale:** {result['severity']['explanation']}")
            st.write(f"**Measurement Principle:** {result['area_details']['note']}")
            st.write(f"**Class Probabilities:**")
            st.json(result["class_probabilities"])


if __name__ == "__main__":
    main()
