"""Streamlit upload-to-segmentation demo for the FloodNet U-Net."""

from pathlib import Path

import streamlit as st
from PIL import Image

from src.config import MODELS_DIR


st.set_page_config(
    page_title="AI Road Waterlogging Detection",
    page_icon="🌊",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the FloodNet segmentation model…")
def load_predictor(model_path: str):
    """Load model weights once per app process."""
    from src.predict import FloodNetPredictor

    return FloodNetPredictor(model_checkpoint_path=model_path)


def main():
    st.title("🌊 AI Road Waterlogging Detection")
    st.caption("FloodNet semantic segmentation with project-defined road coverage tiers")

    with st.sidebar:
        model_path = st.text_input(
            "U-Net checkpoint",
            value=str(MODELS_DIR / "floodnet_unet_best.pth"),
        )
        st.markdown("### Project-defined severity thresholds")
        st.markdown("- NONE: ≤ 0.5%\n- LOW: > 0.5% to 15%\n- MODERATE: > 15% to 40%\n- SEVERE: > 40%")
        st.caption("These tiers estimate 2D predicted road coverage. They do not measure water depth or real-world safety.")

    checkpoint_path = Path(model_path).expanduser()
    if not checkpoint_path.is_file():
        st.warning(f"Checkpoint not found: `{checkpoint_path}`. Train the FloodNet U-Net or select a valid checkpoint.")
        return

    try:
        predictor = load_predictor(str(checkpoint_path.resolve()))
    except Exception as exc:
        st.error(f"Could not load the FloodNet checkpoint: {exc}")
        return

    metrics = predictor.checkpoint_metadata.get("val_metrics") or {}
    if metrics.get("road_flooded_iou", 0) == 0:
        st.warning(
            "This checkpoint records Road-flooded validation IoU of 0.0. "
            "Segmentation is available for demonstration, but flooded-road area and severity are not validated."
        )

    upload = st.file_uploader("Upload a road or aerial image", type=["jpg", "jpeg", "png", "webp"])
    if upload is None:
        st.info("Choose a JPEG, PNG, or WEBP image to run semantic segmentation.")
        return

    if upload.size > 25 * 1024 * 1024:
        st.error("Image is too large. Maximum upload size is 25 MB.")
        return

    try:
        image = Image.open(upload)
        width, height = image.size
        if width < 64 or height < 64:
            st.error("Image dimensions must be at least 64 × 64 pixels.")
            return
        if width > 12000 or height > 12000 or width * height > 40_000_000:
            st.error("Image dimensions are too large. Limit: 12,000 pixels per side and 40 megapixels total.")
            return
        image.verify()
        upload.seek(0)
        image = Image.open(upload).convert("RGB")
    except Exception as exc:
        st.error(f"Please upload a valid image file. Details: {exc}")
        return

    with st.spinner("Running per-pixel FloodNet segmentation…"):
        try:
            result = predictor.predict(image)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            return

    severity = result["severity"]
    area = severity["road_waterlogged_pct"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted flooded road area", f"{area:.2f}%" if area is not None else "Unavailable")
    col2.metric("Project-defined tier", severity["severity_level"])
    col3.metric("Predicted road pixels", f"{severity['total_road_pixels']:,}")

    left, middle, right = st.columns(3)
    left.image(result["original_image"], caption="Original image", use_container_width=True)
    middle.image(result["mask_image"], caption="Predicted FloodNet classes", use_container_width=True)
    right.image(result["overlay_image"], caption="Road-flooded pixels highlighted", use_container_width=True)

    if area is None:
        st.info("No road pixels were predicted, so the road-area percentage and severity tier are unavailable.")
    st.write(severity["explanation"])
    with st.expander("Class coverage and checkpoint details"):
        st.json(result["class_probabilities"])
        st.write(f"Checkpoint epoch: {result['checkpoint_metadata'].get('epoch')}")
        st.write(f"Validation metrics: {metrics}")
    st.caption(
        "FloodNet training imagery is aerial. Predictions may not generalize to street-level images. "
        "Water outside predicted road pixels is excluded from the road-area calculation."
    )


if __name__ == "__main__":
    main()
