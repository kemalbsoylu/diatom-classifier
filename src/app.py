"""
Streamlit Web UI for the Diatom Classifier Pipeline.
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from pathlib import Path
import io

# -- Setup Paths --
BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_PATH = BASE_DIR / "models" / "yolo_diatom_detector.pt"
RESNET_PATH = BASE_DIR / "models" / "v2_resnet18_weighted.pkl"

st.set_page_config(page_title="Diatom AI", page_icon="🔬", layout="wide")

@st.cache_resource
def load_models():
    from ultralytics import YOLO
    from fastai.vision.all import load_learner

    yolo = YOLO(YOLO_PATH)
    resnet = load_learner(RESNET_PATH)
    return yolo, resnet

st.title("🔬 Diatom Detection & Classification AI")
st.markdown("""
Upload a microscope image. Use **Full Slide Analysis** to automatically detect and classify multiple diatoms, 
or use **Single Diatom Crop** if you already have a cropped image of a single diatom.
""")

with st.spinner("Loading AI Models into memory..."):
    yolo_model, resnet_model = load_models()

# -- Sidebar Controls --
st.sidebar.header("Configuration")
app_mode = st.sidebar.radio("Select Analysis Mode:", ["Full Slide Analysis", "Single Diatom Crop"])

conf_threshold = 0.25
if app_mode == "Full Slide Analysis":
    conf_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25, 0.05)
    st.sidebar.markdown("*Lowering the threshold finds more diatoms but increases false positives. (Default: 0.25)*")

# -- Sidebar Footer (Portfolio & License) --
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("Developed by **Kemal Soylu**")
st.sidebar.markdown("[View Source Code on GitHub](https://github.com/kemalbsoylu/diatom-classifier)")
st.sidebar.markdown("""
<small>
<b>Licenses:</b> Code (MIT), Detector (AGPL-3.0).<br>
<b>Data:</b> Trained on dataset by Gündüz et al. (CC BY-NC-SA 4.0).
</small>
""", unsafe_allow_html=True)

# -- Main File Uploader --
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Silence PyCharm type warning
    assert not isinstance(uploaded_file, list)

    original_image = Image.open(uploaded_file).convert("RGB")

    st.markdown("---")
    st.subheader("Analysis Results")

    # -----------------------------------------------------
    # MODE 1: SINGLE CROP CLASSIFICATION
    # -----------------------------------------------------
    if app_mode == "Single Diatom Crop":
        with st.spinner("Classifying diatom..."):
            from fastai.vision.core import PILImage
            fastai_img = PILImage.create(original_image)
            pred_class, pred_idx, probs = resnet_model.predict(fastai_img)
            conf = probs[pred_idx].item() * 100

            st.success("Classification Complete!")
            st.metric(label="Predicted Genus", value=pred_class, delta=f"{conf:.2f}% Confidence")

            st.info("Note: This mode bypassed the automatic detector and evaluated the entire image as a single diatom. For best results, ensure your image is cropped tightly around the diatom with a maximum of 15% background margin.")

        st.markdown("---")
        st.subheader("Image Viewer")
        st.image(original_image, caption="Original Upload", use_container_width=False)

    # -----------------------------------------------------
    # MODE 2: FULL SLIDE YOLO + RESNET
    # -----------------------------------------------------
    else:
        display_image = original_image.copy()
        report_data = []
        diatom_count = 0

        with st.spinner("Scanning slide & Classifying..."):
            results = yolo_model(original_image, conf=conf_threshold, verbose=False)[0]

            draw = ImageDraw.Draw(display_image)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except IOError:
                font = ImageFont.load_default()

            from fastai.vision.core import PILImage

            for box in results.boxes.xyxy:
                diatom_count += 1
                x1, y1, x2, y2 = map(int, box.tolist())

                # Apply 15% margin for cropping
                box_w, box_h = x2 - x1, y2 - y1
                margin_x, margin_y = int(box_w * 0.15), int(box_h * 0.15)

                crop_x1 = max(0, x1 - margin_x)
                crop_y1 = max(0, y1 - margin_y)
                crop_x2 = min(original_image.width, x2 + margin_x)
                crop_y2 = min(original_image.height, y2 + margin_y)

                cropped_img = original_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                # Classify with ResNet
                fastai_img = PILImage.create(cropped_img)
                pred_class, pred_idx, probs = resnet_model.predict(fastai_img)
                conf = probs[pred_idx].item() * 100

                # Draw on display image
                label_text = f"{pred_class} ({conf:.1f}%)"
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1 - 25), label_text, fill="red", font=font)

                # Save to report
                report_data.append({
                    "ID": diatom_count,
                    "Genus": pred_class,
                    "Confidence": f"{conf:.2f}%"
                })

        # Render Full Slide Results
        if report_data:
            st.success(f"Successfully found {diatom_count} diatoms!")
            df = pd.DataFrame(report_data)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name=f"analysis_{uploaded_file.name}.csv",
                mime="text/csv",
            )
        else:
            st.warning("No diatoms detected. Try lowering the detection confidence threshold in the sidebar.")

        st.markdown("---")
        st.subheader("Image Viewer")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original Upload", use_container_width=True)
        with col2:
            st.image(display_image, caption="Analyzed Image", use_container_width=True)
