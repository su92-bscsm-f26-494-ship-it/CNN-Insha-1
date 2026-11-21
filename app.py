import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Skin Image Colour Classifier",
    page_icon="🔬",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "model.pkl"


@st.cache_resource
def load_model():
    """Load the trained scikit-learn model once per Streamlit session."""
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def extract_features(image: Image.Image) -> pd.DataFrame:
    """Extract the same six features used when the model was trained."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image_array = np.asarray(image, dtype=np.float32)

    width, height = image.size
    if height == 0:
        raise ValueError("The uploaded image has invalid dimensions.")

    mean_r = float(image_array[:, :, 0].mean())
    mean_g = float(image_array[:, :, 1].mean())
    mean_b = float(image_array[:, :, 2].mean())

    return pd.DataFrame(
        [[width, height, width / height, mean_r, mean_g, mean_b]],
        columns=[
            "width",
            "height",
            "aspect_ratio",
            "mean_r",
            "mean_g",
            "mean_b",
        ],
    )


st.title("🔬 Skin Image Colour Classifier")
st.write(
    "Upload an image to classify its overall colour features using the "
    "trained Random Forest model."
)

st.warning(
    "**Research/demo only:** This model predicts colour-based proxy labels "
    "(`Darker/Pigmented` or `Lighter`). It was trained using labels derived "
    "from average image intensity, not dermatologist-confirmed diagnoses. "
    "It cannot detect skin cancer, diagnose a skin condition, or assess "
    "whether a lesion is benign or malignant. Do not use it for medical decisions."
)

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Classify image", type="primary"):
            model = load_model()
            features = extract_features(image)
            prediction = model.predict(features)[0]

            st.subheader("Prediction")
            st.success(f"Predicted colour category: **{prediction}**")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                probability_table = pd.DataFrame(
                    {
                        "Category": model.classes_,
                        "Model score": probabilities,
                    }
                ).set_index("Category")
                st.caption(
                    "Model scores are classifier probabilities for these "
                    "proxy labels; they are not medical risk probabilities."
                )
                st.bar_chart(probability_table)

            with st.expander("View extracted image features"):
                st.dataframe(features, use_container_width=True)

    except Exception as error:
        st.error(f"Could not process this image: {error}")

st.divider()
st.caption("Model: RandomForestClassifier | Input: six image colour/size features")
