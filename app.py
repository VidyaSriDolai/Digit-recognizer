"""Step 2: Streamlit app - Handwritten Digit Recognizer (CNN on MNIST)."""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.banner {
    background: linear-gradient(90deg, #1e3a8a, #2563eb);
    padding: 1.3rem 1.6rem; border-radius: 12px; margin-bottom: 1rem;
}
.banner h1 { color: white; margin: 0; font-size: 1.9rem; }
.banner p { color: #dbeafe; margin: 0.3rem 0 0 0; }
</style>
<div class="banner">
  <h1>✍️ Handwritten Digit Recognizer</h1>
  <p>Deep Learning · Convolutional Neural Network · MNIST dataset</p>
</div>
""", unsafe_allow_html=True)


# ---------- Helpers ----------
@st.cache_resource
def load_model():
    return keras.models.load_model("digit_model.h5", compile=False)

def load_metrics():
    if os.path.exists("metrics.json"):
        with open("metrics.json") as f:
            return json.load(f)
    return None


def preprocess(gray):
    """gray: 2D uint8 array (white digit on black). Returns centered 28x28 in 0-1."""
    ys, xs = np.where(gray > 20)
    if len(xs) == 0:
        return None
    digit = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    digit = Image.fromarray(digit).resize(
        (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    canvas28 = Image.new("L", (28, 28), 0)
    canvas28.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    return np.array(canvas28) / 255.0


def show_prediction(img28, model):
    probs = model.predict(img28.reshape(1, 28, 28, 1), verbose=0)[0]
    top3 = np.argsort(probs)[::-1][:3]
    st.metric("Predicted digit", int(top3[0]), f"{probs[top3[0]] * 100:.1f}% confident")

    st.markdown("**Top 3 predictions**")
    for rank, d in enumerate(top3, 1):
        st.write(f"{rank}. Digit **{d}** — {probs[d] * 100:.1f}%")
        st.progress(float(probs[d]))

    st.markdown("**Probability for every digit**")
    st.bar_chart(pd.DataFrame({"probability": probs}, index=range(10)))


model = load_model()
metrics = load_metrics()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🧠 Model Info")
    st.write("**Type:** Convolutional Neural Network")
    st.write("**Dataset:** MNIST (70,000 digit images)")
    st.write("**Input:** 28 × 28 grayscale image")
    st.write("**Output:** 10 classes (digits 0–9)")
    st.write("**Optimizer:** Adam")
    st.write("**Loss:** Sparse categorical crossentropy")
    if metrics:
        st.divider()
        st.metric("Test accuracy", f"{metrics['test_accuracy'] * 100:.2f}%")
        st.metric("Parameters", f"{metrics['parameters']:,}")
    st.divider()
    st.markdown("**Architecture**")
    st.code("Conv2D(32) → MaxPool\nConv2D(64) → MaxPool\nFlatten → Dropout(0.3)\nDense(128) → Dense(10)", language=None)

# ---------- Tabs ----------
tab_draw, tab_upload, tab_perf = st.tabs(["✏️ Draw", "📁 Upload Image", "📊 Model Performance"])

with tab_draw:
    st.write("Draw a single digit (0–9) in the box. The prediction updates live.")
    c1, c2 = st.columns(2)
    with c1:
        canvas = st_canvas(
            fill_color="black", stroke_width=18, stroke_color="white",
            background_color="black", height=280, width=280,
            drawing_mode="freedraw", key="canvas",
        )
    with c2:
        if canvas.image_data is not None:
            gray = np.array(Image.fromarray(canvas.image_data.astype("uint8")).convert("L"))
            img = preprocess(gray)
            if img is not None:
                show_prediction(img, model)
            else:
                st.info("Draw a digit to get a prediction.")

with tab_upload:
    st.write("Upload a photo or image of a handwritten digit (dark ink on light paper works best).")
    file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if file is not None:
        pil = Image.open(file).convert("L")
        gray = np.array(ImageOps.autocontrast(pil))
        if gray.mean() > 127:          # light background -> invert to white-on-black
            gray = 255 - gray
        c1, c2 = st.columns(2)
        with c1:
            st.image(pil, caption="Uploaded image", width=280)
        with c2:
            img = preprocess(gray)
            if img is not None:
                st.image(img, caption="What the model sees (28×28)", width=140, clamp=True)
                show_prediction(img, model)
            else:
                st.warning("Couldn't find a digit in this image. Try a clearer one.")

with tab_perf:
    st.subheader("Training results")
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test accuracy", f"{metrics['test_accuracy'] * 100:.2f}%")
        m2.metric("Test loss", f"{metrics['test_loss']:.4f}")
        m3.metric("Training images", f"{metrics['train_images']:,}")
        m4.metric("Test images", f"{metrics['test_images']:,}")
    else:
        st.info("Run train_model.py to generate metrics.json.")

    if os.path.exists("training_curves.png"):
        st.markdown("**Accuracy and loss curves**")
        st.image("training_curves.png", use_column_width=True)
    if os.path.exists("confusion_matrix.png"):
        st.markdown("**Confusion matrix**")
        st.caption("Rows = actual digit, columns = predicted digit. A strong diagonal means mostly correct predictions.")
        st.image("confusion_matrix.png", width=550)

st.caption("Built with TensorFlow/Keras and Streamlit")
