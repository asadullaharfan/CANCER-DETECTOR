
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🔬",
    layout="wide"
)

# 3D animated background and styling
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 20% 20%,
    #172554 0%, #080d20 45%, #020617 100%);
    color: white;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3, p, label {
    color: white !important;
}

.glass {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
}

.stButton>button {
    background: linear-gradient(90deg,#2563eb,#9333ea);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
}

#particles {
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    pointer-events: none;
}
</style>

<canvas id="particles"></canvas>

<script>
const canvas = document.getElementById("particles");
const ctx = canvas.getContext("2d");
let particles = [];

function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    particles = Array.from({length: 65}, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        z: Math.random() * 2 + 1,
        dx: (Math.random() - 0.5) * 0.5,
        dy: (Math.random() - 0.5) * 0.5
    }));
}

function animate() {
    ctx.clearRect(0,0,canvas.width,canvas.height);

    particles.forEach(p => {
        p.x += p.dx;
        p.y += p.dy;

        if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.dy *= -1;

        ctx.beginPath();
        ctx.arc(p.x,p.y,p.z,0,Math.PI*2);
        ctx.fillStyle = "rgba(96,165,250,0.7)";
        ctx.fill();
    });

    requestAnimationFrame(animate);
}

window.addEventListener("resize",resize);
resize();
animate();
</script>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_model.keras")


model = load_model()

class_names = [
    "akiec", "bcc", "bkl", "df",
    "mel", "nv", "vasc"
]

st.title("🔬 DermaScan AI")
st.markdown("### Experimental Skin Lesion Classification")

st.warning(
    "Educational project only. This AI cannot diagnose or rule out "
    "skin cancer. Consult a qualified healthcare professional "
    "for any skin concern."
)

st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("Choose Image Input")

input_method = st.radio(
    "Select your option:",
    ["📁 Upload Picture", "📷 Live Camera"],
    horizontal=True
)

uploaded_image = None

if input_method == "📁 Upload Picture":
    file = st.file_uploader(
        "Upload skin image",
        type=["jpg", "jpeg", "png"]
    )

    if file:
        uploaded_image = Image.open(file).convert("RGB")

else:
    camera_image = st.camera_input("Capture image using camera")

    if camera_image:
        uploaded_image = Image.open(camera_image).convert("RGB")

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_image:
    st.image(uploaded_image, caption="Selected Image", width=400)

    if st.button("🔍 Run Experimental Prediction"):
        image = uploaded_image.resize((224,224))
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        with st.spinner("Processing image..."):
            predictions = model.predict(image_array)
            index = np.argmax(predictions[0])

        st.subheader("Model Output")
        st.write("Predicted class:", class_names[index])
        st.write(
            "Model score:",
            f"{predictions[0][index] * 100:.2f}%"
        )

        st.error(
            "This is an experimental model output, not a diagnosis. "
            "The score is not a calibrated medical probability. "
            "Do not use it for healthcare decisions."
        )

st.markdown("---")
st.caption(
    "DermaScan AI | Educational Project | HAM10000 Dataset"
)
