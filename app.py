import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="VisionAI | CIFAR-10 Classifier",
    layout="wide"
)

# 2. ADVANCED CSS (Glassmorphism & Custom Styling)
st.markdown("""
    <style>
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: white;
    }
    
    /* Custom Card Styling for Prediction */
    .result-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5);
        margin: 10px 0;
    }

    /* Prediction Result Text */
    .pred-text {
        color: #00d2ff;
        font-size: 3.5rem !important;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin: 5px 0;
        text-shadow: 0 0 20px rgba(0, 210, 255, 0.5);
    }

    /* Sub-headers and Text */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }

    /* File Uploader styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 10px;
    }

    /* Metric Styling Override */
    [data-testid="stMetricValue"] {
        color: #00d2ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CORE LOGIC (STRICTLY PRESERVED) ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

classes = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

@st.cache_resource
def load_model():
    model = CNN()
    try:
        model.load_state_dict(
            torch.load("cnn_model.pth", map_location=torch.device("cpu"))
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# --- END CORE LOGIC ---

# 3. HEADER
st.write("") 
st.markdown("<h1 style='text-align: center; font-size: 4rem;'> VisionAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #aaa;'>Deep Learning Image Classification Interface</p>", unsafe_allow_html=True)
st.write("")

# 4. MAIN TABS
tab1, tab2 = st.tabs([" Classifier Engine", " Dataset Information"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Drop image here", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Input Frame", use_container_width=True)
        else:
            st.info("Awaiting image upload to trigger inference...")
            st.markdown("""
                <div style="padding: 50px; border: 2px dashed rgba(255,255,255,0.1); border-radius: 20px; text-align: center;">
                    <p style="color: #666;">Images are automatically resized to 32x32 for the CNN model</p>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.subheader("🔍 Prediction Results")
        if uploaded_file:
            # Inference Processing
            img = transform(image)
            img = img.unsqueeze(0)

            with torch.no_grad():
                outputs = model(img)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                conf_val = confidence.item()

            # Glassmorphism Result Card
            st.markdown(f"""
                <div class="result-card">
                    <p style="margin:0; font-size: 1rem; color: #888; text-transform: uppercase; font-weight: bold;">Top Inference</p>
                    <h1 class="pred-text">{classes[predicted.item()]}</h1>
                    <p style="font-size: 1.2rem; color: white;">Confidence Level: {conf_val*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("Neural Network Probability Map")
            st.progress(conf_val)
            
            with st.expander("Full Probability Breakdown"):
                for i in range(len(classes)):
                    prob = float(probabilities[0][i])
                    st.write(f"**{classes[i].capitalize()}**")
                    st.progress(prob)
        else:
            st.warning("Upload an image on the left to see model output.")

with tab2:
    st.markdown("## Dataset: CIFAR-10")
    
    # Summary Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Images", "60,000")
    m2.metric("Training Set", "50,000")
    m3.metric("Test Set", "10,000")

    st.markdown("---")

    col_desc, col_list = st.columns([1.5, 1], gap="large")

    with col_desc:
        st.subheader("Technical Profile")
        st.markdown("""
        The **CIFAR-10** dataset is a foundational computer vision collection containing 
        **60,000 32x32 pixel color images**. It is widely used in machine learning for 
        benchmarking classification algorithms.
        
        Because the resolution is very low ($32 \times 32$), the model must learn to recognize 
        the core essence of shapes and patterns rather than relying on high-frequency details.
        """)
        st.success("✅ This model uses a 3-layer Convolutional Neural Network (CNN) optimized for this data.")

    with col_list:
        st.subheader("Class Labels")
        l1, l2 = st.columns(2)
        with l1:
            st.write("✈️ airplane")
            st.write("🚗 automobile")
            st.write("🐦 bird")
            st.write("🐱 cat")
            st.write("🦌 deer")
        with l2:
            st.write("🐶 dog")
            st.write("🐸 frog")
            st.write("🐴 horse")
            st.write("🚢 ship")
            st.write("🚚 truck")

# 5. FOOTER
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p style='text-align: center; color: #555;'>Made with using PyTorch + Streamlit Advanced CSS</p>", unsafe_allow_html=True)
