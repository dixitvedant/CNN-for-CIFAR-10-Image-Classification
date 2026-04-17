import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="VisionAI | CIFAR-10",
    page_icon="🧠",
    layout="wide"
)

# 2. ADVANCED CSS (Glassmorphism & Custom Styling)
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #121212 0%, #1a1a2e 100%);
    }
    
    /* Custom Card Styling */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Prediction Text */
    .pred-text {
        color: #00d2ff;
        font-size: 3rem !important;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 10px 0;
    }

    /* Status badges */
    .stAlert {
        background: rgba(0, 210, 255, 0.1) !important;
        color: white !important;
        border: 1px solid #00d2ff !important;
    }

    /* Buttons & Uploaders */
    .stFileUploader {
        border: 2px dashed rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 20px;
    }
    
    hr {
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- CORE LOGIC (UNTOUCHED) ---
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

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

@st.cache_resource
def load_model():
    model = CNN()
    try:
        model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device("cpu")))
    except:
        pass # Handle missing file gracefully for the UI demo
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# --- END CORE LOGIC ---

# 3. UI LAYOUT
st.write("") # Spacing
col_h1, col_h2 = st.columns([2, 1])

with col_h1:
    st.markdown("<h1 style='font-size: 3.5rem; margin-bottom: 0;'>VisionAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; color: #888;'>CIFAR-10 Deep Learning Classification Engine</p>", unsafe_allow_html=True)

with col_h2:
    st.write("")
    with st.expander("ℹ️ Model Specs"):
        st.write("**Architecture:** 3-Layer CNN")
        st.write("**Classes:** 10 Categories")
        st.write("**Framework:** PyTorch")

st.markdown("---")

# 4. MAIN INTERACTION
c1, c2 = st.columns([1, 1], gap="large")

with c1:
    st.subheader("📤 Input Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Current Selection", use_container_width=True)
    else:
        # Placeholder info
        st.info("Please upload an image (JPG/PNG) to begin inference.")

with c2:
    st.subheader("🔍 Analysis")
    if uploaded_file:
        # Run prediction
        img = transform(image)
        img = img.unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            conf_score = confidence.item()

        # Modern Result Card
        st.markdown(f"""
            <div class="result-card">
                <p style="margin:0; font-size: 0.9rem; color: #aaa; text-transform: uppercase;">Top Prediction</p>
                <h1 class="pred-text">{classes[predicted.item()]}</h1>
                <p style="font-size: 1.1rem; color: white;">Confidence: {conf_score*100:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Confidence Bar
        st.write("Match Confidence")
        st.progress(conf_score)
        
        # Detail expansion
        with st.expander("View Classification Breakdown"):
            # Using simple progress bars for each class
            for i in range(10):
                prob = float(probabilities[0][i])
                st.write(f"{classes[i].capitalize()}")
                st.progress(prob)
    else:
        st.markdown("""
            <div style="background: rgba(255,255,255,0.02); border-radius: 15px; padding: 50px; text-align: center; border: 1px dashed rgba(255,255,255,0.1);">
                <p style="color: #666;">Waiting for image input...</p>
            </div>
        """, unsafe_allow_html=True)

# 5. FOOTER
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("---")
f1, f2 = st.columns(2)
with f1:
    st.caption("Developed with PyTorch & Streamlit Native CSS")
with f2:
    st.markdown("<p style='text-align: right; color: #555;'>Version 1.0.0</p>", unsafe_allow_html=True)
