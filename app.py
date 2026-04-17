import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import requests

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="CIFAR-10 Pro Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS FOR ADVANCED UI
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .css-1r6slb0 { /* Sidebar styling */
        background-color: #161b22;
    }
    div[data-testid="stExpander"] {
        border: none !important;
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_status_code=True)

# 3. LOTTIE ANIMATION HELPER
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ghp9v83m.json")

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
    # Ensure the file exists or wrap in try-except for first-run safety
    try:
        model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device("cpu")))
    except FileNotFoundError:
        st.error("Model file 'cnn_model.pth' not found!")
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# --- END CORE LOGIC ---

# 4. SIDEBAR DESIGN
with st.sidebar:
    st_lottie(lottie_ai, height=200, key="ai_side")
    st.title("Model Settings")
    st.info("This model is trained on the CIFAR-10 dataset to recognize 10 different object categories.")
    st.divider()
    st.markdown("### Technologies Used")
    st.write("• PyTorch\n• Streamlit\n• TorchVision")

# 5. MAIN PAGE UI
header_col1, header_col2 = st.columns([2, 1])
with header_col1:
    st.title("🧠 CIFAR-10 Vision AI")
    st.markdown("#### *Real-time Image Classification using Convolutional Neural Networks*")
with header_col2:
    st_lottie(lottie_ai, height=150, key="ai_main")

tab1, tab2 = st.tabs(["🚀 Classifier", "📚 Information"])

with tab1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Drop an image here", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("### Preview")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### Inference Result")
            with st.spinner("Processing through Neural Network..."):
                img = transform(image)
                img = img.unsqueeze(0)

                with torch.no_grad():
                    outputs = model(img)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                
                # Dynamic UI Card for result
                st.markdown(f"""
                    <div class="prediction-card">
                        <h1 style="margin:0; font-size: 50px;">{classes[predicted.item()].upper()}</h1>
                        <p style="font-size: 20px; opacity: 0.8;">Confidence: {confidence.item()*100:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.progress(float(confidence.item()))
                
                # Show full probability breakdown
                with st.expander("Show detailed probabilities"):
                    prob_dict = {classes[i]: float(probabilities[0][i]) for i in range(10)}
                    st.bar_chart(prob_dict)

with tab2:
    st.markdown("""
    ### About the Dataset
    The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. 
    There are 50,000 training images and 10,000 test images.
    
    ### Architecture Details
    - **Convolutional Layer 1**: 32 filters, ReLU, MaxPool
    - **Convolutional Layer 2**: 64 filters, ReLU, MaxPool
    - **Convolutional Layer 3**: 128 filters, ReLU, MaxPool
    - **Fully Connected**: 256 Hidden Units -> 10 Output Classes
    """)

# 6. FOOTER
st.markdown("---")
st.markdown("<center>Made with ❤️ using PyTorch and Streamlit | v2.0</center>", unsafe_allow_html=True)
