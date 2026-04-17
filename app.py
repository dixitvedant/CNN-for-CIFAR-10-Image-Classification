import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import CNN


# page settings
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🧠",
    layout="centered"
)


# class labels
classes = [
'airplane',
'automobile',
'bird',
'cat',
'deer',
'dog',
'frog',
'horse',
'ship',
'truck'
]


# load trained model
@st.cache_resource
def load_model():

    model = CNN()

    model.load_state_dict(
        torch.load(
            "cnn_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model()


# image transform
transform = transforms.Compose([

    transforms.Resize((32,32)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )

])


# UI HEADER
st.markdown(
"""
# 🧠 CIFAR-10 Image Classifier

Deep Learning Web App using **PyTorch + Streamlit**
"""
)


# dataset info
with st.expander("📚 About CIFAR-10 Dataset"):

    st.write(
"""
CIFAR-10 is a foundational computer vision dataset containing **60,000 32x32 pixel color images** in **10 distinct classes**.

Dataset split:

• 50,000 training images  
• 10,000 testing images  

Classes included:

✈️ airplane  
🚗 automobile  
🐦 bird  
🐱 cat  
🦌 deer  
🐶 dog  
🐸 frog  
🐴 horse  
🚢 ship  
🚚 truck  

It is widely used in machine learning for training and benchmarking image classification models.
"""
)


# upload section
st.markdown("## 📤 Upload Image")

uploaded_file = st.file_uploader(
    "Upload an image (jpg, png)",
    type=["jpg","png","jpeg"]
)


# prediction
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_column_width=True
        )


    img = transform(image)

    img = img.unsqueeze(0)


    with torch.no_grad():

        outputs = model(img)

        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities,1)


    with col2:

        st.markdown("### 🔍 Prediction Result")

        st.success(
            classes[predicted.item()]
        )

        st.write(
            "Confidence:",
            f"{confidence.item()*100:.2f}%"
        )


        st.progress(
            float(confidence.item())
        )


# footer
st.markdown(
"""
---
Made using with PyTorch and Streamlit
"""
)
