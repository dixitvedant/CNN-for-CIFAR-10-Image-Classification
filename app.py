import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms


# page configuration
st.set_page_config(
    page_title="CIFAR-10 CNN Classifier",
    layout="centered"
)


# CNN architecture (same as training)
class CNN(nn.Module):

    def __init__(self):

        super(CNN,self).__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc_layers = nn.Sequential(

            nn.Linear(4*4*128,256),
            nn.ReLU(),

            nn.Linear(256,10)
        )


    def forward(self,x):

        x = self.conv_layers(x)

        x = x.view(x.size(0),-1)

        x = self.fc_layers(x)

        return x



# class labels
classes = [
"airplane",
"automobile",
"bird",
"cat",
"deer",
"dog",
"frog",
"horse",
"ship",
"truck"
]


# load model
@st.cache_resource
def load_model():

    model = CNN()

    model.load_state_dict(
        torch.load(
            "cnn_model.pth",
            map_location=torch.device("cpu")
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


# TITLE
st.markdown(
"""
# CIFAR-10 Image Classifier
Deep Learning Web App using **PyTorch + Streamlit**
"""
)


# DATASET DESCRIPTION
with st.expander("About CIFAR-10 Dataset"):

    st.write(
"""
CIFAR-10 is a foundational computer vision dataset containing **60,000 32x32 pixel color images in 10 distinct classes**.

It consists of:

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

CIFAR-10 is widely used for training and benchmarking image classification models in Deep Learning.
"""
)


st.markdown("---")


# upload section
st.subheader("Upload Image")

uploaded_file = st.file_uploader(
"Choose an image",
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

        probabilities = torch.nn.functional.softmax(outputs,dim=1)

        confidence , predicted = torch.max(probabilities,1)


    with col2:

        st.subheader("🔍 Prediction")

        st.success(
            classes[predicted.item()]
        )

        st.write(
            f"Confidence: {confidence.item()*100:.2f}%"
        )

        st.progress(float(confidence.item()))


# footer
st.markdown(
"""
---
Made with using PyTorch and Streamlit
"""
)
