import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import CNN

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

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])

st.title("CIFAR-10 Image Classifier")

st.write("Upload image and model will predict class")

uploaded_file = st.file_uploader(
    "Choose image",
    type=["jpg","png","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    img = transform(image)

    img = img.unsqueeze(0)

    with torch.no_grad():

        outputs = model(img)

        _, predicted = torch.max(outputs,1)

    st.success(
        "Prediction: " +
        classes[predicted.item()]
    )
