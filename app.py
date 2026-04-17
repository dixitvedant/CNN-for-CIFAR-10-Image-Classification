import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms


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


# UI
st.title("CIFAR-10 Image Classifier")

st.write(
"Upload image of airplane, car, bird, cat, deer, dog, frog, horse, ship, truck"
)


uploaded_file = st.file_uploader(
"Upload Image",
type=["jpg","png","jpeg"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

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


    st.success(
        f"Prediction: {classes[predicted.item()]}"
    )


    st.write(
        f"Confidence: {confidence.item()*100:.2f}%"
    )
