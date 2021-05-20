import streamlit as st
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torchvision import models, transforms
import numpy as np

from PIL import Image, ImageOps
from model import Plantmodel

classes={0: 'scab',
 1: 'healthy',
 2: 'frog_eye_leaf_spot',
 3: 'rust',
 4: 'complex',
 5: 'powdery_mildew',
 6: 'scab frog_eye_leaf_spot',
 7: 'scab frog_eye_leaf_spot complex',
 8: 'frog_eye_leaf_spot complex',
 9: 'rust frog_eye_leaf_spot',
 10: 'rust complex',}

st.title("Apple Plant Disease CLassifier")


st.header("Classify Disease From Image of Apple Plant Leaves")

st.text("Upload Image to classify it")

image=st.file_uploader("Upload an image to classify",type='jpg')
def predict(image):
    PATH='Plant_2021_epoch10.pth'
    model=Plantmodel()
    model.load_state_dict(torch.load(PATH))
    model.to('cuda')
    device='cuda'
    SIZE=224
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
            )])

    transforms_valid = A.Compose([
    A.Resize(height=SIZE, width=SIZE, p=1.0),
    ToTensorV2(p=1.0),
    ])
   # image=Image.open(image)
   # batch=torch.unsqueeze(transform(image),0)
    print("***********************")
    print(image.shape)
    batch=transforms_valid(image=image)['image']
    print(batch.shape)
    batch=torch.unsqueeze(batch,0)
    model.eval()
    batch=batch.to(device,dtype=torch.float)
    print(batch.shape)
    output=model(batch)
    #output=image
    return output


if image is not None:
    image_file=Image.open(image)
    st.image(image_file,caption='Uploaded Image',use_column_width=True)
    image=np.asarray(image_file)
    #print(image.shape)
   # image=np.expand_dims(image,axis=0)
    #print(image.shape)
    #image=image.transpose(2,0,1)

    output=predict(image)

    predicted=torch.argmax(output,dim=1)
    predicted=predicted.to('cpu').numpy()
    st.write((classes[predicted[0]]))
