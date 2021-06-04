import streamlit as st
from models import getmodel
import torch
import torchvision
from PIL import Image
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A

#Getting our model and loading the weights

model = getmodel()

def main():
    st.write("""
    # This is anime edges character to colorized character using pix2pix
    """)
    st.write("---")

    image = st.file_uploader(label="Please upload low resolution image",
                             accept_multiple_files=False,
                             help="upload image")
    print(type(image))
    if image is not None:
        image = Image.open(image)
        image = np.array(image)

        st.image(actual, caption='Actual', use_column_width=False, width=300)
        st.write("---")
        st.image(mask, caption='input to our model', use_column_width=False, width=300)
        st.write("Please wait to be colorized and get final touch")

        transform_only_mask = A.Compose(
            [
                A.Resize(256, 256),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
                ToTensorV2(),
            ]
        )

        image = transform_only_mask(image=image)
        image["image"] = image["image"].unsqueeze(0)

        out = model(image["image"])

        out = out.squeeze(0)
        out = torchvision.transforms.ToPILImage()(out)


        st.write("""
            # Colorized image
            """)
        st.image(out, caption='Actucal Look Like Picture', use_column_width=False, width=300)
        st.write("you can download it and also share it with others")

if __name__ == '__main__':
    main()