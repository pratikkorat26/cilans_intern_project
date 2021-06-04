import streamlit as st
from srgan_inf import *
import os
import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

SAVED_MODEL_PATH = "saved_model/esrgan-tf2_1/"

model = tf.saved_model.load(SAVED_MODEL_PATH)

st.write("""
# This is super resolution image app
""")
st.write("---")

image = st.file_uploader(label = "Please upload low resolution image",
                 accept_multiple_files = False,
                 help = "upload image")

if image is not None:
    image = Image.open(image)
    st.image(image, caption='Uploaded Image.', use_column_width=False , width = 300)
    st.write("")
    st.write("Please wait to be super resoluted image")


    st.write("---")
    st.write("Proccessing image.....")
    hr_image = preprocess_image(image)
    st.write("done processing....")

    start = time.time()
    fake_image = model(hr_image)
    fake_image = tf.squeeze(fake_image)
    st.write("Time Taken: %f" % (time.time() - start))

    image = np.asarray(fake_image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())

    st.write("""
    # High resoluted image
    """)
    st.image(image, caption='High resolution Image.', use_column_width=False, width=300)
    st.write("you can download it and also share it with others")

