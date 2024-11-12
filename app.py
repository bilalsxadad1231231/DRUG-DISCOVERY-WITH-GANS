import streamlit as st
import pandas as pd
import os
from PIL import Image
import numpy as np
from Model import generator
from helper_function import sample
from packages import MolToSmiles, MolToImage
from global_variables import BOND_DIM, LATENT_DIM, NUM_ATOMS, ATOM_DIM

# Function to clear existing images from the directory
def clear_images(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Directory to store images
IMAGE_DIR = "images"
clear_images(IMAGE_DIR)
# Create the images directory if it doesn't exist
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Streamlit application
st.title("Molecule Generator")

# Use the sidebar for input and button
with st.sidebar:
    # Text input for batch size
    batch_size = st.number_input("Enter batch size:", min_value=1, value=5, step=1)
    # Button to generate images
    if st.button("Generate"):
        # Clear existing images
        clear_images(IMAGE_DIR)
        
        # Generate new molecules
        with st.spinner("Generating molecules..."):
            try:
                molecules = sample(generator, batch_size+20)
                molecules = [m for m in molecules if m is not None]
                
                # Check if there are molecules generated
                if molecules:
                    # Save images
                    for i in range(batch_size):
                        image = MolToImage(molecules[i])
                        array = np.array(image)
                        pil_image = Image.fromarray(array)
                        
                        # Save the image
                        path = os.path.join(IMAGE_DIR, f"image_{i}.png")
                        pil_image.save(path)
            except Exception as e:
                st.error(f"Error generating molecules: {e}")

        # Display images in a grid
        # st.header("Generated Images")
        # for i in range(0, batch_size, 4):
        #     cols = st.columns(4)
        #     for j in range(4):
        #         if i + j < batch_size:
        #             image_path = os.path.join(IMAGE_DIR, f"image_{i + j}.png")
        #             cols[j].image(image_path, caption=f"Image {i + j + 1}")

# Create tabs
tab1, tab2 = st.tabs(["Generated Images", "Molecule Properties Table"])

with tab1:
    st.header("Generated Images")
    for i in range(0, batch_size, 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < batch_size:
                try:
                    image_path = os.path.join(IMAGE_DIR, f"image_{i + j}.png")
                    cols[j].image(image_path, caption=f"Image {i + j + 1}")
                except:
                    continue

with tab2:
    st.header("Molecule Properties Table")
    # Assuming you have a DataFrame `df` with molecule properties
    df = pd.DataFrame({
        "ID": range(1, batch_size+1),
        "logP": np.random.rand(batch_size),
        "Weight": np.random.rand(batch_size) * 100,
        "Hdonor": np.random.randint(0, 5, batch_size),
        "TPSA": np.random.rand(batch_size) * 50
    })
    st.table(df)
