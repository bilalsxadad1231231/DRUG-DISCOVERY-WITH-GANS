# Drug Discovery with WGAN streamlit demo

![Project Overview](https://github.com/bilalsxadad1231231/DRUG-DISCOVERY-WITH-GANS/blob/main/Pictures/Molecule_sample.png)

This project focuses on a generative model for drug discovery, specifically a Wasserstein Generative Adversarial Network (WGAN) that generates molecular structures as graphs. The generator learns to create molecule graphs by producing atom and bond features, while the discriminator assesses the validity of these structures. The project leverages TensorFlow and Keras with graph convolutional layers to process molecular graphs.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Graph Representation of Molecules](#graph-representation-of-molecules)
  - [Graph Generator](#graph-generator)
  - [Graph Discriminator](#graph-discriminator)
  - [WGAN Training](#wgan-training)

## Overview

This project generates molecules using WGAN with a graph-based approach. Molecules are represented by atoms (nodes) and bonds (edges) and are encoded as adjacency and feature tensors, with each molecule mapped to a unique structure.

## Project Structure

- `atom_mapping`: Maps atom symbols to unique indices.
- `bond_mapping`: Maps bond types to unique indices.
- `GraphGenerator`: Defines the model for generating adjacency and feature tensors.
- `GraphDiscriminator`: Defines the model to classify generated molecules as real or fake.
- `GraphWGAN`: Implements the WGAN training loop.

## Installation

Ensure you have the necessary libraries installed:

      ```bash
      pip install tensorflow keras rdkit numpy
## Usage

### Graph Representation of Molecules

Molecules are represented as adjacency and feature matrices:
- **Adjacency Tensor**: Encodes bonds between atoms, indicating bond type or lack of a bond.
- **Feature Tensor**: Encodes atom types.

### Graph Generator

The `GraphGenerator` model takes a latent vector as input and generates continuous adjacency and feature tensors. These tensors represent molecule graphs, including atom types and bonds between atoms.

### Graph Discriminator

The `GraphDiscriminator` uses relational graph convolutional layers to evaluate molecule structures based on their adjacency and feature tensors.

### WGAN Training

`GraphWGAN` coordinates the training of the generator and discriminator with a Wasserstein loss and a gradient penalty.
## Testing

1. **Clone the Repository**
   - Run:
     ```bash
     git clone https://github.com/bilalsxadad1231231/DRUG-DISCOVERY-WITH-GANS/tree/main
     cd DRUG-DISCOVERY-WITH-GANS
     ```

2. **Set Up Model Weights**
   - Ensure pre-trained model weights are available in the `model_weights` folder.

3. **Run Streamlit Demo**
   - Launch the app with:
     ```bash
     streamlit run app.py
     ```
## Training

1. **Training Notebook**
   - The training notebook, `training_WGAN.ipynb`, is located in the `training` folder.
   - This notebook provides step-by-step instructions for training the model.

2. **Run the Notebook**
   - Navigate to the `training` folder and open `training_WGAN.ipynb` in Jupyter or Colab.
   - Follow the instructions in the notebook to preprocess data, train the model, and evaluate results.

3. **Custom Training**
   - Adjust parameters in the notebook as needed to fine-tune the model for your specific requirements.
