# Guiding DIP Early Stopping with DDPM-inspired Supervision

## Overview

Traditional Deep Image Prior (DIP) frameworks involve training a randomly initialized neural network (e.g., U-net, ResNet, CNN) directly on a single target image. This approach leverages the natural image statistics learned by deep neural networks during the training process, internalizing them as model parameters to achieve high-quality image reconstruction. However, DIP faces a critical challenge: how to determine the optimal stopping point for training. Stopping too early results in severe underfitting, failing to capture sufficient details, while overtraining can cause the network to start fitting the noise, leading to degraded image quality. Conventional methods address this by monitoring the trend of training loss or setting a fixed number of training iterations, but these heuristic approaches often have limited effectiveness.

Our project proposes a novel method inspired by the Denoising Diffusion Probabilistic Model (DDPM). DDPM learns the conditional distribution of progressively removing noise, achieving high-quality generation from pure noise to complex data distributions, such as natural images. Introducing the staged denoising training process of DDPM into the DIP framework theoretically helps deep networks capture hierarchical image structure information and provides a more reasonable criterion for determining the stopping point of training.

## Installation

To use this framework, you need to have Python installed on your system along with the following packages:

```bash
pip install -r requirements.txt
```

## Usage

To use the DIP-DDPM framework, simply run the `main.ipynb` notebook. This notebook contains all the necessary steps to train the model and reconstruct the image. Follow the instructions within the notebook to execute the code cells sequentially.

1. Open `main.ipynb` in your preferred Jupyter Notebook environment.
2. Execute each cell in order from top to bottom.
3. The notebook will guide you through the process of loading the image, training the model, and viewing the reconstructed output.

For detailed instructions and explanations, please refer to the comments and markdown cells within `main.ipynb`.

