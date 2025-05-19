# Diffusion Model with Variational Autoencoder

This repository contains an implementation of a diffusion model using a Variational Autoencoder (VAE) with attention mechanisms. The architecture is designed for image generation tasks and follows modern practices for diffusion models.

## Architecture Overview

The project consists of three main components:

1. **Encoder (`encoder.py`)**: Converts input images into a latent representation.
2. **Decoder (`decoder.py`)**: Reconstructs images from the latent representation.
3. **Attention Mechanism (`attention.py`)**: Self-attention module used in both encoder and decoder.

### Key Components

- **Self-Attention**: Implements a multi-head self-attention mechanism that can be used with or without causal masking.
- **VAE Attention Block**: Applies self-attention to feature maps in the spatial domain.
- **VAE Residual Block**: Implements residual connections with group normalization and convolutional layers.
- **Variational Autoencoder**: Composed of encoder and decoder networks with a bottleneck that produces a latent representation.

## Technical Details

- The encoder downsamples the input through multiple convolutional and residual blocks, ultimately producing mean and log variance tensors for the latent space.
- The decoder upsamples from the latent representation back to image space using residual connections and attention mechanisms.
- The model uses SiLU activation functions and GroupNorm normalization layers.
- A special scaling factor (0.18215) is applied at the encoder output and decoder input for numerical stability.

## File Structure

```
├── attention.py   # Implementation of the self-attention mechanism
├── decoder.py     # VAE decoder implementation
└── encoder.py     # VAE encoder implementation
```

## Usage

To use this model for image generation or latent space manipulation:

1. Initialize the encoder and decoder
2. Pass your image through the encoder to get a latent representation
3. Apply your diffusion process in the latent space
4. Decode the processed latent representation to get the output image

```python
import torch
from encoder import VariationalAutoEncoder
from decoder import VariationalAutoDecoder

# Initialize models
encoder = VariationalAutoEncoder()
decoder = VariationalAutoDecoder()

# Process image
image = torch.randn(1, 3, 256, 256)  # Example image batch
noise = torch.randn(1, 4, 32, 32)    # Noise for the sampling process
latent = encoder(image, noise)
reconstruction = decoder(latent)
```

## Requirements

- PyTorch
- Python 3.x

## Notes

- This implementation follows the architecture commonly used in modern diffusion models like DALL-E and Stable Diffusion.
- The attention mechanism is crucial for capturing long-range dependencies in images.
- The VAE is designed to compress the image to a lower-dimensional latent space where the diffusion process can operate more efficiently.

## TODO

Several TODOs are mentioned in the code:

- Add dimension comments to show input/output changes in the encoder and decoder
- Document the specific purpose of some architectural choices
- Clarify the reasoning behind using group normalization in certain places
