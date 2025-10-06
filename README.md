# CIG-with-ViT-n-NCA

This project explores Cellular Image Generation (CIG) using Vision Transformers (ViT) and Neural Cellular Automata (NCA). It includes code and experiments for generating and evolving images, with a focus on flowers and other natural patterns.

## Features

- Image generation using Neural Cellular Automata (NCA)
- Integration with Vision Transformers (ViT) for enhanced pattern recognition
- Pretrained model checkpoints for various flower types
- Visualization of generated and trained images
- Jupyter notebook for interactive experimentation

## Project Structure

```
├── src/
│   ├── file.ipynb                # Main Jupyter notebook for experiments
├── README.md                     # Project documentation
```

## Getting Started

1. **Clone the repository**

    ```sh
    git clone <repo-url>
    cd CIG-with-ViT-n-NCA
    ```

2. **Install dependencies**

```sh
	```sh
	pip install torch torchvision matplotlib diffusers transformers accelerate safetensors
	```
	- Recommended: Use a virtual environment (e.g., `venv` or `conda`).
	- Install required packages:
	  ```sh
	  pip install torch torchvision matplotlib diffusers transformers accelerate safetensors
	  ```
3. **Run the notebook**
	- Open `src/file.ipynb` in Jupyter Lab or VS Code.
	- Follow the cells to train, generate, and visualize images.

## Checkpoints
Pretrained model weights for various flowers will be in `src/checkpoints/`. You can use these to skip training and directly generate images.



## Requirements
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- diffusers
- transformers
- accelerate
- safetensors

## License
This project is for research and educational purposes.

