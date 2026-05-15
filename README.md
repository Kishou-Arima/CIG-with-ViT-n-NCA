# CIG-with-ViT-n-NCA

This project explores Cellular Image Generation (CIG) using Vision Transformers (ViT) and Neural Cellular Automata (NCA). It includes code and experiments for generating and evolving images, with a focus on flowers and other natural patterns.

## Features

- Image generation using Neural Cellular Automata (NCA)
- Integration with Vision Transformers (ViT) for enhanced pattern recognition
- Pretrained model checkpoints for various flower types
- Visualization of generated and trained images
- Jupyter notebook for interactive experimentation

## Project Structure

```sh
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


## Pure-math GPU NCA prototype

A Python prototype is available at `src/pure_math_gpu_nca.py`. It uses the required CLIP ViT text encoder for prompt embeddings, then runs the NCA neural-network rollout as explicit CuPy mathematics on the GPU. Matplotlib is used to save or display the generated output. The NCA path avoids `torch.nn`, torchvision, diffusers, and custom CUDA/C++ code. Dataset training uses Pillow and SciPy only to read Flowers102 images and `.mat` split files, and the default trainer uses backprop through explicit tensor operations before exporting `.npz` weights back to the CuPy inference path.

Install the runtime pieces you actually need:

```sh
pip install torch matplotlib
pip install pillow scipy
pip install git+https://github.com/openai/CLIP.git
pip install cupy-cuda12x  # choose the CuPy wheel matching your CUDA version
```

Train CuPy NCA weights on the local Flowers102 dataset:

```sh
python src/pure_math_gpu_nca.py "yellow iris" --train --train-iters 200 --train-samples 16 --width 64 --height 64 --steps 24
```

This reads `src/.dataset/flowers102/flowers-102`, optimizes NCA arrays with backpropagation through the rollout, and saves weights to `src/checkpoints/pure_math_flowers102_nca.npz`. The old finite-difference trainer is still available with `--train-method spsa`.

Run a generation:

```sh
python src/pure_math_gpu_nca.py "yellow iris" --output yellow_iris.png --width 128 --height 128 --steps 48 --show
```

Generate with trained weights:

```sh
python src/pure_math_gpu_nca.py "yellow iris" --weights src/checkpoints/pure_math_flowers102_nca.npz --output yellow_iris_trained.png --show
```

See [`docs/PROJECT_UNDERSTANDING.md`](./docs/PROJECT_UNDERSTANDING.md) for a cell-by-cell explanation of `src/file.ipynb`, the dependency gap, and the migration path from the notebook to the CLIP-conditioned CuPy NCA implementation.

## Checkpoints

Pretrained model weights for various flowers will be in `src/checkpoints/`. You can use these to skip training and directly generate images.

## Requirements

- Python 3.8+
- PyTorch
- OpenAI CLIP
- CuPy (for the pure-math GPU NCA prototype)
- Pillow and SciPy (for local Flowers102 training)
- torchvision
- matplotlib
- diffusers
- transformers
- accelerate
- safetensors

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

If you use this code or ideas in your research, please cite the associated research paper:

> Conditional Image Generation with Vision Transformers and Neural Cellular Automata  
> *Author(s): Utkaarsh Saha, Alan Muthappan Rebeira, Dr. Shabnam Sadeghi Esfahlani*  
> *Year: 2025*  
> *Anglia Ruskin University, EBE Conference 2025*
> *Paper Under Review*

This project is intended for research and educational purposes.
Use at your own risk.
