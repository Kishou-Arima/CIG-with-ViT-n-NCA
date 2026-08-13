# Conditional Image Generation with ViT and NCA

A research project exploring **text-conditioned flower image generation** with a CLIP Vision Transformer (ViT) text encoder and a Neural Cellular Automaton (NCA). Given a prompt such as `"yellow iris"`, CLIP supplies a semantic conditioning vector and the NCA grows an image over a sequence of GPU-accelerated cellular updates.

The main implementation is [`src/pure_math_gpu_nca.py`](src/pure_math_gpu_nca.py). It keeps the NCA rollout explicit: convolution, FiLM conditioning, update masks, and state evolution are implemented as CuPy array operations. PyTorch is used to obtain frozen CLIP embeddings and, when training, to calculate gradients before the learned NCA weights are exported as portable `.npz` files.

## Features

- CLIP ViT text embeddings for prompt conditioning.
- GPU-based NCA image synthesis using CuPy.
- Optional backpropagation or SPSA training on the Oxford 102 Flowers dataset.
- Positional features and an adjacent-pixel detail loss for sharper learned outputs.
- Exportable NCA checkpoints in `.npz` format.
- A CuPy/Torch rollout-parity regression test.

## Repository layout

```text
src/
  pure_math_gpu_nca.py              # Generator and training CLI
  checkpoints/
    pure_math_flowers102_nca.npz    # Included trained checkpoint
    azalea_positional_nca.npz       # Included trained checkpoint
tests/
  test_rollout_parity.py            # CuPy/Torch rollout parity test
docs/
  PROJECT_UNDERSTANDING.md          # Background and notebook migration notes
```

Local datasets, generated images, and most checkpoint outputs are intentionally not required for a fresh clone.

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU with a working CUDA installation
- A CuPy wheel that matches the installed CUDA version

Create and activate a virtual environment, then install the dependencies:

```sh
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install numpy torch matplotlib pillow scipy
python -m pip install cupy-cuda12x  # Choose the wheel matching your CUDA version.
python -m pip install git+https://github.com/openai/CLIP.git
```

`pillow` and `scipy` are needed for local Flowers102 training. Generation requires CUDA-capable CuPy; there is no CPU inference fallback.

## Generate an image

Generate an image from a text prompt:

```sh
python src/pure_math_gpu_nca.py "yellow iris" --output yellow_iris.png --width 128 --height 128 --steps 48 --show
```

To use an exported checkpoint, pass it with `--weights`:

```sh
python src/pure_math_gpu_nca.py "yellow iris" --weights src/checkpoints/pure_math_flowers102_nca.npz --output yellow_iris_trained.png --show
```

Useful controls include `--width`, `--height`, `--steps`, `--update-rate`, `--clip-model`, `--clip-device`, and `--pigment-strength`.

## Train on Flowers102

Download and extract the [Oxford 102 Flowers dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) so the directory contains `jpg/`, `imagelabels.mat`, and `setid.mat`. By default, the expected location is:

```text
src/.dataset/flowers102/flowers-102/
```

Start a small training run with:

```sh
python src/pure_math_gpu_nca.py "yellow iris" --train --train-iters 200 --train-samples 16 --width 64 --height 64 --steps 24
```

Backpropagation is the default and recommended training method. It trains against one reference image by default (`--train-target exemplar`) for a sharper deterministic output. Use `--train-target mean` to learn a softer class average, or `--train-method spsa` for the finite-difference alternative.

By default, checkpoints are written to `src/checkpoints/pure_math_flowers102_nca.npz`. Common training options are:

- `--train-class`: Flowers102 class name or 1-based ID; defaults to the prompt.
- `--train-scope`: Select `output`, `update`, or `all` NCA parameters to optimize. CLIP always remains frozen.
- `--detail-loss-weight`: Tune the adjacent-pixel detail loss.
- `--validation-split` and `--validation-samples`: Configure held-out evaluation.
- `--save-weights`: Choose a checkpoint destination.

## Verify rollout parity

The regression test checks that CuPy inference and the PyTorch backpropagation rollout produce matching results for identical weights, conditioning vectors, masks, and pigment settings:

```sh
python -m unittest tests/test_rollout_parity.py -v
```

## Documentation

[`docs/PROJECT_UNDERSTANDING.md`](docs/PROJECT_UNDERSTANDING.md) explains the earlier notebook, the NCA architecture, and the migration to the current CLIP-conditioned CuPy implementation.

## License

This project is distributed under the [MIT License](LICENSE). It may be used, modified, and redistributed subject to the license terms.

## Citation

If this work contributes to your research, please cite:

> *Conditional Image Generation with Vision Transformers and Neural Cellular Automata*<br>
> Utkaarsh Saha, Alan Muthappan Rebeira, and Dr. Shabnam Sadeghi Esfahlani<br>
> Anglia Ruskin University, EBE Conference 2025<br>
> Paper under review
