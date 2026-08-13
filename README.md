# CIG with ViT and NCA

This research project explores conditional image generation with a CLIP Vision Transformer text encoder and a Neural Cellular Automaton (NCA), with a focus on Flowers102 images.

The main implementation is [`src/pure_math_gpu_nca.py`](./src/pure_math_gpu_nca.py). It uses CLIP for text embeddings and performs the NCA rollout as explicit CuPy GPU array mathematics. Torch is used for CLIP and for the optional backpropagation trainer; trained NCA weights are exported as portable `.npz` files for CuPy inference.

## Project layout

```text
src/
  pure_math_gpu_nca.py     # CLIP-conditioned CuPy NCA and trainer
  file.ipynb               # Earlier interactive experiments
  .dataset/flowers102/     # Local Flowers102 dataset (not committed)
  checkpoints/             # Exported NCA weights
tests/
  test_rollout_parity.py   # CuPy/Torch rollout parity regression test
```

## Setup

Use a Python virtual environment, then install the runtime dependencies:

```sh
pip install torch matplotlib pillow scipy
pip install git+https://github.com/openai/CLIP.git
pip install cupy-cuda12x  # Choose the CuPy wheel matching your CUDA version.
```

The generator requires a CUDA-capable CuPy installation. Pillow and SciPy are needed only for local Flowers102 training.

## Generate an image

```sh
python src/pure_math_gpu_nca.py "yellow iris" --output yellow_iris.png --width 128 --height 128 --steps 48 --show
```

Generate with previously trained weights:

```sh
python src/pure_math_gpu_nca.py "yellow iris" --weights src/checkpoints/pure_math_flowers102_nca.npz --output yellow_iris_trained.png --show
```

## Train on Flowers102

Place the dataset at `src/.dataset/flowers102/flowers-102`, containing `jpg/`, `imagelabels.mat`, and `setid.mat`, then run:

```sh
python src/pure_math_gpu_nca.py "yellow iris" --train --train-iters 200 --train-samples 16 --width 64 --height 64 --steps 24
```

Backpropagation is the default training method. The default `--train-target exemplar` learns a single reference photo, which is the recommended mode for a sharp deterministic result; `--train-target mean` deliberately learns a softer class average. The trainer also includes a positional feature basis and an adjacent-pixel detail loss; tune the latter with `--detail-loss-weight`.

Use `--train-method spsa` for the finite-difference alternative. Training evaluates held-out validation images by default; configure this with `--validation-split` and `--validation-samples`. CLIP remains frozen in all training scopes; `--train-scope all` additionally trains the NCA's CLIP projection arrays.

The trained weights are saved to `src/checkpoints/pure_math_flowers102_nca.npz` by default.

## Verify backend parity

The regression test checks that CuPy inference and the Torch backprop rollout produce matching results for the same weights, conditioning vector, masks, and pigment setting:

```sh
python -m unittest tests/test_rollout_parity.py -v
```

See [`docs/PROJECT_UNDERSTANDING.md`](./docs/PROJECT_UNDERSTANDING.md) for a cell-by-cell explanation of the earlier notebook and the migration path to the CLIP-conditioned CuPy NCA.

## License

This project is licensed under the MIT License. It is intended for research and educational use.

If you use this work in research, please cite:

> Conditional Image Generation with Vision Transformers and Neural Cellular Automata
>
> *Utkaarsh Saha, Alan Muthappan Rebeira, Dr. Shabnam Sadeghi Esfahlani*
>
> *Anglia Ruskin University, EBE Conference 2025*
>
> *Paper under review*
