# Project understanding and pure-math GPU direction

## What the current project does

This repository is a research notebook project for **Conditional Image Generation with Vision Transformers and Neural Cellular Automata**. The README describes the intended system as Cellular Image Generation (CIG) using a Vision Transformer text/image encoder and Neural Cellular Automata (NCA) to generate flower images.

The executable work is concentrated in `src/file.ipynb`:

1. **Dataset and text embeddings**
   - Downloads the Oxford 102 Flowers dataset through `torchvision.datasets.Flowers102`.
   - Resizes images to `128x128` tensors.
   - Defines the 102 flower class names.
   - Loads CLIP `ViT-B/32` and converts prompts/class names into normalized 512-dimensional text embeddings.

2. **Dataset visualization**
   - Displays all images of a target class, currently `yellow iris`, across train/validation/test splits.

3. **Condition projection**
   - Defines `TextConditioningHead`, a small PyTorch MLP that maps each CLIP 512-dimensional text embedding into a normalized 128-dimensional conditioning vector for the NCA.

4. **FiLM-conditioned Neural Cellular Automata**
   - Defines `FiLMNCA` with:
     - a 32-channel cellular state, where the first 3 channels are RGB and the rest are hidden state;
     - a 3x3 convolutional perception layer;
     - a FiLM layer that turns the 128-dimensional condition vector into per-feature scale/shift terms;
     - a 1x1-convolution update MLP;
     - stochastic per-cell updates;
     - RGB clamping after each rollout step.
   - Starts from a small central hidden-state seed and iterates the NCA to form an image.

5. **Training loop for per-class experts**
   - Wraps Flowers102 samples with precomputed text embeddings.
   - Trains per-class NCA experts, currently defaulting to `yellow iris` and `giant white arum lily`.
   - Uses a combined pixel L1 loss and CLIP image/text similarity loss.
   - Saves latest, best, and final PyTorch checkpoints.

6. **Diffusion baseline / fast generation cell**
   - Loads `stabilityai/sd-turbo` through `diffusers` and generates images from a text prompt.
   - This is a separate diffusion-model baseline, not the NCA itself.

## Current dependency profile

The notebook depends on a high-level Python ML stack:

- PyTorch / CUDA through PyTorch
- torchvision
- OpenAI CLIP
- matplotlib
- tqdm
- diffusers / transformers / accelerate / safetensors for the SD-Turbo baseline
- Jupyter execution environment

That means the current version is **not** pure math in the sense of standalone GPU kernels. Most operations are delegated to ML frameworks and pretrained models.

## What “pure mathematics running on GPU, with no extra libraries” can realistically mean

A fully faithful clone of the notebook cannot be created without either:

1. keeping pretrained model weights and reimplementing the necessary tensor kernels, or
2. replacing learned components with analytic mathematical functions.

The first route is possible but large: CLIP text encoding, CLIP image encoding, convolution layers, normalization, optimizer, checkpoint parsing, and diffusion inference would all need standalone implementations and exported weights.

The practical route added in this repository keeps the useful learned/text component and makes the NCA itself pure GPU mathematics: a **CLIP-conditioned CuPy NCA prototype**. It uses:

- CLIP's ViT text encoder for the prompt embedding, because that is the conditioning signal the notebook is built around;
- a CuPy implementation of the NCA neural network, where the 3x3 perception convolution, FiLM conditioning, 1x1 update network, stochastic cell mask, and rollout loop are explicit array mathematics on the GPU;
- optional `.npz` weight loading so exported notebook weights can replace the deterministic analytic fallback weights later;
- matplotlib for saving and displaying generated outputs.

This produces mathematically generated, CLIP-conditioned flower-like images. It is not expected to match the trained notebook output until learned NCA and conditioning-head weights are exported and loaded.

## Migration plan toward a faithful Python GPU-math version

1. **CLIP-conditioned CuPy inference path** — done in `src/pure_math_gpu_nca.py`.
2. **Export trained NCA and conditioning-head weights** from PyTorch checkpoints into `.npz` arrays.
3. **Replace the deterministic fallback weights** with exported `cond_w`, `cond_b`, `perc_w`, `film_w`, `film_b`, `update_w1`, `update_b1`, `update_w2`, and `update_b2` arrays.
4. **Keep CLIP for ViT embeddings**, while keeping the NCA rollout independent of `torch.nn` and expressed as CuPy math.
5. **Optional training**:
   - a full CuPy training path would require hand-written gradients or another autodiff choice;
   - a practical split is notebook-based training plus CuPy-only NCA inference.
