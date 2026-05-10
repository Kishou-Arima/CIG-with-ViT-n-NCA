#!/usr/bin/env python3
"""CLIP-conditioned NCA whose neural-network rollout is explicit GPU math.

This file keeps the parts the project actually needs:
- CLIP for ViT text embeddings.
- CuPy for the NCA neural-network math on GPU.
- Matplotlib for displaying/saving generated outputs.

The NCA itself does not use ``torch.nn`` or a high-level training/inference
framework.  Its convolution, FiLM conditioning, 1x1 update network, stochastic
cell mask, and rollout loop are written as array mathematics over CuPy tensors.

Example:
    python src/pure_math_gpu_nca.py "yellow iris" --output yellow_iris.png --show
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

TAU = 2.0 * math.pi
CLIP_DIM = 512
COND_DIM = 128
STATE_CH = 32
PERCEPT_CH = 64
HIDDEN_CH = 64
DEFAULT_STEPS = 48
DEFAULT_UPDATE_RATE = 0.5

cp = None
plt = None
torch = None
clip = None


def require_module(module_name: str, package_hint: str) -> None:
    """Fail early with a clear install hint before importing optional modules."""

    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(f"Missing required module '{module_name}'. Install it with: {package_hint}")


def load_runtime_modules() -> None:
    """Load the required runtime packages after CLI parsing.

    Imports are intentionally lazy so ``--help`` works in lightweight
    environments.  No import is wrapped in try/except.
    """

    require_module("cupy", "pip install cupy-cuda12x  # choose the wheel matching your CUDA version")
    require_module("torch", "pip install torch")
    require_module("clip", "pip install git+https://github.com/openai/CLIP.git")
    require_module("matplotlib", "pip install matplotlib")

    global cp, plt, torch, clip
    import cupy as cupy_module
    import matplotlib.pyplot as pyplot_module
    import torch as torch_module
    import clip as clip_module

    cp = cupy_module
    plt = pyplot_module
    torch = torch_module
    clip = clip_module


def xp() -> object:
    """Return the loaded CuPy module."""

    if cp is None:
        raise RuntimeError("CuPy has not been loaded. Call load_runtime_modules() first.")
    return cp


def fract(x):
    """Fractional part for CuPy arrays."""

    array_api = xp()
    return x - array_api.floor(x)


def analytic_weight(rows: int, cols: int, salt: int, scale: float = 1.0):
    """Create deterministic mathematical weights directly on the GPU."""

    array_api = xp()
    r = array_api.arange(rows, dtype=array_api.float32).reshape(rows, 1)
    c = array_api.arange(cols, dtype=array_api.float32).reshape(1, cols)
    values = array_api.sin((r + 1.0) * 12.9898 + (c + 1.0) * 78.233 + salt * 37.719)
    return scale * (2.0 * fract(values * 43758.5453) - 1.0).astype(array_api.float32)


def analytic_conv_weight(out_ch: int, in_ch: int, kernel: int, salt: int, scale: float):
    """Create deterministic 3x3 convolution weights as math, not nn layers."""

    array_api = xp()
    flat = analytic_weight(out_ch, in_ch * kernel * kernel, salt, scale=scale)
    return flat.reshape(out_ch, in_ch, kernel, kernel).astype(array_api.float32)


def load_or_create_nca_weights(weight_path: Path | None = None) -> dict[str, object]:
    """Load exported NCA weights or create deterministic math weights.

    Expected ``.npz`` keys, when supplied, are:
    ``cond_w``, ``cond_b``, ``perc_w``, ``film_w``, ``film_b``,
    ``update_w1``, ``update_b1``, ``update_w2``, and ``update_b2``.
    """

    array_api = xp()
    if weight_path is not None:
        data = array_api.load(weight_path)
        return {key: data[key].astype(array_api.float32) for key in data.files}

    return {
        "cond_w": analytic_weight(CLIP_DIM, COND_DIM, 101, scale=1.0 / math.sqrt(CLIP_DIM)),
        "cond_b": analytic_weight(1, COND_DIM, 103, scale=0.02).reshape(COND_DIM),
        "perc_w": analytic_conv_weight(PERCEPT_CH, STATE_CH, 3, 107, scale=1.0 / math.sqrt(STATE_CH * 9)),
        "film_w": analytic_weight(COND_DIM, 2 * PERCEPT_CH, 109, scale=1.0 / math.sqrt(COND_DIM)),
        "film_b": analytic_weight(1, 2 * PERCEPT_CH, 113, scale=0.05).reshape(2 * PERCEPT_CH),
        "update_w1": analytic_weight(PERCEPT_CH, HIDDEN_CH, 127, scale=1.0 / math.sqrt(PERCEPT_CH)),
        "update_b1": analytic_weight(1, HIDDEN_CH, 131, scale=0.02).reshape(HIDDEN_CH),
        "update_w2": analytic_weight(HIDDEN_CH, STATE_CH, 137, scale=0.35 / math.sqrt(HIDDEN_CH)),
        "update_b2": analytic_weight(1, STATE_CH, 139, scale=0.005).reshape(STATE_CH),
    }


def clip_text_embedding(prompt: str, clip_model_name: str, clip_device: str):
    """Use CLIP's ViT text tower to produce a normalized 512-D embedding."""

    if clip_device == "auto":
        clip_device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = clip.load(clip_model_name, device=clip_device)
    model.eval()
    tokens = clip.tokenize([prompt]).to(clip_device)
    with torch.no_grad():
        embedding = model.encode_text(tokens).float()
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze(0).detach().cpu().numpy().astype("float32")


def project_clip_to_condition(clip_embedding, weights: dict[str, object]):
    """Project the CLIP vector into the NCA conditioning space with CuPy math."""

    array_api = xp()
    z = array_api.asarray(clip_embedding, dtype=array_api.float32)
    z = z / array_api.maximum(array_api.linalg.norm(z), array_api.float32(1.0e-8))
    cond = array_api.tanh(z @ weights["cond_w"] + weights["cond_b"])
    return cond / array_api.maximum(array_api.linalg.norm(cond), array_api.float32(1.0e-8))


def coordinate_grid(width: int, height: int):
    """Normalized coordinate grids shaped [H, W]."""

    array_api = xp()
    y, x = array_api.meshgrid(
        array_api.arange(height, dtype=array_api.float32),
        array_api.arange(width, dtype=array_api.float32),
        indexing="ij",
    )
    nx = (2.0 * x - width) / float(width)
    ny = (2.0 * y - height) / float(height)
    return nx, ny


def seed_state(width: int, height: int, cond):
    """Create a central hidden-state seed for NCA growth."""

    array_api = xp()
    state = array_api.zeros((STATE_CH, height, width), dtype=array_api.float32)
    nx, ny = coordinate_grid(width, height)
    seed = array_api.exp(-90.0 * (nx * nx + ny * ny)).astype(array_api.float32)

    for channel in range(3, STATE_CH):
        phase = cond[channel % COND_DIM] * TAU
        state[channel] = seed * (0.5 + 0.5 * array_api.sin(phase + float(channel)))

    state[0] = 0.03 * seed
    state[1] = 0.025 * seed
    state[2] = 0.02 * seed
    return state, nx, ny


def conv3x3_same(state, kernel):
    """Explicit 3x3 convolution written as CuPy tensor math."""

    array_api = xp()
    padded = array_api.pad(state, ((0, 0), (1, 1), (1, 1)), mode="edge")
    out_ch = kernel.shape[0]
    height, width = state.shape[1:]
    result = array_api.zeros((out_ch, height, width), dtype=array_api.float32)

    for ky in range(3):
        for kx in range(3):
            window = padded[:, ky : ky + height, kx : kx + width]
            result += array_api.tensordot(kernel[:, :, ky, kx], window, axes=([1], [0]))
    return result


def deterministic_mask(width: int, height: int, step: int, update_rate: float):
    """Deterministic stochastic update mask computed on GPU."""

    array_api = xp()
    y, x = array_api.meshgrid(
        array_api.arange(height, dtype=array_api.float32),
        array_api.arange(width, dtype=array_api.float32),
        indexing="ij",
    )
    noise = fract(array_api.sin(x * 12.9898 + y * 78.233 + step * 37.719) * 43758.5453)
    return (noise <= update_rate).astype(array_api.float32).reshape(1, height, width)


def nca_step(state, cond, weights: dict[str, object], step: int, update_rate: float, nx, ny):
    """One FiLM-conditioned NCA update, all in CuPy math."""

    array_api = xp()
    perception = conv3x3_same(state, weights["perc_w"])

    film = cond @ weights["film_w"] + weights["film_b"]
    gamma, beta = film[:PERCEPT_CH], film[PERCEPT_CH:]
    perception = perception * (1.0 + gamma.reshape(PERCEPT_CH, 1, 1)) + beta.reshape(PERCEPT_CH, 1, 1)

    hidden = array_api.maximum(
        array_api.tensordot(weights["update_w1"].T, perception, axes=([1], [0]))
        + weights["update_b1"].reshape(HIDDEN_CH, 1, 1),
        0.0,
    )
    delta = (
        array_api.tensordot(weights["update_w2"].T, hidden, axes=([1], [0]))
        + weights["update_b2"].reshape(STATE_CH, 1, 1)
    )

    next_state = state + 0.025 * delta * deterministic_mask(state.shape[2], state.shape[1], step, update_rate)
    next_state[3:] *= 0.992

    angle = array_api.arctan2(ny, nx)
    radial = array_api.sqrt(nx * nx + ny * ny)
    for channel in range(3):
        pigment = 0.035 * array_api.sin(
            (channel + 2.0) * angle + 18.0 * radial + 5.0 * cond[(channel + 7) % COND_DIM]
        )
        next_state[channel] = array_api.clip(next_state[channel] + pigment, 0.0, 1.0)

    return next_state.astype(array_api.float32)


def generate(prompt: str, args: argparse.Namespace):
    """Encode text with CLIP, then run the NCA neural network as GPU math."""

    weights = load_or_create_nca_weights(Path(args.weights) if args.weights else None)
    embedding = clip_text_embedding(prompt, args.clip_model, args.clip_device)
    cond = project_clip_to_condition(embedding, weights)
    state, nx, ny = seed_state(args.width, args.height, cond)

    for step in range(args.steps):
        state = nca_step(state, cond, weights, step, args.update_rate, nx, ny)

    return cp.asnumpy(cp.clip(state[:3], 0.0, 1.0).transpose(1, 2, 0))


def display_or_save(rgb, prompt: str, output: Path | None, show: bool) -> None:
    """Display and/or save the generated RGB array with matplotlib."""

    figure = plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(rgb)
    plt.title(f'CLIP-conditioned CuPy NCA — "{prompt}"')
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight", pad_inches=0.05, dpi=160)
    if show:
        plt.show()
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLIP-conditioned NCA with neural-network math executed in CuPy on GPU.")
    parser.add_argument("prompt", nargs="?", default="yellow iris", help="Text prompt passed through CLIP's ViT text encoder.")
    parser.add_argument("--output", default="pure_math_gpu_nca.png", help="Image path saved through matplotlib.")
    parser.add_argument("--show", action="store_true", help="Display the generated image with matplotlib.")
    parser.add_argument("--width", type=int, default=128, help="Output image width.")
    parser.add_argument("--height", type=int, default=128, help="Output image height.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="NCA rollout steps.")
    parser.add_argument("--update-rate", type=float, default=DEFAULT_UPDATE_RATE, help="Cell update rate in (0, 1].")
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP model name used for ViT text embeddings.")
    parser.add_argument("--clip-device", default="auto", help="Device for CLIP: auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--weights", default=None, help="Optional .npz file with exported NCA weights.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if not (0.0 < args.update_rate <= 1.0):
        raise ValueError("--update-rate must be in (0, 1].")
    if args.weights is not None and not Path(args.weights).exists():
        raise ValueError(f"--weights file does not exist: {args.weights}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        load_runtime_modules()
        rgb = generate(args.prompt, args)
        display_or_save(rgb, args.prompt, Path(args.output) if args.output else None, args.show)
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(f"Generated {args.output} for prompt {args.prompt!r} using CLIP embeddings and CuPy NCA math.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
