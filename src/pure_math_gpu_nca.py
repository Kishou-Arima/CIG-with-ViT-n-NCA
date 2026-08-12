from __future__ import annotations

import argparse
import importlib.util
import math
import numpy as np
import warnings
from pathlib import Path

TAU = 2.0 * math.pi
CLIP_DIM = 512
COND_DIM = 128
STATE_CH = 32
PERCEPT_CH = 64
HIDDEN_CH = 64
DEFAULT_STEPS = 48
DEFAULT_UPDATE_RATE = 0.5
DEFAULT_PIGMENT_STRENGTH = 0.0
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / ".dataset" / "flowers102" / "flowers-102"
DEFAULT_SAVE_WEIGHTS = SCRIPT_DIR / "checkpoints" / "pure_math_flowers102_nca.npz"
FLOWERS102_CLASS_NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]

cp = None
plt = None
torch = None
clip = None
Image = None
loadmat = None


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
    require_module("packaging", "pip install packaging")
    require_module("clip", "pip install git+https://github.com/openai/CLIP.git")
    require_module("matplotlib", "pip install matplotlib")

    global cp, plt, torch, clip
    import cupy as cupy_module
    import matplotlib.pyplot as pyplot_module
    import packaging as packaging_module
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import pkg_resources as pkg_resources_module
    import torch as torch_module

    if not hasattr(pkg_resources_module, "packaging"):
        pkg_resources_module.packaging = packaging_module

    import clip as clip_module

    cp = cupy_module
    plt = pyplot_module
    torch = torch_module
    clip = clip_module


def load_training_modules() -> None:
    """Load image and MATLAB-file readers needed only for dataset training."""

    require_module("PIL", "pip install pillow")
    require_module("scipy", "pip install scipy")

    global Image, loadmat
    from PIL import Image as pillow_image
    from scipy.io import loadmat as scipy_loadmat

    Image = pillow_image
    loadmat = scipy_loadmat


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
        try:
            if hasattr(data, "files"):
                keys = data.files
            elif hasattr(data, "keys"):
                keys = list(data.keys())
            elif hasattr(data, "npz_file") and hasattr(data.npz_file, "files"):
                keys = data.npz_file.files
            else:
                raise ValueError(f"Could not read keys from .npz weights file: {weight_path}")
            return {key: array_api.asarray(data[key], dtype=array_api.float32) for key in keys}
        finally:
            if hasattr(data, "close"):
                data.close()

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


def resolve_flowers102_root(dataset_root: str | Path) -> Path:
    """Return the validated local Flowers102 root."""

    root = Path(dataset_root).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    if not root.exists():
        raise ValueError(f"--dataset-root does not exist: {root}")
    if not (root / "jpg").is_dir():
        raise ValueError(f"--dataset-root must contain a jpg directory: {root}")
    if not (root / "imagelabels.mat").is_file():
        raise ValueError(f"--dataset-root must contain imagelabels.mat: {root}")
    if not (root / "setid.mat").is_file():
        raise ValueError(f"--dataset-root must contain setid.mat: {root}")
    return root


def flowers102_class_id(name_or_id: str) -> int:
    """Convert a Flowers102 class name or 1-based class id into an id."""

    text = str(name_or_id).strip().lower()
    if text.isdigit():
        class_id = int(text)
        if 1 <= class_id <= len(FLOWERS102_CLASS_NAMES):
            return class_id
        raise ValueError(f"Flowers102 class id must be 1..102, got {class_id}.")
    for idx, class_name in enumerate(FLOWERS102_CLASS_NAMES, start=1):
        if text == class_name:
            return idx
    choices = ", ".join(FLOWERS102_CLASS_NAMES[:8])
    raise ValueError(f"Unknown Flowers102 class {name_or_id!r}. Examples: {choices}, ...")


def split_image_ids(dataset_root: Path, split: str) -> np.ndarray:
    """Read Flowers102 split ids from setid.mat as zero-based image ids."""

    if loadmat is None:
        raise RuntimeError("Training modules have not been loaded. Call load_training_modules() first.")
    split_data = loadmat(dataset_root / "setid.mat")
    split_keys = {
        "train": ["trnid"],
        "val": ["valid"],
        "test": ["tstid"],
        "trainval": ["trnid", "valid"],
        "all": ["trnid", "valid", "tstid"],
    }
    ids = [np.asarray(split_data[key]).reshape(-1) for key in split_keys[split]]
    return np.concatenate(ids).astype(np.int64) - 1


def flowers102_image_paths(dataset_root: Path, class_name: str, split: str, max_samples: int) -> list[Path]:
    """Return image paths for one Flowers102 class and split."""

    if loadmat is None:
        raise RuntimeError("Training modules have not been loaded. Call load_training_modules() first.")
    class_id = flowers102_class_id(class_name)
    labels = np.asarray(loadmat(dataset_root / "imagelabels.mat")["labels"]).reshape(-1).astype(np.int64)
    image_ids = split_image_ids(dataset_root, split)
    selected = [image_id for image_id in image_ids if labels[image_id] == class_id]
    if max_samples > 0:
        selected = selected[:max_samples]
    if not selected:
        raise ValueError(f"No Flowers102 images found for class {class_name!r} in split {split!r}.")
    return [dataset_root / "jpg" / f"image_{image_id + 1:05d}.jpg" for image_id in selected]


def load_training_images(paths: list[Path], width: int, height: int) -> list[np.ndarray]:
    """Load Flowers102 images as HWC float32 arrays in [0, 1]."""

    if Image is None:
        raise RuntimeError("Training modules have not been loaded. Call load_training_modules() first.")
    images = []
    for path in paths:
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            images.append(np.asarray(resized, dtype=np.float32) / 255.0)
    return images


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


def nca_step(state, cond, weights: dict[str, object], step: int, update_rate: float, nx, ny, pigment_strength: float):
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

    if pigment_strength != 0.0:
        angle = array_api.arctan2(ny, nx)
        radial = array_api.sqrt(nx * nx + ny * ny)
        for channel in range(3):
            pigment = pigment_strength * array_api.sin(
                (channel + 2.0) * angle + 18.0 * radial + 5.0 * cond[(channel + 7) % COND_DIM]
            )
            next_state[channel] = array_api.clip(next_state[channel] + pigment, 0.0, 1.0)

    return next_state.astype(array_api.float32)


def nca_rollout(
    cond,
    weights: dict[str, object],
    width: int,
    height: int,
    steps: int,
    update_rate: float,
    pigment_strength: float,
):
    """Run the NCA forward rollout and return RGB output plus final state."""

    state, nx, ny = seed_state(width, height, cond)
    for step in range(steps):
        state = nca_step(state, cond, weights, step, update_rate, nx, ny, pigment_strength)
    rgb = cp.clip(state[:3], 0.0, 1.0).transpose(1, 2, 0)
    return rgb, state


def target_loss(weights: dict[str, object], cond, target_rgb, args: argparse.Namespace):
    """Mean squared image loss between NCA output and one target image."""

    rgb, _ = nca_rollout(cond, weights, args.width, args.height, args.steps, args.update_rate, args.pigment_strength)
    diff = rgb - target_rgb
    return cp.mean(diff * diff)


def trainable_weight_keys(scope: str) -> list[str]:
    """Select which mathematical NCA arrays are optimized by SPSA."""

    scopes = {
        "output": ["update_w2", "update_b2"],
        "update": ["film_w", "film_b", "update_w1", "update_b1", "update_w2", "update_b2"],
        "all": ["cond_w", "cond_b", "perc_w", "film_w", "film_b", "update_w1", "update_b1", "update_w2", "update_b2"],
    }
    return scopes[scope]


def save_nca_weights(weights: dict[str, object], output_path: Path) -> None:
    """Save CuPy weights as a NumPy .npz file for later inference."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **{key: cp.asnumpy(value) for key, value in weights.items()})


def torch_device_name(clip_device: str) -> str:
    """Choose a torch device for backprop training."""

    if clip_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return clip_device


def torch_from_weights(weights: dict[str, object], device: str, train_keys: list[str]) -> dict[str, object]:
    """Copy CuPy weights into torch tensors for explicit-math backprop."""

    tensor_weights = {}
    for key, value in weights.items():
        tensor = torch.tensor(cp.asnumpy(value), dtype=torch.float32, device=device)
        tensor.requires_grad_(key in train_keys)
        tensor_weights[key] = tensor
    return tensor_weights


def save_torch_weights(weights: dict[str, object], output_path: Path) -> None:
    """Save torch tensor weights as .npz arrays consumed by CuPy inference."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **{key: value.detach().cpu().numpy() for key, value in weights.items()})


def torch_fract(x):
    return x - torch.floor(x)


def torch_coordinate_grid(width: int, height: int, device: str):
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    nx = (2.0 * x - width) / float(width)
    ny = (2.0 * y - height) / float(height)
    return nx, ny


def torch_seed_state(width: int, height: int, cond, device: str):
    state = torch.zeros((STATE_CH, height, width), dtype=torch.float32, device=device)
    nx, ny = torch_coordinate_grid(width, height, device)
    seed = torch.exp(-90.0 * (nx * nx + ny * ny)).to(torch.float32)

    channels = []
    for channel in range(STATE_CH):
        if channel == 0:
            channels.append(0.03 * seed)
        elif channel == 1:
            channels.append(0.025 * seed)
        elif channel == 2:
            channels.append(0.02 * seed)
        else:
            phase = cond[channel % COND_DIM] * TAU
            channels.append(seed * (0.5 + 0.5 * torch.sin(phase + float(channel))))
    state = torch.stack(channels, dim=0)
    return state, nx, ny


def torch_conv3x3_same(state, kernel):
    padded = torch.nn.functional.pad(state.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    return torch.nn.functional.conv2d(padded, kernel).squeeze(0)


def torch_deterministic_mask(width: int, height: int, step: int, update_rate: float, device: str):
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    noise = torch_fract(torch.sin(x * 12.9898 + y * 78.233 + step * 37.719) * 43758.5453)
    return (noise <= update_rate).to(torch.float32).reshape(1, height, width)


def torch_nca_step(state, cond, weights: dict[str, object], step: int, update_rate: float, nx, ny, pigment_strength: float):
    perception = torch_conv3x3_same(state, weights["perc_w"])
    film = cond @ weights["film_w"] + weights["film_b"]
    gamma, beta = film[:PERCEPT_CH], film[PERCEPT_CH:]
    perception = perception * (1.0 + gamma.reshape(PERCEPT_CH, 1, 1)) + beta.reshape(PERCEPT_CH, 1, 1)

    hidden = torch.relu(
        torch.tensordot(weights["update_w1"].T, perception, dims=([1], [0]))
        + weights["update_b1"].reshape(HIDDEN_CH, 1, 1)
    )
    delta = (
        torch.tensordot(weights["update_w2"].T, hidden, dims=([1], [0]))
        + weights["update_b2"].reshape(STATE_CH, 1, 1)
    )

    mask = torch_deterministic_mask(state.shape[2], state.shape[1], step, update_rate, state.device)
    next_state = state + 0.025 * delta * mask
    decay = torch.ones((STATE_CH, 1, 1), dtype=torch.float32, device=state.device)
    decay[3:] = 0.992
    next_state = next_state * decay

    if pigment_strength != 0.0:
        angle = torch.atan2(ny, nx)
        radial = torch.sqrt(nx * nx + ny * ny)
        rgb_channels = []
        for channel in range(3):
            pigment = pigment_strength * torch.sin(
                (channel + 2.0) * angle + 18.0 * radial + 5.0 * cond[(channel + 7) % COND_DIM]
            )
            rgb_channels.append(torch.clamp(next_state[channel] + pigment, 0.0, 1.0))
        next_state = torch.cat([torch.stack(rgb_channels, dim=0), next_state[3:]], dim=0)
    return next_state


def torch_nca_rollout(cond, weights: dict[str, object], width: int, height: int, steps: int, update_rate: float, pigment_strength: float):
    state, nx, ny = torch_seed_state(width, height, cond, cond.device)
    for step in range(steps):
        state = torch_nca_step(state, cond, weights, step, update_rate, nx, ny, pigment_strength)
    return torch.clamp(state[:3], 0.0, 1.0).permute(1, 2, 0), state


def manual_adam_step(params: list[object], moments: dict[int, tuple[object, object]], step: int, learning_rate: float) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    with torch.no_grad():
        for param in params:
            if param.grad is None:
                continue
            ident = id(param)
            if ident not in moments:
                moments[ident] = (torch.zeros_like(param), torch.zeros_like(param))
            m, v = moments[ident]
            grad = torch.clamp(param.grad, -1.0, 1.0)
            m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            param.addcdiv_(m_hat, torch.sqrt(v_hat) + eps, value=-learning_rate)
            param.clamp_(-5.0, 5.0)
            param.grad = None


def train_on_flowers102_backprop(args: argparse.Namespace) -> dict[str, object]:
    """Train with backprop through explicit tensor math, then export .npz weights."""

    load_training_modules()
    device = torch_device_name(args.clip_device)
    class_name = args.train_class or args.prompt
    paths = flowers102_image_paths(args.dataset_root, class_name, args.train_split, args.train_samples)
    targets_np = load_training_images(paths, args.width, args.height)
    targets = [torch.tensor(target, dtype=torch.float32, device=device) for target in targets_np]

    base_weights = load_or_create_nca_weights(Path(args.weights) if args.weights else None)
    keys = trainable_weight_keys(args.train_scope)
    weights = torch_from_weights(base_weights, device, keys)
    embedding = clip_text_embedding(class_name, args.clip_model, device)
    cond = project_clip_to_condition(embedding, base_weights)
    cond_t = torch.tensor(cp.asnumpy(cond), dtype=torch.float32, device=device)
    params = [weights[key] for key in keys]
    moments: dict[int, tuple[object, object]] = {}

    torch.manual_seed(args.seed)
    print(f"Backprop training on {len(targets)} Flowers102 image(s) for class {class_name!r}.")
    print(f"Optimizing {', '.join(keys)}; saving to {args.save_weights}.")

    for iteration in range(1, args.train_iters + 1):
        target = targets[(iteration - 1) % len(targets)]
        rgb, _ = torch_nca_rollout(
            cond_t,
            weights,
            args.width,
            args.height,
            args.steps,
            args.update_rate,
            args.pigment_strength,
        )
        mse = torch.mean((rgb - target) ** 2)
        l1 = torch.mean(torch.abs(rgb - target))
        loss = mse + 0.25 * l1
        loss.backward()
        manual_adam_step(params, moments, iteration, args.learning_rate)

        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.train_iters:
            print(f"iter {iteration:05d}/{args.train_iters}: loss={float(loss.detach()):.6f} mse={float(mse.detach()):.6f} l1={float(l1.detach()):.6f}")

    save_torch_weights(weights, Path(args.save_weights))
    return {key: cp.asarray(value.detach().cpu().numpy(), dtype=cp.float32) for key, value in weights.items()}


def train_on_flowers102(args: argparse.Namespace) -> dict[str, object]:
    """Train NCA weights against Flowers102 images using explicit CuPy math.

    The optimizer is SPSA: each step estimates a gradient from two loss
    evaluations under random +/- perturbations, then applies SGD directly to
    the CuPy weight arrays. This keeps training framework-free while still
    learning from real dataset pixels.
    """

    load_training_modules()
    class_name = args.train_class or args.prompt
    paths = flowers102_image_paths(args.dataset_root, class_name, args.train_split, args.train_samples)
    targets = load_training_images(paths, args.width, args.height)
    weights = load_or_create_nca_weights(Path(args.weights) if args.weights else None)
    embedding = clip_text_embedding(class_name, args.clip_model, args.clip_device)
    cond = project_clip_to_condition(embedding, weights)
    keys = trainable_weight_keys(args.train_scope)

    cp.random.seed(args.seed)
    print(f"Training on {len(targets)} Flowers102 image(s) for class {class_name!r}.")
    print(f"Optimizing {', '.join(keys)} with SPSA; saving to {args.save_weights}.")

    for iteration in range(1, args.train_iters + 1):
        target_np = targets[(iteration - 1) % len(targets)]
        target = cp.asarray(target_np, dtype=cp.float32)
        perturbations = {
            key: cp.where(cp.random.random(weights[key].shape) < 0.5, -1.0, 1.0).astype(cp.float32) for key in keys
        }

        for key in keys:
            weights[key] += args.spsa_eps * perturbations[key]
        loss_plus = target_loss(weights, cond, target, args)

        for key in keys:
            weights[key] -= 2.0 * args.spsa_eps * perturbations[key]
        loss_minus = target_loss(weights, cond, target, args)

        scale = (loss_plus - loss_minus) / (2.0 * args.spsa_eps)
        for key in keys:
            weights[key] += args.spsa_eps * perturbations[key]
            weights[key] -= args.learning_rate * scale * perturbations[key]
            weights[key] = cp.clip(weights[key], -5.0, 5.0)

        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.train_iters:
            current_loss = float(target_loss(weights, cond, target, args).get())
            plus = float(loss_plus.get())
            minus = float(loss_minus.get())
            print(f"iter {iteration:05d}/{args.train_iters}: loss={current_loss:.6f} spsa+= {plus:.6f} spsa-= {minus:.6f}")

    save_nca_weights(weights, Path(args.save_weights))
    return weights


def generate(prompt: str, args: argparse.Namespace):
    """Encode text with CLIP, then run the NCA neural network as GPU math."""

    weights = load_or_create_nca_weights(Path(args.weights) if args.weights else None)
    embedding = clip_text_embedding(prompt, args.clip_model, args.clip_device)
    cond = project_clip_to_condition(embedding, weights)
    rgb, _ = nca_rollout(cond, weights, args.width, args.height, args.steps, args.update_rate, args.pigment_strength)
    return cp.asnumpy(rgb)


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
    parser.add_argument(
        "--pigment-strength",
        type=float,
        default=DEFAULT_PIGMENT_STRENGTH,
        help="Optional analytic RGB pigment swirl strength. Keep 0 for learned flower training/generation.",
    )
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP model name used for ViT text embeddings.")
    parser.add_argument("--clip-device", default="auto", help="Device for CLIP: auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--weights", default=None, help="Optional .npz file with exported NCA weights.")
    parser.add_argument("--train", action="store_true", help="Train NCA weights on the local Flowers102 dataset before exiting.")
    parser.add_argument(
        "--train-method",
        choices=["backprop", "spsa"],
        default="backprop",
        help="Training optimizer: backprop uses explicit tensor math gradients; spsa uses finite differences.",
    )
    parser.add_argument("--train-class", default=None, help="Flowers102 class name or 1-based id. Defaults to the prompt.")
    parser.add_argument(
        "--train-split",
        choices=["train", "val", "test", "trainval", "all"],
        default="train",
        help="Flowers102 split used for training.",
    )
    parser.add_argument("--train-iters", type=int, default=200, help="Number of training iterations.")
    parser.add_argument("--train-samples", type=int, default=16, help="Maximum class images to keep in memory; 0 means all.")
    parser.add_argument(
        "--train-scope",
        choices=["output", "update", "all"],
        default="update",
        help="Which NCA weight arrays training should optimize.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.002, help="Training learning rate.")
    parser.add_argument("--spsa-eps", type=float, default=0.01, help="SPSA perturbation size.")
    parser.add_argument("--save-weights", default=str(DEFAULT_SAVE_WEIGHTS), help="Path for trained .npz NCA weights.")
    parser.add_argument("--log-every", type=int, default=10, help="Training log interval.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed for SPSA perturbations.")
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Flowers102 root beside this script, containing jpg/, imagelabels.mat, and setid.mat.",
    )
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
    if args.train_iters <= 0:
        raise ValueError("--train-iters must be positive.")
    if args.train_samples < 0:
        raise ValueError("--train-samples must be non-negative.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.spsa_eps <= 0.0:
        raise ValueError("--spsa-eps must be positive.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    args.dataset_root = resolve_flowers102_root(args.dataset_root)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        load_runtime_modules()
        if args.train:
            if args.train_method == "backprop":
                train_on_flowers102_backprop(args)
            else:
                train_on_flowers102(args)
        else:
            rgb = generate(args.prompt, args)
            display_or_save(rgb, args.prompt, Path(args.output) if args.output else None, args.show)
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    if args.train:
        print(f"Saved trained CuPy NCA weights to {args.save_weights}.")
    else:
        print(f"Generated {args.output} for prompt {args.prompt!r} using CLIP embeddings and CuPy NCA math.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
