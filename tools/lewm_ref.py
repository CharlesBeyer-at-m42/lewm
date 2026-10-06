"""Shared helpers for the LeWM Python tools.

Run with the reference implementation's interpreter (``../lpworldmodel/.venv/bin/python``) so
its ``models`` package is importable. Vocabulary follows ``../../CONTEXT.md``:

- *model file*: GGUF holding LeWM weights + hyperparameters (``lewm.*`` metadata)
- *reference dump*: GGUF holding one fixed input and the reference implementation's outputs
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REFERENCE_ROOT = HERE.parent.parent / "lpworldmodel"
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

import gguf  # noqa: E402

ARCH = "lewm"
COMPONENTS = ("encoder", "action_encoder", "predictor")


@dataclass
class Hparams:
    """Everything needed to rebuild the LpWM path. Mirrors the ``lewm.*`` metadata keys."""

    image_size: int
    patch_size: int
    dim: int
    encoder_depth: int
    encoder_heads: int
    encoder_dim_head: int
    encoder_mlp_dim: int
    encoder_feature: str  # "cls" | "patch"
    encoder_proj_hidden: int
    code_dim: int  # projector output dim; the width of a code
    action_dim: int  # chunked action width
    action_smoothed_dim: int
    action_mlp_hidden: int
    predictor_depth: int
    predictor_heads: int
    predictor_dim_head: int
    predictor_mlp_dim: int
    predictor_num_frames: int  # == history window
    predictor_num_patches: int
    predictor_proj_hidden: int
    link: str  # "identity" | "relu" | "reprelu"

    def to_gguf(self, w: gguf.GGUFWriter) -> None:
        for k, v in asdict(self).items():
            key = f"{ARCH}.{k}"
            if isinstance(v, str):
                w.add_string(key, v)
            else:
                w.add_uint32(key, int(v))

    @classmethod
    def from_gguf(cls, r: gguf.GGUFReader) -> "Hparams":
        kw = {}
        for f in cls.__dataclass_fields__.values():
            field = r.fields[f"{ARCH}.{f.name}"]
            raw = field.parts[field.data[0]]
            if f.type == "str":
                kw[f.name] = bytes(raw).decode("utf-8")
            else:
                kw[f.name] = int(raw[0])
        return cls(**kw)


def hparams_from_modules(encoder, action_encoder, predictor, link) -> Hparams:
    """Derive hyperparameters from the live modules, so checkpoint and random-init exports share
    one code path and nothing is read from a config that might not match the weights."""
    enc_block = encoder.transformer.layers[0]
    pred_block = predictor.transformer.layers[0]
    enc_inner = enc_block.attn.to_qkv.out_features // 3
    pred_inner = pred_block.attn.to_qkv.out_features // 3
    return Hparams(
        image_size=encoder.grid_size * encoder.patch_size,
        patch_size=encoder.patch_size,
        dim=encoder.dim,
        encoder_depth=len(encoder.transformer.layers),
        encoder_heads=enc_block.attn.heads,
        encoder_dim_head=enc_inner // enc_block.attn.heads,
        encoder_mlp_dim=enc_block.mlp.net[1].out_features,
        encoder_feature=encoder.feature,
        encoder_proj_hidden=encoder.projector.net[0].out_features,
        code_dim=encoder.projector.net[3].out_features,
        action_dim=action_encoder.patch_embed.in_channels,
        action_smoothed_dim=action_encoder.patch_embed.out_channels,
        action_mlp_hidden=action_encoder.embed[0].out_features,
        predictor_depth=len(predictor.transformer.layers),
        predictor_heads=pred_block.attn.heads,
        predictor_dim_head=pred_inner // pred_block.attn.heads,
        predictor_mlp_dim=pred_block.mlp.net[1].out_features,
        predictor_num_frames=predictor.temporal_pos.shape[1],
        predictor_num_patches=predictor.num_patches,
        predictor_proj_hidden=predictor.output_proj.net[0].out_features,
        link=link.kind if link is not None else "identity",
    )


def build_modules(hp: Hparams):
    """Instantiate the reference modules for a set of hyperparameters (random init)."""
    from models.infojepa_modules import ARPredictor, Embedder, Link
    from models.vit_encoder import ViTEncoder

    encoder = ViTEncoder(
        image_size=hp.image_size,
        patch_size=hp.patch_size,
        dim=hp.dim,
        depth=hp.encoder_depth,
        heads=hp.encoder_heads,
        mlp_dim=hp.encoder_mlp_dim,
        dim_head=hp.encoder_dim_head,
        feature=hp.encoder_feature,
        proj_hidden=hp.encoder_proj_hidden,
        proj_dim=hp.code_dim,
    )
    assert hp.action_mlp_hidden % hp.code_dim == 0
    action_encoder = Embedder(
        in_chans=hp.action_dim,
        emb_dim=hp.code_dim,
        smoothed_dim=hp.action_smoothed_dim,
        mlp_scale=hp.action_mlp_hidden // hp.code_dim,
    )
    predictor = ARPredictor(
        num_frames=hp.predictor_num_frames,
        depth=hp.predictor_depth,
        heads=hp.predictor_heads,
        mlp_dim=hp.predictor_mlp_dim,
        input_dim=hp.code_dim,
        hidden_dim=hp.code_dim,
        output_dim=hp.code_dim,
        num_patches=hp.predictor_num_patches,
        dim_head=hp.predictor_dim_head,
        pred_proj_hidden=hp.predictor_proj_hidden,
    )
    link = Link(hp.link)
    return encoder, action_encoder, predictor, link


def load_checkpoint_modules(path: Path):
    """Unpickle a reference checkpoint (whole modules) and return the LpWM-path components."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    encoder = payload["encoder"]
    predictor = payload["predictor"]
    action_encoder = payload["action_encoder"]
    link = payload.get("link")
    # accelerate may have wrapped modules; unwrap defensively
    for name in ("encoder", "predictor", "action_encoder"):
        m = locals()[name]
        if hasattr(m, "module"):
            raise RuntimeError(f"{name} is wrapped ({type(m).__name__}); unwrap before export")
    return encoder, action_encoder, predictor, link, payload.get("epoch")


def mup_metadata(encoder, action_encoder, predictor, base_width: int) -> dict:
    """Per-tensor muP learning-rate multipliers and weight-decay flags, mirroring
    models/mup.py::mup_param_groups: matrix-like (Linear/Conv weights) get base_width / fan_in,
    everything else 1.0; weight decay only on tensors with dim >= 2."""
    from models.mup import mup_fan_in_map

    meta: dict = {"lewm.mup.base_width": int(base_width)}
    for prefix, module in zip(COMPONENTS, (encoder, action_encoder, predictor)):
        fmap = mup_fan_in_map(module)
        for name, p in module.named_parameters():
            mult = float(base_width) / float(fmap[name]) if name in fmap else 1.0
            meta[f"lewm.mup.lr_mult.{prefix}.{name}"] = mult
            meta[f"lewm.mup.wd.{prefix}.{name}"] = bool(p.dim() >= 2)
    return meta


def state_tensors(encoder, action_encoder, predictor):
    """Yield (name, float32 numpy) for every parameter, names mirroring the reference."""
    for prefix, module in zip(COMPONENTS, (encoder, action_encoder, predictor)):
        for name, t in module.state_dict().items():
            yield f"{prefix}.{name}", t.detach().cpu().float().contiguous().numpy()


def write_gguf(path: Path, hp: Hparams | None, tensors, extra_meta: dict | None = None) -> None:
    w = gguf.GGUFWriter(str(path), ARCH)
    if hp is not None:
        hp.to_gguf(w)
    for k, v in (extra_meta or {}).items():
        if isinstance(v, str):
            w.add_string(k, v)
        elif isinstance(v, bool):
            w.add_bool(k, v)
        elif isinstance(v, int):
            w.add_uint32(k, v)
        elif isinstance(v, float):
            w.add_float32(k, v)
        else:
            raise TypeError(f"unsupported metadata type for {k}: {type(v)}")
    for name, arr in tensors:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def read_gguf_tensors(path: Path) -> tuple[gguf.GGUFReader, dict[str, np.ndarray]]:
    r = gguf.GGUFReader(str(path))
    out = {}
    for t in r.tensors:
        # ReaderTensor.data already has the original (numpy-order) shape
        out[t.name] = np.array(t.data, dtype=np.float32, copy=True)
    return r, out


def load_model_file(path: Path):
    """Rebuild the reference modules from a model file and load its weights, strictly."""
    r, tensors = read_gguf_tensors(path)
    hp = Hparams.from_gguf(r)
    encoder, action_encoder, predictor, link = build_modules(hp)
    for prefix, module in zip(COMPONENTS, (encoder, action_encoder, predictor)):
        sd = {}
        for name, arr in tensors.items():
            if name.startswith(prefix + "."):
                sd[name[len(prefix) + 1 :]] = torch.from_numpy(arr)
        # Files written by lewm-train carry ggml's dimension count, which drops leading singleton
        # dims (cls_token (1,1,384) arrives as (384,)); restore the module's shape.
        for name, t in module.state_dict().items():
            if name in sd and sd[name].shape != t.shape and sd[name].numel() == t.numel():
                sd[name] = sd[name].reshape(t.shape)
        missing, unexpected = module.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"{prefix}: missing={missing} unexpected={unexpected}")
    for m in (encoder, action_encoder, predictor, link):
        m.eval()
    return hp, encoder, action_encoder, predictor, link
