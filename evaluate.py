import os
import hydra
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from omegaconf import DictConfig, OmegaConf

from src.te_versions.model_te import GPT
from src.evaluator import evaluate_hella_swag


def maybe_init_dist(device: str):
    if dist.is_initialized():
        return

    backend = "nccl"
    store = dist.HashStore()
    dist.init_process_group(backend=backend, store=store, rank=0, world_size=1)


def build_model(cfg: DictConfig, device: torch.device):
    model = GPT(
        n_embd=cfg.model.n_embd,
        vocab_size=cfg.model.vocab_size,
        block_size=cfg.model.block_size,
        n_heads=cfg.model.n_heads,
        head_size=cfg.model.head_size,
        rope_head_size=cfg.model.rope_head_size,
        kv_latent_size=cfg.model.kv_latent_size,
        q_latent_size=cfg.model.q_latent_size,
        n_layers=cfg.model.n_layers,
        n_shared_experts=cfg.model.n_shared_experts,
        n_routed_experts=cfg.model.n_routed_experts,
        topk_experts=cfg.model.topk_experts,
        expert_hidden_size=cfg.model.expert_hidden_size,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    return model


def _load_state(model, state_dict: dict):
    model.load_state_dict(state_dict, strict=False)


def load_checkpoint_any_format(model, checkpoint_path: str, device: torch.device):
    if os.path.isdir(checkpoint_path) and os.path.exists(os.path.join(checkpoint_path, ".metadata")):
        # Load raw tensors first, then apply to module. This avoids pickling
        # issues with compiled/module code objects during DCP planning.
        state = {"model": model.state_dict()}
        dcp.load(
            state_dict=state,
            storage_reader=dcp.FileSystemReader(checkpoint_path),
            no_dist=True,
        )
        _load_state(model, state["model"])
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Common training checkpoint layout
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        _load_state(model, checkpoint["model_state_dict"])
        return

    # Common converted DCP layout: top-level container with model/optimizer/step
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_payload = checkpoint["model"]

        # Some conversions keep model state directly under `model`.
        if isinstance(model_payload, dict) and "model_state_dict" in model_payload:
            model_payload = model_payload["model_state_dict"]

        if isinstance(model_payload, dict):
            _load_state(model, model_payload)
            return

    if isinstance(checkpoint, dict):
        _load_state(model, checkpoint)
        return

    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(version_base=None, config_name="config_eval", config_path="config")
def main(cfg: DictConfig):
    print("=" * 40)
    print("Evaluation Configuration")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 40)

    device = torch.device(cfg.eval.device)

    # Setup optimizations
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    maybe_init_dist(device.type)

    print(f"Building model...")
    model = build_model(cfg, device).eval()

    print(f"Loading checkpoint from: {cfg.eval.checkpoint}")
    load_checkpoint_any_format(model, cfg.eval.checkpoint, device)

    print("Evaluating on HellaSwag...")
    # Evaluate HellaSwag using evaluator
    acc = evaluate_hella_swag(model, device)

    print(f"\n---> HellaSwag Accuracy: {acc * 100:.2f}%\n")

    cleanup_dist()


if __name__ == "__main__":
    main()
