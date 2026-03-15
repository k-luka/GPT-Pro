import inspect
import os

import hydra
from omegaconf import DictConfig, OmegaConf


def _call_dcp_to_torch_save(func, src_dir: str, dst_file: str):
    # Try the common positional signature first.
    try:
        return func(src_dir, dst_file)
    except TypeError:
        pass

    # Fallback for possible keyword-only variations across torch versions.
    sig = inspect.signature(func)
    kwargs = {}
    for name in sig.parameters:
        lname = name.lower()
        if "dcp" in lname or "checkpoint" in lname or "source" in lname:
            kwargs[name] = src_dir
        elif (
            "torch" in lname or "save" in lname or "output" in lname or "dest" in lname
        ):
            kwargs[name] = dst_file

    if not kwargs:
        raise RuntimeError(
            "Could not infer dcp_to_torch_save signature for this torch version"
        )

    return func(**kwargs)


@hydra.main(
    version_base=None, config_name="config_convert_checkpoint", config_path="config"
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    src_dir = cfg.convert.input_checkpoint_dir
    dst_file = cfg.convert.output_pt_file

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Input checkpoint dir not found: {src_dir}")
    if not os.path.exists(os.path.join(src_dir, ".metadata")):
        raise FileNotFoundError(
            f"Input checkpoint dir does not look like DCP (missing .metadata): {src_dir}"
        )

    out_dir = os.path.dirname(dst_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    except Exception as exc:
        raise RuntimeError(
            "This torch build does not expose torch.distributed.checkpoint.format_utils.dcp_to_torch_save. "
            "Please upgrade torch to a version that includes this utility."
        ) from exc

    _call_dcp_to_torch_save(dcp_to_torch_save, src_dir, dst_file)

    print(f"Converted DCP checkpoint -> {dst_file}")
    print("Set inference.checkpoint to this .pt file for single-GPU inference.")


if __name__ == "__main__":
    main()
