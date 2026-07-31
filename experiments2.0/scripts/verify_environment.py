from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys


REQUIRED_DISTRIBUTIONS = (
    "groundingdino-py",
    "torch",
    "torchvision",
    "transformers",
    "numpy",
    "pandas",
    "scikit-learn",
    "pycocotools",
    "kagglehub",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow an environment without CUDA (not suitable for the full run)",
    )
    arguments = parser.parse_args()
    versions = {}
    missing = []
    for name in REQUIRED_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)

    import torch
    import groundingdino

    try:
        import groundingdino._C  # noqa: F401

        groundingdino_extension = True
    except ImportError:
        groundingdino_extension = False

    cuda_available = torch.cuda.is_available()
    inside_venv = sys.prefix != sys.base_prefix
    cuda_test = None
    compute_capability = None
    if cuda_available:
        device = torch.device("cuda:0")
        left = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        right = torch.eye(4, dtype=torch.float32, device=device)
        result = left @ right
        torch.cuda.synchronize(device)
        cuda_test = bool(torch.equal(result.cpu(), left.cpu()))
        compute_capability = list(torch.cuda.get_device_capability(device))

    from groundingdino.models.GroundingDINO.ms_deform_attn import (
        multi_scale_deformable_attn_pytorch,
    )

    operator_device = torch.device("cuda:0" if cuda_available else "cpu")
    operator_value = torch.arange(
        8, dtype=torch.float32, device=operator_device
    ).reshape(1, 4, 1, 2)
    operator_shapes = torch.tensor(
        [[2, 2]], dtype=torch.long, device=operator_device
    )
    operator_locations = torch.full(
        (1, 1, 1, 1, 1, 2),
        0.5,
        dtype=torch.float32,
        device=operator_device,
    )
    operator_weights = torch.ones(
        (1, 1, 1, 1, 1), dtype=torch.float32, device=operator_device
    )
    operator_output = multi_scale_deformable_attn_pytorch(
        operator_value,
        operator_shapes,
        operator_locations,
        operator_weights,
    )
    deformable_attention_test = (
        tuple(operator_output.shape) == (1, 1, 2)
        and bool(torch.isfinite(operator_output).all().item())
        and operator_output.device == operator_device
    )

    report = {
        "python": sys.version,
        "environment_type": "venv" if inside_venv else "system",
        "environment_prefix": sys.prefix,
        "base_interpreter_prefix": sys.base_prefix,
        "platform": platform.platform(),
        "packages": versions,
        "missing": missing,
        "cuda_available": cuda_available,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "compute_capability": compute_capability,
        "cuda_tensor_operation_passed": cuda_test,
        "groundingdino_import": bool(groundingdino),
        "groundingdino_compiled_extension": groundingdino_extension,
        "deformable_attention_backend": (
            "compiled_extension"
            if groundingdino_extension
            else "groundingdino-py_pytorch"
        ),
        "deformable_attention_device": str(operator_device),
        "deformable_attention_test_passed": deformable_attention_test,
    }
    print(json.dumps(report, indent=2))

    if missing:
        raise SystemExit(f"Missing required distributions: {missing}")
    if not inside_venv:
        raise SystemExit(
            "The verification command must run inside experiments2.0/.venv"
        )
    if versions["groundingdino-py"] != "0.4.0":
        raise SystemExit("RQ1 requires groundingdino-py==0.4.0")
    if not cuda_available and not arguments.allow_cpu:
        raise SystemExit("CUDA is not available; full RQ1 extraction would be impractical.")
    if cuda_available and not cuda_test:
        raise SystemExit("CUDA was detected but a tensor operation failed.")
    if not deformable_attention_test:
        raise SystemExit(
            "The GroundingDINO deformable-attention operation failed."
        )


if __name__ == "__main__":
    main()
