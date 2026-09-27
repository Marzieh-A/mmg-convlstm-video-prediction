"""Stage 0 - Environment audit and device management."""

from __future__ import annotations

import platform
import sys

import torch


def _gb(n_bytes):
    if n_bytes is None:
        return "unknown"
    return "{:.2f} GB".format(n_bytes / (1024 ** 3))


def get_device():
    """Return the preferred compute device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_environment_report():
    """Collect and print a full environment report. Returns the report dict."""
    report = {}

    report["python_version"] = sys.version.split()[0]
    report["platform"] = platform.platform()
    report["torch_version"] = torch.__version__

    try:
        import torchvision
        report["torchvision_version"] = torchvision.__version__
    except Exception as e:
        report["torchvision_version"] = "IMPORT FAILED: {}".format(e)

    cuda_ok = torch.cuda.is_available()
    report["cuda_available"] = cuda_ok

    if cuda_ok:
        report["cuda_version"] = torch.version.cuda
        report["cudnn_version"] = torch.backends.cudnn.version()
        report["gpu_count"] = torch.cuda.device_count()
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        report["gpu_name"] = props.name
        report["total_memory"] = _gb(props.total_memory)
        free, total = torch.cuda.mem_get_info(idx)
        report["free_memory_now"] = _gb(free)
        report["cudnn_enabled"] = torch.backends.cudnn.enabled
    else:
        report["cuda_version"] = torch.version.cuda
        report["gpu_name"] = None

    for mod_name in ("numpy", "yaml", "matplotlib", "pytest"):
        try:
            mod = __import__(mod_name)
            report[mod_name + "_version"] = getattr(mod, "__version__", "ok")
        except Exception as e:
            report[mod_name + "_version"] = "MISSING: {}".format(e)

    if cuda_ok:
        try:
            x = torch.randn(64, 64, device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            if torch.isfinite(y).all().item():
                report["cuda_compute_smoke_test"] = "PASS"
            else:
                report["cuda_compute_smoke_test"] = "FAIL (non-finite)"
        except Exception as e:
            report["cuda_compute_smoke_test"] = "FAIL: {}".format(e)
    else:
        report["cuda_compute_smoke_test"] = "SKIPPED (no CUDA)"

    lines = ["=" * 60, "STAGE 0 - ENVIRONMENT AUDIT REPORT", "=" * 60]
    for k in report:
        lines.append("{:>26}: {}".format(k, report[k]))
    lines.append("=" * 60)
    ok = cuda_ok and report.get("cuda_compute_smoke_test") == "PASS"
    if ok:
        lines.append("VERDICT: ENVIRONMENT OK (CUDA available)")
    else:
        lines.append("VERDICT: ENVIRONMENT BROKEN OR CPU-ONLY - DO NOT PROCEED")
    print("\n".join(lines))

    return report


if __name__ == "__main__":
    r = print_environment_report()
    if not (torch.cuda.is_available() and r.get("cuda_compute_smoke_test") == "PASS"):
        sys.exit(1)
