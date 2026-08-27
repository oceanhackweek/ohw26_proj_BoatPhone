"""UNUSED -- see decision 0012. B0 returned NO-GO on the whole pretrained-checkpoint path
(this module's own OOM finding was one of three independent reasons, and not even the
decisive one: an independent ~12-CPU-day-per-corpus-pass compute cost is fatal on its own,
and the released checkpoint has no Engine Noise output to score in the first place). Kept
unimported and undeleted on purpose -- its docstring below is the best surviving evidence
for the corrected OOM finding (a 3.90 GB cgroup v2 cap on this container, not the host's
30 GB, misdiagnosed at the time as a possible architectural wall). See
`docs/decisions/0012-b0-model-viability-outcome.md`.

Make the ONC SSAMBA / Vision-Mamba checkpoint importable and runnable on CPU.

WHY THIS EXISTS (and why it is a library module, not a notebook cell)
--------------------------------------------------------------------
`external/onc_ssamba/onc_ssamba/models/models_mamba.py` imports `mamba_ssm`
*unconditionally at module import time*, and `mamba_ssm==2.2.5` in turn does an
UNGUARDED `import selective_scan_cuda` in
`mamba_ssm/ops/selective_scan_interface.py`. `selective_scan_cuda` is a compiled
CUDA extension. On a CPU-only host it cannot be built (no nvcc, no GPU), so the
repo's model package cannot even be imported, let alone run.

This module installs a MINIMAL, LOUD, FULLY-DOCUMENTED CPU path:

  1. A stub `selective_scan_cuda` whose every entry point RAISES. It exists only
     to satisfy the unguarded import. It never returns a number, so nothing can
     silently fall through to a wrong answer (CLAUDE.md invariant 5).
  2. `mamba_ssm`'s OWN pure-PyTorch reference implementation
     (`selective_scan_ref`) is substituted for `selective_scan_fn`. This is not
     an invention of ours: it is the reference kernel the CUDA kernel is
     validated against, shipped in the same file.
  3. Every `Mamba` mixer is forced onto the slow path (`use_fast_path = False`),
     because the fast path calls `mamba_inner_fn`, which is CUDA-only.
  4. `causal_conv1d` is absent, so `mamba_ssm` already falls back to
     `torch.nn.functional.conv1d` on its own. We assert that fallback is what is
     in force rather than assuming it.

CONSEQUENCE TO STATE WHEREVER A NUMBER FROM THIS PATH APPEARS: results produced
through this shim come from the reference selective scan, not the fused CUDA
kernel. They are numerically equivalent up to floating-point associativity, and
they are MUCH slower -- see the measured seconds/sample recorded by B0-6.

NOT PART OF THE ENVIRONMENT. `timm`, `einops`, `mamba_ssm`, `triton` and
`transformers` are NOT installed in the shared OHW hub environment and were NOT
installed into it. They must be present on `sys.path` (e.g. a `pip install
--no-deps --target <dir>` directory placed on PYTHONPATH). `prepare_cpu_mamba()`
raises with that instruction if they are missing.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import types

# Third-party packages the ONC model path needs that the hub environment does
# NOT provide. Source: measured 2026-08-27 against /home/.pixi/envs/default
# (B0-5 dependency audit), cross-checked with external/onc_ssamba/requirements.txt.
MISSING_FROM_HUB_ENV: tuple[str, ...] = ("timm", "einops", "mamba_ssm", "triton", "transformers")

# Name of the CUDA extension mamba_ssm 2.2.5 imports unguarded. Source:
# mamba_ssm/ops/selective_scan_interface.py line 20 of the 2.2.5 sdist.
CUDA_EXT_MODULE_NAME: str = "selective_scan_cuda"

# Symbols mamba_ssm/utils/generation.py imports from `transformers.generation`.
# They were REMOVED from transformers >= 5. They are used only by
# MambaLMHeadModel, a text-generation path VisionMamba never touches, so
# aliasing them to their modern equivalent cannot affect a spectrogram forward
# pass. Source: transformers 5.16.1 exports GenerateDecoderOnlyOutput only.
_REMOVED_TRANSFORMERS_GENERATION_ALIASES: dict[str, str] = {
    "GreedySearchDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
    "SampleDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
}


def _install_cuda_stub() -> None:
    """Register a `selective_scan_cuda` whose every function raises."""
    if CUDA_EXT_MODULE_NAME in sys.modules:
        return
    stub = types.ModuleType(CUDA_EXT_MODULE_NAME)
    stub.__doc__ = "CPU stub for mamba_ssm's unguarded CUDA import. Every entry point raises."

    def _no_cuda(*_args, **_kwargs):
        raise RuntimeError(
            "selective_scan_cuda is a CPU stub installed by boatphone.onc_model_cpu; "
            "the caller must be routed to mamba_ssm's selective_scan_ref instead"
        )

    stub.fwd = _no_cuda
    stub.bwd = _no_cuda
    sys.modules[CUDA_EXT_MODULE_NAME] = stub


def _alias_removed_transformers_symbols() -> None:
    """Restore names mamba_ssm imports that transformers >= 5 no longer exports."""
    gen = importlib.import_module("transformers.generation")
    for old_name, new_name in _REMOVED_TRANSFORMERS_GENERATION_ALIASES.items():
        if not hasattr(gen, old_name):
            setattr(gen, old_name, getattr(gen, new_name))


def prepare_cpu_mamba() -> dict[str, str]:
    """Make `mamba_ssm` importable on CPU and force its pure-PyTorch kernel.

    Returns a provenance dict naming what was substituted, so a caller can print
    it next to any number it produces. Raises ImportError, naming the package, if
    a required third-party module is absent -- it never degrades quietly.
    """
    for name in MISSING_FROM_HUB_ENV:
        if name == "mamba_ssm":
            continue  # imported below, after the stub is in place
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise ImportError(
                f"'{name}' is required by the ONC SSAMBA model path and is NOT in the shared "
                "OHW hub environment. Install it OUT OF TREE and put it on PYTHONPATH, e.g.\n"
                f"    python3 -m pip install --no-deps --target <dir> {name}\n"
                f"    export PYTHONPATH=<dir>\n"
                "Do not install it into /home/.pixi/envs/default; it would not persist for "
                "teammates and would make the analysis unreproducible."
            ) from exc

    _install_cuda_stub()
    _alias_removed_transformers_symbols()

    interface = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
    mamba_simple = importlib.import_module("mamba_ssm.modules.mamba_simple")

    if mamba_simple.causal_conv1d_fn is not None:
        raise RuntimeError(
            "causal_conv1d is present; this shim was written for the CPU case where it is "
            "absent and mamba_ssm falls back to F.conv1d. Re-verify before trusting output."
        )

    # Substitute mamba_ssm's own reference selective scan for the CUDA one.
    mamba_simple.selective_scan_fn = interface.selective_scan_ref

    return {
        "selective_scan": "mamba_ssm.ops.selective_scan_interface.selective_scan_ref (pure PyTorch)",
        "causal_conv1d": "absent -- mamba_ssm falls back to torch.nn.functional.conv1d",
        "cuda_extension": f"{CUDA_EXT_MODULE_NAME} stubbed; all entry points raise",
        "mamba_ssm_version": getattr(importlib.import_module("mamba_ssm"), "__version__", "unknown"),
    }


def force_slow_path(model) -> int:
    """Set `use_fast_path = False` on every Mamba mixer. Returns how many were set.

    The fast path calls `mamba_inner_fn`, which is CUDA-only. Raises if it finds
    no mixers at all, rather than silently returning 0 and leaving a CUDA call
    live inside the forward pass.
    """
    n = 0
    for module in model.modules():
        if hasattr(module, "use_fast_path"):
            module.use_fast_path = False
            n += 1
    if n == 0:
        raise RuntimeError("no module with `use_fast_path` found -- is this really a Mamba model?")
    return n


def add_onc_repo_to_path(onc_model_dir: pathlib.Path) -> None:
    """Put the ONC repo clone on sys.path so `import onc_ssamba` resolves to it."""
    p = str(pathlib.Path(onc_model_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


__all__ = [
    "MISSING_FROM_HUB_ENV",
    "CUDA_EXT_MODULE_NAME",
    "prepare_cpu_mamba",
    "force_slow_path",
    "add_onc_repo_to_path",
]
