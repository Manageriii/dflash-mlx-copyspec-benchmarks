# Apple Silicon 2-Bit Ternary Speculative Decoding Enhancements (`dflash-mlx` Patch & Benchmarks)

**Phase-Aware `CopySpec` N-Gram Gating + EAGLE Draft Backend + L1 Prefix Cache Optimizations on 24 GB Apple Silicon (`M4 Pro`)**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

> **Upstream Attribution:** These patches, draft backends, and Apple Silicon `M4 Pro` (24 GB) benchmarks build directly upon **[NakliTechie/dflash-mlx-bonsai2](https://github.com/NakliTechie/dflash-mlx-bonsai2)** and **[bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx)** (Apache-2.0), targeting PrismML's 2-bit Hadamard-packed 27B models and `z-lab/DFlash` speculative decoding. Full upstream copyright notices are preserved in [NOTICE](NOTICE).

---

## What This Patch Set Adds

Running a 27B reasoning + coding model inside **24 GB unified memory** on an Apple M4 Pro exposes severe tension between `<think>` chain-of-thought phases (where N-gram prompt copy rates drop and prematurely disable speculative lookup) and structured code emission (where multi-order N-gram + block speculative verification shine).

1. **Phase-Aware `CopySpecAutoGate` (`copyspec.py`, `spec_epoch.py`):**
   * Prevents `<think>` tokens from triggering the 24-cycle dormant penalty window right before code emission begins.
   * Re-engages `CopySpecIndex` automatically upon `</think>` closure and enables high-confidence (`>=8`-gram) matches across reasoning traces, raising `copyspec_hits` per generation from `3` to `57–61`.
2. **EAGLE Feature-Fusion Draft Backend (`eagle_backend.py`, `eagle_drafter.py`):**
   * Adds lightweight EAGLE draft head integration alongside DFlash2 block verification in MLX.
3. **Effort-Scaled Reasoning Budget & Bounded L1 Prefix Cache (`full_upstream_diff.patch`):**
   * Keeps peak MLX active memory around `8.82 GB` (`10.17 GB` peak) across 128K-capable sessions on 24 GB Macs with zero Metal `kIOGPUCommandBufferCallbackErrorOutOfMemory` crashes.

---

## Measured Results on MacBook Pro 14" (`M4 Pro`, 12-Core CPU / 16-Core GPU, 24 GB RAM)

| Workload | Outcome | Median Speed (`tok/s`) | Notes |
| :--- | :---: | :---: | :--- |
| **Plain 2-Bit 27B Decode Baseline** | Pass | `~21.5 tok/s` | No drafter |
| **Upstream DFlash2 (Code / Chat Template)** | Pass | `27–33 tok/s` | `~4.1–4.6` tokens/cycle |
| **Canonical 3-Task Code & Game Suite (Patched)** | **3/3 PASS (100%)** | **`70.98 tok/s` median** | `59.7 – 76.3 tok/s`, `83.4–87.1%` accepted |
| **Thinking + Code (`medium` / `high` effort)** | **100% PASS** | **`65.7 / 60.3 tok/s`** | Up from `~38 tok/s` (`TPC = 5.93–6.39`) |
| **`novel-coding-v1` (12 Frozen Algorithmic Tasks)** | **12/12 PASS (100%)** | **`38.5 – 44.0 tok/s`** | `xhigh` reasoning + novel synthesis |

---

## Applying the Patch to Upstream `dflash-mlx-bonsai2`

```bash
git clone https://github.com/NakliTechie/dflash-mlx-bonsai2.git
cd dflash-mlx-bonsai2
git apply /path/to/full_upstream_diff.patch
cp /path/to/eagle_backend.py /path/to/eagle_drafter.py dflash_mlx/draft/
pytest
```
