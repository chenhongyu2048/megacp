# Mega-CP

Context-parallel attention for NVIDIA Hopper GPUs.

- **MegaRing** implements forward and backward attention with variable-length
  sequences and hierarchical ring topologies.
- **MegaDCP** implements head-sharded inference with dedicated communication
  CTAs, bandwidth-aware split planning, history reduction, output pull and
  final combine in a persistent kernel.

The repository contains the core library, correctness tests and two small
benchmarks. The native backend supports BF16 attention with head dimension 128
on a single node of 2, 4 or 8 Hopper GPUs. MegaDCP supports 4 or 8 local query
heads and DCP group sizes of 2, 4 or 8.

## Install

Requirements: Linux x86_64, Python 3.12, a CUDA 12.x toolkit, a linkable CUDA
driver library, and a C++20 compiler. The Python environment pins PyTorch
2.11.0 with CUDA 12.8 and Triton 3.6.0.

```bash
git clone https://github.com/chenhongyu2048/megacp.git
cd megacp
git submodule update --init --recursive
uv sync --frozen --no-install-project
make PYTHON=.venv/bin/python
```

Only CUTLASS and ThunderKittens are required as submodules. `CUDA_HOME` selects
the toolkit, and `CUTLASS_DIR` can point to an external CUTLASS checkout.
Build the native extensions in this checkout before running the library.

## Python entry points

```python
from megacp import (
    DCPMegaAttentionRunner,
    TKParallelTensor,
    backward_varlen_mega_ring,
    forward_varlen_mega_ring,
)
```

MegaDCP runners own preallocated workspaces and IPC buffers. Initialize NCCL
process groups, construct a runner with the local query-head and batch
capacities, and call `forward_chunk_prefill_varlen`. Use
`capture_last_forward` for fixed-shape CUDA Graph replay, or
`prepare_graph_forward` with `forward_prepared_graph` for changing batches in
an externally captured serving graph. A runner must be used serially and
closed after use.

MegaRing receives explicit IPC K/V arenas and per-sequence ring metadata.
The benchmark programs demonstrate buffer allocation and workspace reuse;
the hierarchical tests demonstrate mixed ring sizes. The `min_fa3_op` and
`min_fa3_dcp` modules expose the underlying FlashAttention backend and runner
interfaces.

## Correctness

Run the host-side planner and metadata tests without GPUs:

```bash
python -m unittest tests.test_metadata tests.test_planner \
  tests.test_packed_metadata tests.test_cpu_prepare
```

After the native build, the same command also exercises the C++ planner and
queue builder. Native tests are skipped if the extension has not been built.

Four-GPU MegaDCP output, LSE, replay and serving-graph checks:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  --module tests.test_dcp_distributed \
  --dcp-size 4 --hq-local 4 --num-splits 0 --block-n 176 \
  --q-lengths 1,17,33 --history-lengths 515,1030,2061 --repeat 2

CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  --module tests.test_dcp_graph --pipeline
```

Four-GPU MegaRing forward and backward checks:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  --module tests.test_ring_forward --seqlen 256 --num-comp-sm 64 --num-comm-sm 8

CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  --module tests.test_ring_backward --seqlen 256 --num-comp-sm 64 --num-comm-sm 8
```

`tests/` also covers local FlashAttention, KV-cache attention, remote loads,
hierarchical ring execution and backward binding validation.

## Minimal benchmarks

MegaDCP with automatic split selection:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  --module benchmarks.benchmark_dcp --dcp-size 4 --mode eager

CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  --module benchmarks.benchmark_dcp --dcp-size 4 --mode graph
```

Eager mode reports the persistent kernel's internal CUDA-event time; queue
reset and launch barriers are outside that interval. Graph mode reports the
complete captured replay, including its resets and synchronization.

Causal MegaRing forward or backward:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  --module benchmarks.benchmark_ring --sequence-length 2048 --direction forward

CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  --module benchmarks.benchmark_ring --sequence-length 2048 --direction backward
```

MegaRing uses preallocated workspaces and CUDA events around each call. Both
benchmarks emit JSON with the median and p90 of the per-sample maximum latency
across ranks. Run `--help` for shape, communication-SM and sampling options.

## Source attribution

The backend includes adapted FlashAttention Hopper sources. Their copyright
notices are retained. See [third-party attribution](THIRD_PARTY_NOTICES.md)
and `LICENSES/` for the upstream license texts.
