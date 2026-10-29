# Third-party sources

Mega-CP includes adapted Hopper attention kernels and attention-state helpers.
Their original copyright notices are retained in the source files.

| Source | Revision / location | License text |
| --- | --- | --- |
| FlashAttention | `c75d019dea9d910312974417bc28f190dfdda6d9`; adapted files in `csrc/` and `include/` | [BSD-3-Clause](LICENSES/flash-attention.BSD-3-Clause.txt) |
| CUTLASS | `7127592069c2fe01b041e174ba4345ef9b279671`; pinned submodule and compatibility headers | [BSD-3-Clause](LICENSES/cutlass.BSD-3-Clause.txt) |
| ThunderKittens | Pinned `third_party/ThunderKittens` submodule; CUDA IPC support | [MIT](LICENSES/thunderkittens.MIT.txt) |
| vLLM | `a89015c6df8eeb37a843b717c97a5be1355de83d`; LSE correction and attention-state merge in `min_fa3_dcp.py` | [Apache-2.0](LICENSES/vllm.Apache-2.0.txt) |

The stream/event ordering and TP/DCP topology were cross-checked against SGLang
revision `8d6549bc4039d33635844495d86684677a4f0df8`. The core runner does not
require vLLM or SGLang at runtime.
