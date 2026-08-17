---
layout: post
title:  "Building CUDA graphs, not capturing"
date:   2026-08-17 01:27:19 -0700
author: Boyue Li
---

CUDA graph is a great feature for minimizing kernel launching overhead.
Kernel launching will be a bottleneck when kernel execution time is too short that the CPU does not have  enough time to launch the next kernel before the end of the current kernel.
Inference or small model training falls into this regime.
A CUDA graph is usually created by "capturing" all launched kernels within a range.
The captured graph can later be "replayed" in the captured order to replicate the whole execution process.

However, the CUDA graph is static because it will call the same op on the same sets of inputs when replayed.
This limitation caused some difficulties for downstream developers:

1. SGLang has to capture different graphs for different prefill token budgets [[doc]](https://docs.sglang.io/docs/advanced_features/piecewise_cuda_graph),
but change of input shape will trigger recompilation.
Graph capture and `torch.compile` can cost quite a while for a large model.

2. `torch.compile` supports a `reduce-overhead` mode in which PyTorch will run the optimized module with CUDA graph.
But "reduction of overhead can come at the cost of more memory usage, as we will cache the workspace memory required for the invocation" [[link]](https://docs.pytorch.org/docs/2.13/generated/torch.compile.html).

I think it's probably time to rebuild infra without PyTorch for maximum efficiency, at least for LLM labs running large scale training/inference jobs.
There are a few reasons:
1. Training/inference start up time can be much faster.
1. It’s easy to handle different input shapes: either build a new graph or just update kernel configurations.
We may need to recompile kernels for different token buckets, but it will definitely be faster than compiling all layers with `torch.compile`.
1. People often need to build special kernels for certain parts of the model to improve performance.
But porting these kernels to PyTorch is some unnecessary overhead.
1. It’s possible to achieve an extremely small memory footprint if we manage buffers manually.
This is definitely not a scalable solution, but I believe it’s worth investing for extremely large scales.

Next, I'll use two examples to explain how to build CUDA graph with custom kernels.

## Example 1: building a CUDA graph with Triton kernels

We first define a toy kernel that sets all elements of an input tensor to 0.
Please install this package before running the example.
Full code is available at [[code]](https://github.com/liboyue/cuda-graph-tools/blob/main/examples/toy_example.py).
[[Here]](https://github.com/liboyue/cuda-graph-tools/blob/main/examples/cute_dsl_example.py) is a slightly more complex example with CuTeDSL.


```python
import torch
import triton
import triton.language as tl

@triton.jit
def zero_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)

def fill_zeros(output):
    output = output.reshape(-1)
    n_elements = len(output)
    return zero_kernel[(triton.cdiv(n_elements, 1024),)](output, n_elements, BLOCK_SIZE=1024)

x = torch.randn(128, 32, device="cuda")
fill_zeros(x)
assert torch.abs(x).sum().item() == 0
```

Next, we will try to launch this kernel with CUDA driver API.
CUDA kernels are launched through CUDA driver or CUDA runtime APIs
[[doc]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15).
Users need to specify a pointer to the kernel, grid and block shapes,
the kernel’s parameters and a few extra parameters to launch a kernel.
Kernel parameters can be memory addresses, shapes and other configurations.

Triton strips `tl.constexpr` parameters from the generated kernel definition,
which makes it very easy for us to guess what arguments the kernel expects.
But it’s a bit tricky to find out how launching parameters are decided because it’s buried in Triton’s repo.
But we surely can hack it.
NVIDIA CUDA Profiling Tools Interface (CUPTI) provides a way to intercept CUDA driver calls [[code]](https://github.com/liboyue/cuda-graph-tools/blob/main/cpp/cupti_hook.cpp).
So we can extract these parameters when a Triton kernel is launched.

```python
from cuda.bindings import driver as cuda
from cuda_graph_experiments.kernels.utils import checkCudaErrors, cupti_hook

with cupti_hook() as hook:
    fill_zeros(x)
    launch_args = hook.get_args()

c_arg_1 = ctypes.c_void_p(x.data_ptr())
c_arg_2 = ctypes.c_int32(x.numel())
global_scratch = torch.zeros(0, device="cuda")
c_arg_3 = ctypes.c_void_p(global_scratch.data_ptr())
profile_scratch = torch.zeros(0, device="cuda")
c_arg_4 = ctypes.c_void_p(profile_scratch.data_ptr())

kernel_params = (ctypes.c_void_p * 4)(
    *[ctypes.c_void_p(ctypes.addressof(c_arg)) for c_arg in [c_arg_1, c_arg_2, c_arg_3, c_arg_4]]
)

x[:] = torch.randn_like(x)
assert torch.abs(x).sum().item() != 0
checkCudaErrors(cuda.cuLaunchKernel(*launch_args, 0, kernel_params, 0))
assert torch.abs(x).sum().item() == 0
```

We can build a CUDA graph once we confirm launching kernels with driver API can work.
We need to pack all parameters to a `cuda.CUDA_KERNEL_NODE_PARAMS()` object and make sure all parameter objects are alive when running a CUDA graph.
```python
node_params = cuda.CUDA_KERNEL_NODE_PARAMS()
node_params.func = launch_args[0]
node_params.gridDimX = launch_args[1]
node_params.gridDimY = launch_args[2]
node_params.gridDimZ = launch_args[3]
node_params.blockDimX = launch_args[4]
node_params.blockDimY = launch_args[5]
node_params.blockDimZ = launch_args[6]
node_params.sharedMemBytes = launch_args[7]
node_params.kernelParams = kernel_params
node_params.extra = 0

graph = checkCudaErrors(cuda.cuGraphCreate(0))
checkCudaErrors(cuda.cuGraphAddKernelNode(graph, [], 0, node_params))
graph_exec = checkCudaErrors(cuda.cuGraphInstantiate(graph, 0))
stream = checkCudaErrors(cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_DEFAULT))

x[:] = torch.randn_like(x)
assert torch.abs(x).sum().item() != 0
checkCudaErrors(cuda.cuGraphLaunch(graph_exec, stream))
assert torch.abs(x).sum().item() == 0
```

## Example 2: building forward and backward CUDA graphs for model training
[[This file]](https://github.com/liboyue/cuda-graph-tools/blob/main/cuda_graph_tools/layers/transformer.py#L1548) contains a working LLM implementation built with Triton kernels and CUDA graph.
It is designed to:
1. Fuse kernels: Fuse QK norm with RoPE, residual add with matmul.
1. Reuse buffers aggressively: manually allocate buffers, reuse when possible to reduce memory footprint.
1. Recompute activations during backward to reduce memory usage for larger batch size.
1. Provide some observability by CUDA events.

In this way, 
infra teams can easily introduce new kernels or tune activation checkpointing plans without worrying about breaking PyTorch Autograd or `torch.compile`.
And the improvements can directly translate to iteration speed:
faster pre-training may save a few days of training time,
faster inference may mean RL can train on even longer sequence length for long-horizon tasks.

However, it's difficult to use other open-source kernel libraries (e.g. FlashAttention) as we may not have a way to create kernel parameters.
Ideally, these libraries can all implement a function to create kernel launching parameters to enable directly calling their kernels.
Another possibility is to fully migrate to one DSL so that it would be easier to build customized kernels,
e.g. [[TileLang kernels]](https://github.com/deepseek-ai/TileKernels) from DeepSeek.
Then, building CUDA graph with kernels would remain a good option for performance.


#### Note: visualizing CUDA graphs
We can visualize CUDA graphs by plotting the kernel's names and dependencies.
This is a single-layer LLM's forward and backward graph.

<img src="/assets/fwd_bwd.svg" width="300rem">
