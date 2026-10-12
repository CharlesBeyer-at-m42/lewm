---
status: accepted
date: 2026-09-23
---

# LeWM is a standalone C++20 project that depends on ggml as a library

We are rewriting the LpWM world model in native code. LeWM lives in its own repository with ggml as a pinned git submodule, is written in conservative C++20 (no modules, no coroutines), and reads llama.cpp only as example code for ggml usage.

## Considered options

- **Build inside llama.cpp** as an example or tool, reusing its loader and backend scheduler. Rejected: its model loading and graph building are shaped around large language models, and a world model with image encoder, action conditioning, and autoregressive rollout fits none of it.
- **Fork llm.c** and swap the transformer for the world model. Rejected: its hand-written CUDA kernels are educational but every new operator is a kernel from scratch, and there is no backend abstraction.
- **Pure C11** like ggml itself. Rejected: GGUF loading, graph construction, and resource management are noticeably more verbose in C for no gain, and the closest reference code (llama.cpp) is C++.
- **Link against the sibling ggml checkout by relative path** or **vendor a copy**. Rejected in favour of a submodule so ggml is bumped deliberately, as llama.cpp and whisper.cpp do.
