# Atomic Operation
A CUDA kernel can perform an atomic add operation on a memory location through a function call:

```int  atomicAdd(int* addr, int val)```