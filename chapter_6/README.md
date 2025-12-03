## Some important points
### Constant memory
To use constant memory, the host code needs to allocate and copy constant memory variables in a different way than global memory variables. most of time we define global var such:
```
__constant__ float F[KERNEL_SIZE * KERNEL_SIZE];
```

The memory transfer from host to device in constant memory is follow:
```
cudaMemcpyToSymbol(void* devPtr, void* srcPtr, size_t sizeInBytes);
```

### Pinned Memory
**Pageability:**

By default, memory allocated on the host (e.g., using malloc in C++) is "pageable." This means the operating system can move these memory pages between physical RAM and disk (virtual memory) as needed. 

**GPU Access:**

The GPU cannot directly access pageable host memory. When a data transfer from pageable host memory to device memory is initiated, the CUDA driver first performs an implicit copy of the data to a temporary, internal, page-locked (pinned) host buffer. Then, the data is transferred from this pinned buffer to the GPU's device memory. 

**Pinned(Page-Locked) Allocation:**
When you allocate pinned memory using functions like cudaMallocHost() or cudaHostAlloc(), you are explicitly telling the operating system to keep these memory pages in physical RAM and prevent them from being swapped out.

**Direct Transfer (Pinned Memory):**

If the memory is already pinned using ```cudaMallocHost(void **ptr, size_t size)``` or ```cudaHostAlloc(void ** pHost, size_t size, unsigned int flags)```, the device can access this memory directly, making transfers faster.
free pinned memory using ```cudaFreeHost```.

flags:
* ```cudaHostAllocDefault``` Default, allocates pinned memory, same behavior as calling cudaMallocHost().
* ```cudaHostAllocMapped```Allocates pinned memory that can also be mapped into the devices address space, allowing the device to directly access host memory. In other words zero-copy memory that is directly accessible from the GPU.
* ```cudaHostAllocPortable```Memory that can be shared across different GPUs in the system.

**Zero-Copy Memory:**

Zero-copy memory allows the device to directly access host memory without the need to explicitly copy data between host and device. This is useful for applications that need to avoid redundant copies of data.

Allocated using ```cudaHostAlloc()``` with the ```cudaHostAllocMapped``` flag.
Mapped to the device using ```cudaHostGetDevicePointer()```.

example: ```cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)```