# CUDA debug utility function for deep learning operator

## Requirements

- `cmake` (>=3.13.2, modify if needed)
- `nvcc` (CUDA C Compiler)

## General steps

1. Copy-paste the concerned .cu kernel file to project root.
2. Make sure arguments are c primitive types, e.g. float* rather than at::tensor.
3. Append a `main()` function in .cu file (see example).
4. Compile and run with the following commands:
```bash
mkdir build && cd build
cmake ..
make
./<exec_name>  (e.g. ./example)
```
5. Now use `cuda-gdb` for tracing!

## Example

```cpp
#include "debug_utils.h"

...

scalar_t *attx;
attx = Ones<scalar_t>({4, 3, 1, 64});
example_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(attx);
cudaFree(attx); 
checkCudaErrors(cudaGetLastError());
```

## Notes

1. One must call `cudaFree()` at the end of program. Otherwise it would
not be able to break inside kernel code.

2. When seeing some error like"too many resources requeted, error code 0x7",
consider reduce CUDA_NUM_THREADS from 1024 to a smaller one (e.g. 256).


## Acknowledgement

The demo example.cu is taken from 
https://github.com/researchmm/tasn
