/* Round 40 fix: nvcc shim for _reduce_kernel.cpp.
 *
 * The parallel agent committed _reduce_kernel.cpp with a __global__ kernel
 * and <<<>>> launch syntax inside it. torch.utils.cpp_extension dispatches
 * compilers by file extension — .cpp goes to g++, which cannot parse CUDA
 * syntax. We therefore include the .cpp file from this .cu shim so that
 * nvcc handles compilation while keeping the source-of-truth in the
 * committed .cpp file.
 */
#include "_reduce_kernel.cpp"
