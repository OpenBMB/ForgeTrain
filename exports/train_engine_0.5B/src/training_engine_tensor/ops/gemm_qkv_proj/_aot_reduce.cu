/* R78: Inline 4-way FP32 reducer for AOT bwd_serial path.
 *
 * Replaces at::sum_out (PyTorch generic reduction, ~25-30µs for [4,1280,1024])
 * with a float4-vectorized 4-way add (~8-12µs).
 *
 * Derived from _reduce_kernel_v2.cu (Round 48 v48 reducer) but stripped to
 * a single extern "C" function callable from gemm_cutedsl.cpp.
 *
 * Layout: partials [4, M, N] FP32 contiguous → out [M, N] FP32 contiguous.
 * M=1280, N=1024 → total=1,310,720 elements, n_vec=327,680 float4,
 * n_vec_4=81,920 (each thread: 4 float4 × 4 planes = 64 FP32 loads + 16 stores).
 */
#include <cuda_runtime.h>
#include <stdint.h>

static constexpr int TPB = 256;
static constexpr int VEC_PER_TID = 4;

__global__ void __launch_bounds__(TPB)
reduce_4way_f32_kernel(
    const float4* __restrict__ in,
    float4* __restrict__ out,
    int n_vec_4,
    int plane_stride_vec
) {
    int tid_4 = blockIdx.x * TPB + threadIdx.x;
    if (tid_4 >= n_vec_4) return;

    const int base = tid_4 * VEC_PER_TID;

    float4 acc[VEC_PER_TID];
    #pragma unroll
    for (int v = 0; v < VEC_PER_TID; ++v) {
        acc[v] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const float4* plane = in + p * plane_stride_vec;
        #pragma unroll
        for (int v = 0; v < VEC_PER_TID; ++v) {
            float4 a = plane[base + v];
            acc[v].x += a.x;
            acc[v].y += a.y;
            acc[v].z += a.z;
            acc[v].w += a.w;
        }
    }

    #pragma unroll
    for (int v = 0; v < VEC_PER_TID; ++v) {
        out[base + v] = acc[v];
    }
}

extern "C" void aot_reduce_4way_f32(
    const float* partials,
    float* out,
    int64_t total_elems,
    cudaStream_t stream
) {
    int n_vec = static_cast<int>(total_elems / 4);
    int n_vec_4 = n_vec / VEC_PER_TID;
    int blocks = (n_vec_4 + TPB - 1) / TPB;

    reduce_4way_f32_kernel<<<blocks, TPB, 0, stream>>>(
        reinterpret_cast<const float4*>(partials),
        reinterpret_cast<float4*>(out),
        n_vec_4,
        n_vec
    );
}
