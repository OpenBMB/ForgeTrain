/* Round 48: faster N-way (N ∈ {4, 8}) FP32 reducer for QKV wgrad split-K.
 *
 * Goal: cut the 30 µs reducer kernel down to ~15-20 µs by:
 *   (a) supporting NUM_SPLITS ∈ {4, 8} so the high-level wgrad can pick a
 *       smaller split-K when precision allows (Round 4 measured split-K=4
 *       op-unit PASS; only op-short failed, which is no longer hard gate);
 *   (b) raising per-thread ILP — each thread now reduces 4 float4 lanes
 *       (16 FP32 elements / 64 B) instead of 1, halving the launch
 *       overhead and letting the LSU coalesce more aggressively.
 *
 * Layout assumption:
 *   partials_flat is [L, M, N] FP32 contiguous (L = NUM_SPLITS).
 *   plane stride = M * N elements.  total = M * N elements.
 *   Caller guarantees (total % 16) == 0 (i.e. 4 float4 / thread).
 *
 *   For QKV wgrad: M=1280, N=1024 → total=1310720 = 4096 * 320, divisible by 16.
 *
 * Kernel layout: one thread emits 4 float4 of output.
 *                grid = ceil(n_vec_4 / TPB), TPB = 256.
 *                n_vec_4 = total / 16 = M*N / 16 (each "n_vec_4" is 4 float4).
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

constexpr int TPB_R48 = 256;
constexpr int VEC_PER_TID = 4;  // 4 float4 per thread = 16 FP32

template <int NSPLIT>
__global__ void __launch_bounds__(TPB_R48)
reduce_nway_f32_v48_kernel(
    const float4* __restrict__ in,
    float4* __restrict__ out,
    int n_vec_4,            // = total_elems / 16
    int plane_stride_vec    // = total_elems / 4 (in float4 units)
) {
    int tid_4 = blockIdx.x * TPB_R48 + threadIdx.x;
    if (tid_4 >= n_vec_4) return;

    // Each thread handles 4 consecutive float4 lanes within the same plane,
    // for all NSPLIT planes.  Layout: lane i is at base = tid_4*4 + i.
    const int base = tid_4 * VEC_PER_TID;

    float4 acc[VEC_PER_TID];
    #pragma unroll
    for (int v = 0; v < VEC_PER_TID; ++v) {
        acc[v] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    #pragma unroll
    for (int p = 0; p < NSPLIT; ++p) {
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

// Tail kernel for the (rare) case where n_vec_4 doesn't divide evenly,
// used as a fallback for the last few lanes.  Same as the round-40 8-way
// kernel but templated on NSPLIT.
template <int NSPLIT>
__global__ void reduce_nway_f32_v48_tail_kernel(
    const float4* __restrict__ in,
    float4* __restrict__ out,
    int n_vec,           // = total_elems / 4
    int plane_stride_vec // = total_elems / 4
) {
    int tid = blockIdx.x * TPB_R48 + threadIdx.x;
    if (tid >= n_vec) return;

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    #pragma unroll
    for (int p = 0; p < NSPLIT; ++p) {
        float4 a = in[p * plane_stride_vec + tid];
        acc.x += a.x;
        acc.y += a.y;
        acc.z += a.z;
        acc.w += a.w;
    }
    out[tid] = acc;
}

template <int NSPLIT>
static void launch_reduce_nway_v48(const torch::Tensor& partials_flat,
                                   torch::Tensor& out) {
    TORCH_CHECK(partials_flat.is_cuda(), "partials_flat must be CUDA");
    TORCH_CHECK(partials_flat.dtype() == torch::kFloat32, "must be float32");
    TORCH_CHECK(partials_flat.is_contiguous(), "must be contiguous");
    TORCH_CHECK(partials_flat.size(0) == NSPLIT,
                "leading dim must equal NSPLIT");
    TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kFloat32,
                "out must be CUDA fp32");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    int64_t M, N;
    if (partials_flat.dim() == 3) {
        M = partials_flat.size(1);
        N = partials_flat.size(2);
    } else {
        TORCH_CHECK(partials_flat.dim() == 2, "expect [L, M, N] or [L, MN]");
        M = 1;
        N = partials_flat.size(1);
    }

    int64_t total = M * N;
    TORCH_CHECK((total % 4) == 0, "total elements must be divisible by 4");
    TORCH_CHECK(out.numel() == total, "out shape mismatch");

    int n_vec = static_cast<int>(total / 4);  // float4 units
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    if ((n_vec % VEC_PER_TID) == 0) {
        // Fast path: each thread handles 4 float4.
        int n_vec_4 = n_vec / VEC_PER_TID;
        int blocks = (n_vec_4 + TPB_R48 - 1) / TPB_R48;
        reduce_nway_f32_v48_kernel<NSPLIT><<<blocks, TPB_R48, 0, stream>>>(
            reinterpret_cast<const float4*>(partials_flat.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            n_vec_4,
            n_vec
        );
    } else {
        // Slow path: 1 float4 per thread.
        int blocks = (n_vec + TPB_R48 - 1) / TPB_R48;
        reduce_nway_f32_v48_tail_kernel<NSPLIT><<<blocks, TPB_R48, 0, stream>>>(
            reinterpret_cast<const float4*>(partials_flat.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            n_vec,
            n_vec
        );
    }
}

// ---- 8-way (drop-in replacement for round-40 reducer, faster ILP) ----

torch::Tensor reduce_8way_v48(torch::Tensor partials_flat) {
    int64_t M = (partials_flat.dim() == 3) ? partials_flat.size(1) : 1;
    int64_t N = (partials_flat.dim() == 3) ? partials_flat.size(2)
                                            : partials_flat.size(1);
    auto out = torch::empty({M, N}, partials_flat.options());
    launch_reduce_nway_v48<8>(partials_flat, out);
    return out;
}

void reduce_8way_v48_into(torch::Tensor partials_flat, torch::Tensor out) {
    launch_reduce_nway_v48<8>(partials_flat, out);
}

// ---- 4-way (new for Round 48; pairs with NUM_SPLITS=4 wgrad) ----

torch::Tensor reduce_4way_v48(torch::Tensor partials_flat) {
    int64_t M = (partials_flat.dim() == 3) ? partials_flat.size(1) : 1;
    int64_t N = (partials_flat.dim() == 3) ? partials_flat.size(2)
                                            : partials_flat.size(1);
    auto out = torch::empty({M, N}, partials_flat.options());
    launch_reduce_nway_v48<4>(partials_flat, out);
    return out;
}

void reduce_4way_v48_into(torch::Tensor partials_flat, torch::Tensor out) {
    launch_reduce_nway_v48<4>(partials_flat, out);
}

// ---- 2-way (smallest, lowest reducer cost; precision likely too tight) ----

torch::Tensor reduce_2way_v48(torch::Tensor partials_flat) {
    int64_t M = (partials_flat.dim() == 3) ? partials_flat.size(1) : 1;
    int64_t N = (partials_flat.dim() == 3) ? partials_flat.size(2)
                                            : partials_flat.size(1);
    auto out = torch::empty({M, N}, partials_flat.options());
    launch_reduce_nway_v48<2>(partials_flat, out);
    return out;
}

void reduce_2way_v48_into(torch::Tensor partials_flat, torch::Tensor out) {
    launch_reduce_nway_v48<2>(partials_flat, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_8way_v48", &reduce_8way_v48,
          "Round-48 8-way FP32 reducer with 4-float4 ILP (allocates output)");
    m.def("reduce_8way_v48_into", &reduce_8way_v48_into,
          "Round-48 8-way FP32 reducer with 4-float4 ILP (in-place)");
    m.def("reduce_4way_v48", &reduce_4way_v48,
          "Round-48 4-way FP32 reducer (allocates output)");
    m.def("reduce_4way_v48_into", &reduce_4way_v48_into,
          "Round-48 4-way FP32 reducer (in-place)");
    m.def("reduce_2way_v48", &reduce_2way_v48,
          "Round-48 2-way FP32 reducer (allocates output)");
    m.def("reduce_2way_v48_into", &reduce_2way_v48_into,
          "Round-48 2-way FP32 reducer (in-place)");
}
