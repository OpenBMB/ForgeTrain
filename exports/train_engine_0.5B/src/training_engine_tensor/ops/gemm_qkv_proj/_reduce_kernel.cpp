/* Round 40: custom float4-vectorized 8-way FP32 reduction for QKV wgrad split-K=8.
 *
 * Replaces torch.sum(partials_flat, dim=0) (~45.8us @ H100) with a dedicated
 * coalesced float4 8-way add (~30us). Net savings ~15us in isolation, ~3.5us
 * end-to-end after the side-stream overlap with dgrad shadows most of the
 * reducer tail.
 *
 * Input  : partials_flat [8, M, N] FP32 contiguous, total elements % 4 == 0.
 * Output : out           [M, N]    FP32 contiguous (newly allocated).
 *
 * One thread issues 8 float4 loads (one per split plane), accumulates the
 * 8-way add fully in registers, and writes one float4 of result.
 * Grid = ceil(M*N/4 / 256), block = 256. No SMEM, no atomics, no scratch.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

constexpr int NUM_SPLITS_HARD = 8;

template <int TPB>
__global__ void reduce_8way_f32_kernel(
    const float4* __restrict__ in,
    float4* __restrict__ out,
    int n_vec,
    int plane_stride
) {
    int tid = blockIdx.x * TPB + threadIdx.x;
    if (tid >= n_vec) return;

    float4 a0 = in[                    tid];
    float4 a1 = in[1 * plane_stride + tid];
    float4 a2 = in[2 * plane_stride + tid];
    float4 a3 = in[3 * plane_stride + tid];
    float4 a4 = in[4 * plane_stride + tid];
    float4 a5 = in[5 * plane_stride + tid];
    float4 a6 = in[6 * plane_stride + tid];
    float4 a7 = in[7 * plane_stride + tid];

    float4 r;
    r.x = a0.x + a1.x + a2.x + a3.x + a4.x + a5.x + a6.x + a7.x;
    r.y = a0.y + a1.y + a2.y + a3.y + a4.y + a5.y + a6.y + a7.y;
    r.z = a0.z + a1.z + a2.z + a3.z + a4.z + a5.z + a6.z + a7.z;
    r.w = a0.w + a1.w + a2.w + a3.w + a4.w + a5.w + a6.w + a7.w;

    out[tid] = r;
}

static void launch_reduce(const torch::Tensor& partials_flat,
                          torch::Tensor& out) {
    TORCH_CHECK(partials_flat.is_cuda(), "partials_flat must be CUDA");
    TORCH_CHECK(partials_flat.dtype() == torch::kFloat32, "must be float32");
    TORCH_CHECK(partials_flat.is_contiguous(), "must be contiguous");
    TORCH_CHECK(partials_flat.size(0) == NUM_SPLITS_HARD,
                "leading dim must equal 8");
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
    int n_vec = static_cast<int>(total / 4);

    constexpr int TPB = 256;
    int blocks = (n_vec + TPB - 1) / TPB;

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    reduce_8way_f32_kernel<TPB><<<blocks, TPB, 0, stream>>>(
        reinterpret_cast<const float4*>(partials_flat.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        n_vec,
        n_vec
    );
}

torch::Tensor reduce_8way_fp32(torch::Tensor partials_flat) {
    int64_t M = (partials_flat.dim() == 3) ? partials_flat.size(1) : 1;
    int64_t N = (partials_flat.dim() == 3) ? partials_flat.size(2)
                                            : partials_flat.size(1);
    auto out = torch::empty({M, N}, partials_flat.options());
    launch_reduce(partials_flat, out);
    return out;
}

void reduce_8way_fp32_into(torch::Tensor partials_flat, torch::Tensor out) {
    launch_reduce(partials_flat, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_8way_fp32", &reduce_8way_fp32,
          "Vectorized FP32 8-way reduction (allocates output)");
    m.def("reduce_8way_fp32_into", &reduce_8way_fp32_into,
          "Vectorized FP32 8-way reduction into a pre-allocated output");
}
