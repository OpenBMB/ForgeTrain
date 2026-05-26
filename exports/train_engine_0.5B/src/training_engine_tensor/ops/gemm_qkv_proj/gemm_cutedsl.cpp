/*
 * C++ wrapper for CuTeDSL-exported GEMM kernels (qkv_proj).
 *
 * Links against .o files exported by CuTeDSL's export_to_c(),
 * providing CuTeDSL kernel performance with zero Python dispatch overhead.
 *
 * Directions:
 *   fwd:   C[40960,1280] = A[40960,1024] * B[1280,1024]   BF16→BF16  (static layout)
 *   dgrad: C[40960,1024] = A[40960,1280] * B[1024,1280]   BF16→BF16  (static layout)
 *   wgrad: C[1280,1024,4] = A[1280,10240,4] * B[1024,10240,4]  BF16→FP32 (static layout, split-K=4)
 *
 * R77: all three directions use is_dynamic_layout=False (shapes baked into
 * kernel binary). Descriptor structs are { void *data; } only.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <string>

#include "gemm_fwd.h"
#include "gemm_dgrad.h"
#include "gemm_wgrad.h"

extern "C" void aot_reduce_4way_f32(
    const float* partials, float* out, int64_t total_elems, cudaStream_t stream);

static bool g_initialized = false;
static gemm_fwd_Kernel_Module_t g_fwd_module;
static gemm_dgrad_Kernel_Module_t g_dgrad_module;
static gemm_wgrad_Kernel_Module_t g_wgrad_module;

static void ensure_init() {
    if (!g_initialized) {
        gemm_fwd_Kernel_Module_Load(&g_fwd_module);
        gemm_dgrad_Kernel_Module_Load(&g_dgrad_module);
        gemm_wgrad_Kernel_Module_Load(&g_wgrad_module);
        g_initialized = true;
    }
}


/* ---------- fwd: Y = X @ W^T ----------
 * Static layout: only data pointers needed (shapes baked in at export time).
 *   X[40960, 1024] BF16 row-major
 *   W[1280,  1024] BF16 row-major
 *   Y[40960, 1280] BF16 row-major
 */
torch::Tensor cutedsl_gemm_fwd_fast(torch::Tensor x, torch::Tensor w) {
    ensure_init();
    TORCH_CHECK(x.is_cuda() && w.is_cuda());
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16);

    auto x_c = x.contiguous();
    auto w_c = w.contiguous();

    int64_t K = x_c.size(-1);
    int64_t M = x_c.numel() / K;
    int64_t N = w_c.size(0);

    TORCH_CHECK(M == 40960 && N == 1280 && K == 1024,
                "fwd: expected M=40960 N=1280 K=1024; got M=",
                M, " N=", N, " K=", K);
    TORCH_CHECK(w_c.size(1) == K, "fwd: weight K mismatch");

    auto sizes = x_c.sizes().vec();
    sizes.back() = N;
    auto out = torch::empty(sizes, x_c.options());

    gemm_fwd_Tensor_a_t a_desc = { x_c.data_ptr() };
    gemm_fwd_Tensor_b_t b_desc = { w_c.data_ptr() };
    gemm_fwd_Tensor_c_t c_desc = { out.data_ptr() };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int32_t ret = cute_dsl_gemm_fwd_wrapper(&g_fwd_module, &a_desc, &b_desc, &c_desc, stream);
    TORCH_CHECK(ret == 0, "CuTeDSL qkv fwd kernel failed with code ", ret);

    return out;
}

/* ---------- dgrad: dX = dY @ W ----------
 * Static layout.
 *   dY[40960, 1280] BF16 row-major
 *   W [1280,  1024] BF16 row-major → viewed as B[1024, 1280] b_major=n
 *   dX[40960, 1024] BF16 row-major
 */
torch::Tensor cutedsl_gemm_dgrad_fast(torch::Tensor dy, torch::Tensor w) {
    ensure_init();
    TORCH_CHECK(dy.is_cuda() && w.is_cuda());
    TORCH_CHECK(dy.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16);

    auto dy_c = dy.contiguous();
    auto w_c = w.contiguous();

    int64_t K_in = dy_c.size(-1);      // 1280
    int64_t M = dy_c.numel() / K_in;   // 40960
    int64_t N_out = w_c.size(1);        // 1024

    TORCH_CHECK(M == 40960 && K_in == 1280 && N_out == 1024,
                "dgrad: expected M=40960 K=1280 N=1024; got M=",
                M, " K=", K_in, " N=", N_out);
    TORCH_CHECK(w_c.size(0) == K_in, "dgrad: weight shape mismatch");

    auto sizes = dy_c.sizes().vec();
    sizes.back() = N_out;
    auto dx = torch::empty(sizes, dy_c.options());

    gemm_dgrad_Tensor_a_t a_desc = { dy_c.data_ptr() };
    gemm_dgrad_Tensor_b_t b_desc = { w_c.data_ptr() };
    gemm_dgrad_Tensor_c_t c_desc = { dx.data_ptr() };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int32_t ret = cute_dsl_gemm_dgrad_wrapper(&g_dgrad_module, &a_desc, &b_desc, &c_desc, stream);
    TORCH_CHECK(ret == 0, "CuTeDSL qkv dgrad kernel failed with code ", ret);

    return dx;
}

/* ---------- wgrad: partials = dY_batched * X_batched ----------
 * R77: static layout (shapes baked into kernel binary).
 *   A = dY^T batched: [M=1280, K=10240, L=4] col-major per batch
 *   B = X^T batched:  [N=1024, K=10240, L=4] col-major per batch
 *   C = partials:     [M=1280, N=1024,  L=4] row-major FP32
 *
 * The runtime memory layout of dy_2d[40960,1280] and x_2d[40960,1024]
 * (both row-major) naturally maps to the col-major batched view when
 * reinterpreted as [M, chunk_K, L] with stride (1, M, M*chunk_K) — the
 * kernel accesses the same bytes in the same order.
 *
 * partials_flat[4, 1280, 1024] row-major has identical element layout to
 * the kernel's C[1280, 1024, 4] with stride (N=1024, 1, M*N=1310720).
 */
void cutedsl_gemm_wgrad(torch::Tensor dy_2d, torch::Tensor x_2d,
                        torch::Tensor partials_flat) {
    ensure_init();
    TORCH_CHECK(dy_2d.is_cuda() && x_2d.is_cuda() && partials_flat.is_cuda());
    TORCH_CHECK(dy_2d.dtype() == torch::kBFloat16 && x_2d.dtype() == torch::kBFloat16);
    TORCH_CHECK(partials_flat.dtype() == torch::kFloat32);

    TORCH_CHECK(dy_2d.size(0) == 40960 && dy_2d.size(1) == 1280,
                "wgrad: expected dy_2d[40960,1280]; got [",
                dy_2d.size(0), ",", dy_2d.size(1), "]");
    TORCH_CHECK(x_2d.size(0) == 40960 && x_2d.size(1) == 1024,
                "wgrad: expected x_2d[40960,1024]; got [",
                x_2d.size(0), ",", x_2d.size(1), "]");
    TORCH_CHECK(partials_flat.size(0) == 4 && partials_flat.size(1) == 1280
                && partials_flat.size(2) == 1024,
                "wgrad: expected partials_flat[4,1280,1024]");

    gemm_wgrad_Tensor_a_t a_desc = { dy_2d.data_ptr() };
    gemm_wgrad_Tensor_b_t b_desc = { x_2d.data_ptr() };
    gemm_wgrad_Tensor_c_t c_desc = { partials_flat.data_ptr() };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int32_t ret = cute_dsl_gemm_wgrad_wrapper(&g_wgrad_module, &a_desc, &b_desc, &c_desc, stream);
    TORCH_CHECK(ret == 0, "CuTeDSL qkv wgrad kernel failed with code ", ret);
}

/* ---------- R77: serial bwd — dgrad + wgrad + reduce, one C++ call ----------
 *
 * All three operations run serially on the current CUDA stream.
 * Persistent kernels saturate 132 SMs, so stream overlap gives no benefit
 * (R76 confirmed). This single-entry-point path eliminates ~6-8 µs of
 * Python↔C++ round-trip overhead per bwd call.
 *
 * partials_flat: pre-allocated [4, 1280, 1024] FP32 (reused across calls).
 * d_weight:      pre-allocated [1280, 1024] FP32 output.
 */
std::tuple<torch::Tensor, torch::Tensor>
cutedsl_qkv_bwd_serial(torch::Tensor dy, torch::Tensor x, torch::Tensor w,
                        torch::Tensor partials_flat, torch::Tensor d_weight) {
    ensure_init();

    TORCH_CHECK(dy.is_cuda() && x.is_cuda() && w.is_cuda());
    TORCH_CHECK(dy.dtype() == torch::kBFloat16 &&
                x.dtype()  == torch::kBFloat16 &&
                w.dtype()  == torch::kBFloat16);
    TORCH_CHECK(partials_flat.dtype() == torch::kFloat32);
    TORCH_CHECK(d_weight.dtype() == torch::kFloat32);

    auto dy_c = dy.contiguous();
    auto x_c  = x.contiguous();
    auto w_c  = w.contiguous();

    int64_t N_dy = dy_c.size(-1);            // 1280
    int64_t K_x  = x_c.size(-1);             // 1024
    int64_t M    = dy_c.numel() / N_dy;      // 40960

    TORCH_CHECK(M == 40960 && N_dy == 1280 && K_x == 1024,
                "bwd_serial: shape mismatch M=", M, " N=", N_dy, " K=", K_x);
    TORCH_CHECK(partials_flat.size(0) == 4 && partials_flat.size(1) == 1280
                && partials_flat.size(2) == 1024,
                "bwd_serial: partials_flat shape mismatch");
    TORCH_CHECK(d_weight.size(0) == 1280 && d_weight.size(1) == 1024,
                "bwd_serial: d_weight shape mismatch");

    auto d_input = torch::empty(x_c.sizes(), x_c.options());
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    /* 1. dgrad (static layout) */
    {
        gemm_dgrad_Tensor_a_t a_desc = { dy_c.data_ptr() };
        gemm_dgrad_Tensor_b_t b_desc = { w_c.data_ptr() };
        gemm_dgrad_Tensor_c_t c_desc = { d_input.data_ptr() };
        int32_t ret = cute_dsl_gemm_dgrad_wrapper(
            &g_dgrad_module, &a_desc, &b_desc, &c_desc, stream);
        TORCH_CHECK(ret == 0, "CuTeDSL qkv dgrad failed: ", ret);
    }

    /* 2. wgrad (static layout) — writes to partials_flat[4,1280,1024] */
    {
        auto dy_2d = dy_c.reshape({M, N_dy});
        auto x_2d  = x_c.reshape({M, K_x});
        gemm_wgrad_Tensor_a_t a_desc = { dy_2d.data_ptr() };
        gemm_wgrad_Tensor_b_t b_desc = { x_2d.data_ptr() };
        gemm_wgrad_Tensor_c_t c_desc = { partials_flat.data_ptr() };
        int32_t ret = cute_dsl_gemm_wgrad_wrapper(
            &g_wgrad_module, &a_desc, &b_desc, &c_desc, stream);
        TORCH_CHECK(ret == 0, "CuTeDSL qkv wgrad failed: ", ret);
    }

    /* 3. reduce: d_weight = partials_flat.sum(dim=0)
     * R78: replaced at::sum_out with vectorized 4-way float4 reducer.
     * at::sum_out ~25-30µs → custom kernel ~8-12µs for [4,1280,1024] FP32.
     */
    aot_reduce_4way_f32(
        partials_flat.data_ptr<float>(),
        d_weight.data_ptr<float>(),
        static_cast<int64_t>(1280) * 1024,
        stream);

    return std::make_tuple(d_input, d_weight);
}

/* ---------- R78: standalone reduce (for Python-side overlap path) ---------- */
void cutedsl_aot_reduce(torch::Tensor partials_flat, torch::Tensor d_weight) {
    TORCH_CHECK(partials_flat.is_cuda() && d_weight.is_cuda());
    TORCH_CHECK(partials_flat.dtype() == torch::kFloat32);
    TORCH_CHECK(d_weight.dtype() == torch::kFloat32);
    TORCH_CHECK(partials_flat.size(0) == 4 && partials_flat.size(1) == 1280
                && partials_flat.size(2) == 1024,
                "aot_reduce: expected partials_flat[4,1280,1024]");
    TORCH_CHECK(d_weight.size(0) == 1280 && d_weight.size(1) == 1024,
                "aot_reduce: expected d_weight[1280,1024]");

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    aot_reduce_4way_f32(
        partials_flat.data_ptr<float>(),
        d_weight.data_ptr<float>(),
        static_cast<int64_t>(1280) * 1024,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_fwd_fast",   &cutedsl_gemm_fwd_fast,
          "CuTeDSL qkv_proj fwd (3-D in/out, static layout)");
    m.def("gemm_dgrad_fast", &cutedsl_gemm_dgrad_fast,
          "CuTeDSL qkv_proj dgrad (3-D in/out, static layout)");
    m.def("gemm_wgrad",      &cutedsl_gemm_wgrad,
          "CuTeDSL qkv_proj wgrad (static layout, split-K=4 baked in)");
    m.def("bwd_serial",      &cutedsl_qkv_bwd_serial,
          "CuTeDSL qkv_proj serial bwd: dgrad + wgrad + reduce in one call");
    m.def("aot_reduce",      &cutedsl_aot_reduce,
          "4-way FP32 reduce for wgrad split-K (standalone, for overlap path)");
}
