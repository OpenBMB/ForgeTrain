/*
 * C++ wrapper for CuTeDSL-exported GEMM kernels (fc2 — MLP down projection).
 *
 * Directions:
 *   fwd:   C[40960,1024] = A[40960,4096] * B[1024,4096]   BF16→BF16  (static)
 *   dgrad: C[40960,4096] = A[40960,1024] * B[4096,1024]   BF16→BF16  (static)
 *   wgrad: C[1024,4096]  = A[1024,40960] * B[4096,40960]  BF16→FP32  (dynamic, col-major views)
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "gemm_fwd.h"
#include "gemm_dgrad.h"
#include "gemm_wgrad.h"

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
 * Static layout. X[40960,4096] BF16, W[1024,4096] BF16, Y[40960,1024] BF16.
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

    TORCH_CHECK(M == 40960 && N == 1024 && K == 4096,
                "fwd: expected M=40960 N=1024 K=4096; got M=",
                M, " N=", N, " K=", K);

    auto sizes = x_c.sizes().vec();
    sizes.back() = N;
    auto out = torch::empty(sizes, x_c.options());

    gemm_fwd_Tensor_a_t a_desc = { x_c.data_ptr() };
    gemm_fwd_Tensor_b_t b_desc = { w_c.data_ptr() };
    gemm_fwd_Tensor_c_t c_desc = { out.data_ptr() };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int32_t ret = cute_dsl_gemm_fwd_wrapper(&g_fwd_module, &a_desc, &b_desc, &c_desc, stream);
    TORCH_CHECK(ret == 0, "CuTeDSL fc2 fwd failed with code ", ret);

    return out;
}

/* ---------- dgrad: dX = dY @ W ----------
 * Static layout. dY[40960,1024], W[1024,4096] → B viewed as [4096,1024] b_major=n.
 * dX[40960,4096].
 */
torch::Tensor cutedsl_gemm_dgrad_fast(torch::Tensor dy, torch::Tensor w) {
    ensure_init();
    TORCH_CHECK(dy.is_cuda() && w.is_cuda());
    TORCH_CHECK(dy.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16);

    auto dy_c = dy.contiguous();
    auto w_c = w.contiguous();

    int64_t K_in = dy_c.size(-1);      // 1024
    int64_t M = dy_c.numel() / K_in;   // 40960
    int64_t N_out = w_c.size(1);        // 4096

    TORCH_CHECK(M == 40960 && K_in == 1024 && N_out == 4096,
                "dgrad: expected M=40960 K=1024 N=4096; got M=",
                M, " K=", K_in, " N=", N_out);

    auto sizes = dy_c.sizes().vec();
    sizes.back() = N_out;
    auto dx = torch::empty(sizes, dy_c.options());

    gemm_dgrad_Tensor_a_t a_desc = { dy_c.data_ptr() };
    gemm_dgrad_Tensor_b_t b_desc = { w_c.data_ptr() };
    gemm_dgrad_Tensor_c_t c_desc = { dx.data_ptr() };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int32_t ret = cute_dsl_gemm_dgrad_wrapper(&g_dgrad_module, &a_desc, &b_desc, &c_desc, stream);
    TORCH_CHECK(ret == 0, "CuTeDSL fc2 dgrad failed with code ", ret);

    return dx;
}

/* ---------- wgrad: dW = dY^T @ X ----------
 * Dynamic layout (col-major views).
 *   A = dY^T: [M=1024, K=40960] col-major
 *   B = X^T:  [N=4096, K=40960] col-major
 *   C = dW:   [M=1024, N=4096]  row-major FP32
 */
torch::Tensor cutedsl_gemm_wgrad_fast(torch::Tensor dy, torch::Tensor x) {
    ensure_init();
    TORCH_CHECK(dy.is_cuda() && x.is_cuda());
    TORCH_CHECK(dy.dtype() == torch::kBFloat16 && x.dtype() == torch::kBFloat16);

    auto dy_c = dy.contiguous();
    auto x_c = x.contiguous();

    int64_t N_dy = dy_c.size(-1);     // 1024 (M for wgrad)
    int64_t K_x  = x_c.size(-1);      // 4096 (N for wgrad)
    int64_t K_act = dy_c.numel() / N_dy;  // 40960

    TORCH_CHECK(K_act == 40960 && N_dy == 1024 && K_x == 4096,
                "wgrad: expected K=40960 M=1024 N=4096; got K=",
                K_act, " M=", N_dy, " N=", K_x);

    auto dw = torch::empty({N_dy, K_x}, dy_c.options().dtype(torch::kFloat32));

    int K = (int)K_act;   // 40960
    int M = (int)N_dy;    // 1024
    int N = (int)K_x;     // 4096

    // A col-major: [M=1024, K=40960, 1] stride (1, M, M*K)
    gemm_wgrad_Tensor_a_t a_desc = {
        dy_c.data_ptr(), {M, K, 1}, {(int64_t)M, (int64_t)M * K}
    };
    // B col-major: [N=4096, K=40960, 1] stride (1, N, N*K)
    gemm_wgrad_Tensor_b_t b_desc = {
        x_c.data_ptr(), {N, K, 1}, {(int64_t)N, (int64_t)N * K}
    };
    // C row-major: [M=1024, N=4096, 1] stride (N, 1, M*N)
    gemm_wgrad_Tensor_c_t c_desc = {
        dw.data_ptr(), {M, N, 1}, {(int64_t)N, (int64_t)M * N}
    };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int32_t ret = cute_dsl_gemm_wgrad_wrapper(&g_wgrad_module, &a_desc, &b_desc, &c_desc, stream);
    TORCH_CHECK(ret == 0, "CuTeDSL fc2 wgrad failed with code ", ret);

    return dw;
}

/* ---------- combined bwd ---------- */
std::tuple<torch::Tensor, torch::Tensor>
cutedsl_gemm_bwd_fast(torch::Tensor dy, torch::Tensor x, torch::Tensor w) {
    ensure_init();

    auto d_input = cutedsl_gemm_dgrad_fast(dy, w);
    auto d_weight = cutedsl_gemm_wgrad_fast(dy, x);

    return std::make_tuple(d_input, d_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_fwd_fast",   &cutedsl_gemm_fwd_fast,
          "CuTeDSL fc2 fwd (static layout)");
    m.def("gemm_dgrad_fast", &cutedsl_gemm_dgrad_fast,
          "CuTeDSL fc2 dgrad (static layout)");
    m.def("gemm_wgrad_fast", &cutedsl_gemm_wgrad_fast,
          "CuTeDSL fc2 wgrad (dynamic layout)");
    m.def("gemm_bwd_fast",   &cutedsl_gemm_bwd_fast,
          "CuTeDSL fc2 dgrad+wgrad combined");
}
