/*
 * C++ wrapper for CuTeDSL-exported GEMM kernels (gemm_attn_out_proj).
 *
 * Round 44: ALL directions now use is_dynamic_layout=False — shapes and
 * strides are baked into the kernel binary.  Tensor descriptor is just
 * { void *data; } for all three directions.  This eliminates the ~36µs
 * per-call TMA descriptor-fill overhead that R26 identified as the root
 * cause of AOT being slower than JIT for batched wgrad.
 *
 * Directions:
 *   fwd:   C[M,N] = A[M,K] * B[N,K]         BF16->BF16  M=40960 N=1024 K=1024   STATIC
 *   dgrad: C[M,N] = A[M,K] * B[N,K]         BF16->BF16  M=40960 N=1024 K=1024   STATIC
 *   wgrad: C[M,N,L] = A[M,K,L] * B[N,K,L]  BF16->FP32  M=1024 N=1024 K=20480 L=2  STATIC
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "aop_fwd.h"
#include "aop_dgrad.h"
#include "aop_wgrad.h"

static bool g_initialized = false;
static aop_fwd_Kernel_Module_t   g_fwd_module;
static aop_dgrad_Kernel_Module_t g_dgrad_module;
static aop_wgrad_Kernel_Module_t g_wgrad_module;

static void ensure_init() {
    if (!g_initialized) {
        aop_fwd_Kernel_Module_Load(&g_fwd_module);
        aop_dgrad_Kernel_Module_Load(&g_dgrad_module);
        aop_wgrad_Kernel_Module_Load(&g_wgrad_module);
        g_initialized = true;
    }
}

/* ---------- fwd: Y = X @ W^T ----------
 * R44: is_dynamic_layout=False — descriptor is { void *data; } only.
 * Shapes locked: X[40960,1024] W[1024,1024] Y[40960,1024].
 */
void cutedsl_aop_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    ensure_init();
    TORCH_CHECK(x.size(0) == 40960 && x.size(1) == 1024,
                "fwd: expected X[40960,1024], got [", x.size(0), ",", x.size(1), "]");
    TORCH_CHECK(w.size(0) == 1024 && w.size(1) == 1024,
                "fwd: expected W[1024,1024], got [", w.size(0), ",", w.size(1), "]");

    aop_fwd_Tensor_a_t a = { x.data_ptr() };
    aop_fwd_Tensor_b_t b = { w.data_ptr() };
    aop_fwd_Tensor_c_t c = { out.data_ptr() };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int32_t ret = cute_dsl_aop_fwd_wrapper(&g_fwd_module, &a, &b, &c, stream);
    TORCH_CHECK(ret == 0, "CuTeDSL aop_fwd kernel failed with code ", ret);
}

/* ---------- dgrad: dX = dY @ W ----------
 * R44: is_dynamic_layout=False — descriptor is { void *data; } only.
 * B is compiled with b_major="n" (col-major NK view of W); the col-major
 * strides are baked into the kernel binary.
 * Shapes locked: dY[40960,1024] W[1024,1024] dX[40960,1024].
 */
void cutedsl_aop_dgrad(torch::Tensor dy, torch::Tensor w, torch::Tensor dx) {
    ensure_init();
    TORCH_CHECK(dy.size(0) == 40960 && dy.size(1) == 1024,
                "dgrad: expected dY[40960,1024], got [", dy.size(0), ",", dy.size(1), "]");
    TORCH_CHECK(w.size(0) == 1024 && w.size(1) == 1024,
                "dgrad: expected W[1024,1024], got [", w.size(0), ",", w.size(1), "]");

    aop_dgrad_Tensor_a_t a = { dy.data_ptr() };
    aop_dgrad_Tensor_b_t b = { w.data_ptr() };
    aop_dgrad_Tensor_c_t c = { dx.data_ptr() };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int32_t ret = cute_dsl_aop_dgrad_wrapper(&g_dgrad_module, &a, &b, &c, stream);
    TORCH_CHECK(ret == 0, "CuTeDSL aop_dgrad kernel failed with code ", ret);
}

/* ---------- wgrad: dW = dY^T @ X (split-K=2 batched) ----------
 * R44: is_dynamic_layout=False — descriptor is { void *data; } only.
 * Shapes locked: dY[40960,1024] X[40960,1024] partials[2,1024,1024].
 *
 * Strides baked at compile time:
 *   A[M=1024, K=20480, L=2]: col-major MK, stride=(1, 1024, 20480*1024)
 *   B[N=1024, K=20480, L=2]: col-major NK, stride=(1, 1024, 20480*1024)
 *   C[M=1024, N=1024, L=2]:  c_major=n,    stride=(1024, 1, 1024*1024)
 */
void cutedsl_aop_wgrad(torch::Tensor dy, torch::Tensor x, torch::Tensor partials) {
    ensure_init();
    TORCH_CHECK(dy.size(0) == 40960 && dy.size(1) == 1024,
                "wgrad: expected dY[40960,1024], got [", dy.size(0), ",", dy.size(1), "]");
    TORCH_CHECK(x.size(0) == 40960 && x.size(1) == 1024,
                "wgrad: expected X[40960,1024], got [", x.size(0), ",", x.size(1), "]");
    TORCH_CHECK(partials.size(0) == 2 && partials.size(1) == 1024 && partials.size(2) == 1024,
                "wgrad: expected partials[2,1024,1024], got [",
                partials.size(0), ",", partials.size(1), ",", partials.size(2), "]");

    aop_wgrad_Tensor_a_t a = { dy.data_ptr() };
    aop_wgrad_Tensor_b_t b = { x.data_ptr() };
    aop_wgrad_Tensor_c_t c = { partials.data_ptr() };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int32_t ret = cute_dsl_aop_wgrad_wrapper(&g_wgrad_module, &a, &b, &c, stream);
    TORCH_CHECK(ret == 0, "CuTeDSL aop_wgrad kernel failed with code ", ret);
}

/* ---------- combined bwd: dgrad + wgrad in one pybind11 call ----------
 * Saves one Python→C++ transition (~1-3µs of pybind11 overhead).
 * R44: serial execution (R21 confirmed overlap saves <0.5% but adds
 * ~20µs event/stream overhead for persistent kernels).
 */
void cutedsl_aop_bwd(torch::Tensor dy, torch::Tensor w, torch::Tensor dx,
                     torch::Tensor x, torch::Tensor partials) {
    ensure_init();

    aop_dgrad_Tensor_a_t a_dg = { dy.data_ptr() };
    aop_dgrad_Tensor_b_t b_dg = { w.data_ptr() };
    aop_dgrad_Tensor_c_t c_dg = { dx.data_ptr() };

    aop_wgrad_Tensor_a_t a_wg = { dy.data_ptr() };
    aop_wgrad_Tensor_b_t b_wg = { x.data_ptr() };
    aop_wgrad_Tensor_c_t c_wg = { partials.data_ptr() };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    int32_t ret_dg = cute_dsl_aop_dgrad_wrapper(&g_dgrad_module, &a_dg, &b_dg, &c_dg, stream);
    TORCH_CHECK(ret_dg == 0, "CuTeDSL aop_dgrad failed: ", ret_dg);
    int32_t ret_wg = cute_dsl_aop_wgrad_wrapper(&g_wgrad_module, &a_wg, &b_wg, &c_wg, stream);
    TORCH_CHECK(ret_wg == 0, "CuTeDSL aop_wgrad failed: ", ret_wg);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aop_fwd",   &cutedsl_aop_fwd,   "CuTeDSL gemm_attn_out_proj fwd");
    m.def("aop_dgrad", &cutedsl_aop_dgrad, "CuTeDSL gemm_attn_out_proj dgrad");
    m.def("aop_wgrad", &cutedsl_aop_wgrad, "CuTeDSL gemm_attn_out_proj wgrad (split-K=2)");
    m.def("aop_bwd",   &cutedsl_aop_bwd,   "CuTeDSL gemm_attn_out_proj combined dgrad+wgrad");
}
