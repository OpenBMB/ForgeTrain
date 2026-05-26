/* in-house persistent GEMM — attn_out_proj fwd/dgrad only. Static layout, in-place output.
   wgrad uses the original CuTeDSL operator (better MFU for this shape). */
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "inhouse_aop_fwd.h"
#include "inhouse_aop_dgrad.h"

static bool g_init = false;
static inhouse_aop_fwd_Kernel_Module_t g_fwd;
static inhouse_aop_dgrad_Kernel_Module_t g_dgrad;

static void ensure_init() {
    if (!g_init) {
        inhouse_aop_fwd_Kernel_Module_Load(&g_fwd);
        inhouse_aop_dgrad_Kernel_Module_Load(&g_dgrad);
        g_init = true;
    }
}

void inhouse_aop_fwd_fast(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    ensure_init();
    inhouse_aop_fwd_Tensor_a_t a = { x.data_ptr() };
    inhouse_aop_fwd_Tensor_b_t b = { w.data_ptr() };
    inhouse_aop_fwd_Tensor_c_t c = { out.data_ptr() };
    cute_dsl_inhouse_aop_fwd_wrapper(&g_fwd, &a, &b, &c, c10::cuda::getCurrentCUDAStream().stream());
}

void inhouse_aop_dgrad_fast(torch::Tensor dy, torch::Tensor w, torch::Tensor dx) {
    ensure_init();
    inhouse_aop_dgrad_Tensor_a_t a = { dy.data_ptr() };
    inhouse_aop_dgrad_Tensor_b_t b = { w.data_ptr() };
    inhouse_aop_dgrad_Tensor_c_t c = { dx.data_ptr() };
    cute_dsl_inhouse_aop_dgrad_wrapper(&g_dgrad, &a, &b, &c, c10::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_fwd_fast", &inhouse_aop_fwd_fast, "inhouse aop fwd");
    m.def("gemm_dgrad_fast", &inhouse_aop_dgrad_fast, "inhouse aop dgrad");
}
