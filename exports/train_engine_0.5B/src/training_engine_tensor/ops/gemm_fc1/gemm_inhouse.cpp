/* in-house persistent GEMM — fc1 fwd/dgrad/wgrad. Static layout, in-place output. */
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "inhouse_fc1_fwd.h"
#include "inhouse_fc1_dgrad.h"
#include "inhouse_fc1_wgrad.h"

static bool g_init = false;
static inhouse_fc1_fwd_Kernel_Module_t g_fwd;
static inhouse_fc1_dgrad_Kernel_Module_t g_dgrad;
static inhouse_fc1_wgrad_Kernel_Module_t g_wgrad;

static void ensure_init() {
    if (!g_init) {
        inhouse_fc1_fwd_Kernel_Module_Load(&g_fwd);
        inhouse_fc1_dgrad_Kernel_Module_Load(&g_dgrad);
        inhouse_fc1_wgrad_Kernel_Module_Load(&g_wgrad);
        g_init = true;
    }
}

void inhouse_fc1_fwd_fast(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    ensure_init();
    inhouse_fc1_fwd_Tensor_a_t a = { x.data_ptr() };
    inhouse_fc1_fwd_Tensor_b_t b = { w.data_ptr() };
    inhouse_fc1_fwd_Tensor_c_t c = { out.data_ptr() };
    cute_dsl_inhouse_fc1_fwd_wrapper(&g_fwd, &a, &b, &c, c10::cuda::getCurrentCUDAStream().stream());
}

void inhouse_fc1_dgrad_fast(torch::Tensor dy, torch::Tensor w, torch::Tensor dx) {
    ensure_init();
    inhouse_fc1_dgrad_Tensor_a_t a = { dy.data_ptr() };
    inhouse_fc1_dgrad_Tensor_b_t b = { w.data_ptr() };
    inhouse_fc1_dgrad_Tensor_c_t c = { dx.data_ptr() };
    cute_dsl_inhouse_fc1_dgrad_wrapper(&g_dgrad, &a, &b, &c, c10::cuda::getCurrentCUDAStream().stream());
}

void inhouse_fc1_wgrad_fast(torch::Tensor dy, torch::Tensor x, torch::Tensor dw) {
    ensure_init();
    inhouse_fc1_wgrad_Tensor_a_t a = { dy.data_ptr() };
    inhouse_fc1_wgrad_Tensor_b_t b = { x.data_ptr() };
    inhouse_fc1_wgrad_Tensor_c_t c = { dw.data_ptr() };
    cute_dsl_inhouse_fc1_wgrad_wrapper(&g_wgrad, &a, &b, &c, c10::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_fwd_fast", &inhouse_fc1_fwd_fast, "inhouse fc1 fwd");
    m.def("gemm_dgrad_fast", &inhouse_fc1_dgrad_fast, "inhouse fc1 dgrad");
    m.def("gemm_wgrad", &inhouse_fc1_wgrad_fast, "inhouse fc1 wgrad (FP32)");
}
