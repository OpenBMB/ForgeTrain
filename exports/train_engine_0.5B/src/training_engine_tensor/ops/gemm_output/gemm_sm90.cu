/*
 * CUTLASS 3.x SM90 GEMM kernel for gemm_output operator.
 *
 * Uses CollectiveBuilder with TMA + WGMMA for Hopper (SM90a).
 * FP32 accumulation for numerical precision.
 *
 * CUTLASS 3.x convention: D[m,n] = A[m,k] x B[n,k]
 * B is always "k-contiguous" (ColumnMajor tag → stride (K, 1)),
 * matching PyTorch's row-major [N,K] storage.
 */

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <cute/tensor.hpp>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

using namespace cute;

// ── Fwd tile: (128,128,64) single warp group — avoids C7510 WGMMA serialization ──
using TileShape_Fwd = Shape<_128, _128, _64>;
using ClusterShape_Fwd = Shape<_1, _1, _1>;

// ── Dgrad/Wgrad tile: small N=1024 → narrow N-tile, K=128 for fewer K-iters ──
using TileShape_Bwd = Shape<_128, _64, _128>;
using ClusterShape_Bwd = Shape<_1, _1, _1>;

// ── Layout types for fwd/dgrad (NN: A RowMajor, B ColumnMajor) ──

using LayoutA_NN = cutlass::layout::RowMajor;
using LayoutB_NN = cutlass::layout::ColumnMajor;
using LayoutC_NN = cutlass::layout::RowMajor;

using StrideA_NN = cutlass::gemm::TagToStrideA_t<LayoutA_NN>;
using StrideB_NN = cutlass::gemm::TagToStrideB_t<LayoutB_NN>;
using StrideC_NN = cutlass::gemm::TagToStrideC_t<LayoutC_NN>;

// ── Fwd: BF16→BF16, A RowMajor, B ColumnMajor ──

using CollectiveEpilogue_Fwd = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_Fwd, ClusterShape_Fwd,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::bfloat16_t, LayoutC_NN, 8,
    cutlass::bfloat16_t, LayoutC_NN, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop_Fwd = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, LayoutA_NN, 8,
    cutlass::bfloat16_t, LayoutB_NN, 8,
    float,
    TileShape_Fwd, ClusterShape_Fwd,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_Fwd::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel_Fwd = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop_Fwd,
    CollectiveEpilogue_Fwd
>;
using GemmDevice_Fwd = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_Fwd>;

// ── DGrad: BF16→BF16 (A RowMajor, B ColumnMajor) ──

using CollectiveEpilogue_DGrad = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_Bwd, ClusterShape_Bwd,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::bfloat16_t, LayoutC_NN, 8,
    cutlass::bfloat16_t, LayoutC_NN, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop_DGrad = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, LayoutA_NN, 8,
    cutlass::bfloat16_t, LayoutB_NN, 8,
    float,
    TileShape_Bwd, ClusterShape_Bwd,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_DGrad::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel_DGrad = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop_DGrad,
    CollectiveEpilogue_DGrad
>;
using GemmDevice_DGrad = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_DGrad>;

// ── WGrad: BF16→FP32, A ColumnMajor, B ColumnMajor ──

using LayoutA_TN = cutlass::layout::ColumnMajor;
using LayoutB_TN = cutlass::layout::ColumnMajor;
using LayoutC_TN = cutlass::layout::RowMajor;

using StrideA_TN = cutlass::gemm::TagToStrideA_t<LayoutA_TN>;
using StrideB_TN = cutlass::gemm::TagToStrideB_t<LayoutB_TN>;
using StrideC_TN = cutlass::gemm::TagToStrideC_t<LayoutC_TN>;

using CollectiveEpilogue_WGrad = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_Bwd, ClusterShape_Bwd,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    float, LayoutC_TN, 4,
    float, LayoutC_TN, 4,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop_WGrad = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, LayoutA_TN, 8,
    cutlass::bfloat16_t, LayoutB_TN, 8,
    float,
    TileShape_Bwd, ClusterShape_Bwd,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_WGrad::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel_WGrad = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop_WGrad,
    CollectiveEpilogue_WGrad
>;
using GemmDevice_WGrad = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_WGrad>;


// ── Static workspace caching ────────────────────────────────────────
// Avoid per-call torch::empty() overhead by reusing workspace tensors.

static torch::Tensor g_ws_fwd;
static torch::Tensor g_ws_dgrad;
static torch::Tensor g_ws_wgrad;

static torch::Tensor& get_workspace(torch::Tensor& cached, size_t needed, torch::Device dev) {
    if (!cached.defined() || cached.numel() < static_cast<long>(needed)) {
        cached = torch::empty(
            {std::max(static_cast<long>(needed), 1L)},
            torch::TensorOptions().dtype(torch::kUInt8).device(dev));
    }
    return cached;
}


// ── fwd: logits[M,N] = X[M,K] @ W[N,K]^T ──────────────────────────
void gemm_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    int M = x.size(0);
    int K = x.size(1);
    int N = w.size(0);

    auto stride_A = cutlass::make_cute_packed_stride(StrideA_NN{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB_NN{}, cute::make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC_NN{}, cute::make_shape(M, N, 1));

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    typename GemmDevice_Fwd::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            static_cast<cutlass::bfloat16_t*>(x.data_ptr()),
            stride_A,
            static_cast<cutlass::bfloat16_t*>(w.data_ptr()),
            stride_B,
        },
        {
            {1.0f, 0.0f},
            static_cast<cutlass::bfloat16_t*>(out.data_ptr()), stride_C,
            static_cast<cutlass::bfloat16_t*>(out.data_ptr()), stride_C,
        }
    };

    GemmDevice_Fwd gemm_op;
    size_t ws_size = GemmDevice_Fwd::get_workspace_size(args);
    auto& workspace = get_workspace(g_ws_fwd, ws_size, x.device());

    static bool fwd_validated = false;
    if (!fwd_validated) {
        auto status = gemm_op.can_implement(args);
        TORCH_CHECK(status == cutlass::Status::kSuccess,
            "CUTLASS gemm_fwd can_implement failed: ", cutlass::cutlassGetStatusString(status));
        fwd_validated = true;
    }

    auto status = gemm_op.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS gemm_fwd initialize failed: ", cutlass::cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS gemm_fwd run failed: ", cutlass::cutlassGetStatusString(status));
}


// ── dgrad: dX[M,N] = dY[M,K] @ W_t_contig[N,K]^T ──────────────────
void gemm_dgrad(torch::Tensor dy, torch::Tensor w_t_contig, torch::Tensor dx) {
    int M = dy.size(0);
    int K = dy.size(1);
    int N = w_t_contig.size(0);

    auto stride_A = cutlass::make_cute_packed_stride(StrideA_NN{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB_NN{}, cute::make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC_NN{}, cute::make_shape(M, N, 1));

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    typename GemmDevice_DGrad::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            static_cast<cutlass::bfloat16_t*>(dy.data_ptr()),
            stride_A,
            static_cast<cutlass::bfloat16_t*>(w_t_contig.data_ptr()),
            stride_B,
        },
        {
            {1.0f, 0.0f},
            static_cast<cutlass::bfloat16_t*>(dx.data_ptr()), stride_C,
            static_cast<cutlass::bfloat16_t*>(dx.data_ptr()), stride_C,
        }
    };

    GemmDevice_DGrad gemm_op;
    size_t ws_size = GemmDevice_DGrad::get_workspace_size(args);
    auto& workspace = get_workspace(g_ws_dgrad, ws_size, dy.device());

    static bool dgrad_validated = false;
    if (!dgrad_validated) {
        auto status = gemm_op.can_implement(args);
        TORCH_CHECK(status == cutlass::Status::kSuccess,
            "CUTLASS gemm_dgrad can_implement failed: ", cutlass::cutlassGetStatusString(status));
        dgrad_validated = true;
    }

    auto status = gemm_op.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS gemm_dgrad initialize failed: ", cutlass::cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS gemm_dgrad run failed: ", cutlass::cutlassGetStatusString(status));
}


// ── wgrad: dW[M,N] = dY^T[M,K] @ X_t[N,K]^T  →  FP32 output ──────
void gemm_wgrad(torch::Tensor dy_2d, torch::Tensor x_t_contig, torch::Tensor dw) {
    int K = dy_2d.size(0);
    int M = dy_2d.size(1);
    int N = x_t_contig.size(0);

    auto stride_A = cutlass::make_cute_packed_stride(StrideA_TN{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB_TN{}, cute::make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC_TN{}, cute::make_shape(M, N, 1));

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    typename GemmDevice_WGrad::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            static_cast<cutlass::bfloat16_t*>(dy_2d.data_ptr()),
            stride_A,
            static_cast<cutlass::bfloat16_t*>(x_t_contig.data_ptr()),
            stride_B,
        },
        {
            {1.0f, 0.0f},
            static_cast<float*>(dw.data_ptr()), stride_C,
            static_cast<float*>(dw.data_ptr()), stride_C,
        }
    };

    GemmDevice_WGrad gemm_op;
    size_t ws_size = GemmDevice_WGrad::get_workspace_size(args);
    auto& workspace = get_workspace(g_ws_wgrad, ws_size, dy_2d.device());

    static bool wgrad_validated = false;
    if (!wgrad_validated) {
        auto status = gemm_op.can_implement(args);
        TORCH_CHECK(status == cutlass::Status::kSuccess,
            "CUTLASS gemm_wgrad can_implement failed: ", cutlass::cutlassGetStatusString(status));
        wgrad_validated = true;
    }

    auto status = gemm_op.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS gemm_wgrad initialize failed: ", cutlass::cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS gemm_wgrad run failed: ", cutlass::cutlassGetStatusString(status));
}

void gemm_bwd(torch::Tensor dy, torch::Tensor b_dgrad, torch::Tensor dx,
              torch::Tensor b_wgrad, torch::Tensor dw) {
    gemm_dgrad(dy, b_dgrad, dx);
    gemm_wgrad(dy, b_wgrad, dw);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_fwd",   &gemm_fwd,   "CUTLASS SM90 gemm_output fwd (BF16->BF16)");
    m.def("gemm_dgrad", &gemm_dgrad, "CUTLASS SM90 gemm_output dgrad (BF16->BF16)");
    m.def("gemm_wgrad", &gemm_wgrad, "CUTLASS SM90 gemm_output wgrad (BF16->FP32)");
    m.def("gemm_bwd",   &gemm_bwd,   "CUTLASS SM90 gemm_output combined dgrad+wgrad");
}
