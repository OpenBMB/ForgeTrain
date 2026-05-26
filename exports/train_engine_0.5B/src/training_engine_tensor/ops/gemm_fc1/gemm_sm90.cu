/*
 * CUTLASS 3.x SM90 GEMM for gemm_fc1 — v2 performance-tuned.
 *
 * Changes from v1:
 *   - Explicit hw_info (SM count) for proper persistent scheduling
 *   - L2 rasterization swizzle (max_swizzle_size=8)
 *   - Kept Cooperative schedule (Pingpong regresses on large tiles)
 *
 * Three directions:
 *   fwd:   D[M,N] = A[M,K] * B[N,K]  -- A RowMaj, B ColMaj,  D RowMaj BF16
 *   dgrad: D[M,N] = A[M,K] * B[N,K]  -- A RowMaj, B RowMaj,  D RowMaj BF16
 *   wgrad: D[M,N] = A[M,K] * B[N,K]  -- A ColMaj, B ColMaj,  D RowMaj FP32
 */

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

using namespace cute;

using ElementInput  = cutlass::bfloat16_t;
using ElementAcc    = float;
using ElementEpi    = float;

// ============================================================================
// Forward GEMM: Y = X @ W^T       M=40960 N=8192 K=1024
//   Cooperative + cluster 2x1x1 for TMA multicast of B tiles.
// ============================================================================

namespace fwd {
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
using ElementD = cutlass::bfloat16_t;

static constexpr int AlignAB = 128 / cutlass::sizeof_bits<ElementInput>::value;
static constexpr int AlignD  = 128 / cutlass::sizeof_bits<ElementD>::value;

using TileShape    = Shape<_128, _256, _64>;
using ClusterShape = Shape<_2, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementEpi,
    ElementD, LayoutD, AlignD,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecializedCooperative
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementInput, LayoutA, AlignAB,
    ElementInput, LayoutB, AlignAB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
} // namespace fwd

// ============================================================================
// Dgrad GEMM: dX = dY @ W        M=40960 N=1024 K=8192
//   Cooperative + cluster 2x1x1 for TMA multicast of B (weight).
// ============================================================================

namespace dgrad {
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using ElementD = cutlass::bfloat16_t;

static constexpr int AlignAB = 128 / cutlass::sizeof_bits<ElementInput>::value;
static constexpr int AlignD  = 128 / cutlass::sizeof_bits<ElementD>::value;

// N=1024: use 128x256 tile for 2 MMA warp groups (atom_layout 2,1,1).
// 1024/256 = 4 N-tiles, cluster 1x1 avoids unnecessary multicast overhead.
using TileShape    = Shape<_128, _256, _64>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementEpi,
    ElementD, LayoutD, AlignD,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecializedCooperative
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementInput, LayoutA, AlignAB,
    ElementInput, LayoutB, AlignAB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
} // namespace dgrad

// ============================================================================
// Wgrad GEMM: dW = dY^T @ X      M=8192 N=1024 K=40960
//   Cooperative + cluster 1x1x1.
// ============================================================================

namespace wgrad {
using LayoutA = cutlass::layout::ColumnMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using ElementD = float;

static constexpr int AlignAB = 128 / cutlass::sizeof_bits<ElementInput>::value;
static constexpr int AlignD  = 128 / cutlass::sizeof_bits<ElementD>::value;

// M=8192 N=1024 K=40960: use 128x256 tile for 2 MMA warp groups.
// K=40960 → 640 K-steps per tile, so even 256 tiles = enough work per SM.
using TileShape    = Shape<_128, _256, _64>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementEpi,
    ElementD, LayoutD, AlignD,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecializedCooperative
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementInput, LayoutA, AlignAB,
    ElementInput, LayoutB, AlignAB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
} // namespace wgrad

// ============================================================================
// Cached workspace + HW info
// ============================================================================

static torch::Tensor g_ws_fwd;
static torch::Tensor g_ws_dgrad;
static torch::Tensor g_ws_wgrad;
static int g_sm_count = 0;

static int get_sm_count() {
    if (g_sm_count == 0) {
        int device_id = 0;
        cudaGetDevice(&device_id);
        cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, device_id);
    }
    return g_sm_count;
}

template <typename GemmOp>
void* ensure_workspace(torch::Tensor& cached, const typename GemmOp::Arguments& args) {
    size_t needed = GemmOp::get_workspace_size(args);
    if (needed == 0) return nullptr;
    if (!cached.defined() || cached.numel() < static_cast<int64_t>(needed)) {
        cached = torch::empty({static_cast<int64_t>(needed)},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    }
    return cached.data_ptr();
}

// ============================================================================
// Run CUTLASS GEMM with hw_info and L2 swizzle
// ============================================================================

template <typename GemmOp>
void run_gemm(
    const void* ptr_A, const void* ptr_B, void* ptr_D,
    int M, int N, int K,
    torch::Tensor& ws_cache,
    cudaStream_t stream)
{
    using Kernel = typename GemmOp::GemmKernel;
    using StrideA = typename Kernel::StrideA;
    using StrideB = typename Kernel::StrideB;
    using StrideD = typename Kernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    using ElementA = typename Kernel::CollectiveMainloop::ElementA;
    using ElementB = typename Kernel::CollectiveMainloop::ElementB;
    using ElementD = typename Kernel::CollectiveEpilogue::ElementD;

    typename GemmOp::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = cute::make_shape(M, N, K, 1);

    args.mainloop.ptr_A = static_cast<const ElementA*>(ptr_A);
    args.mainloop.dA = stride_A;
    args.mainloop.ptr_B = static_cast<const ElementB*>(ptr_B);
    args.mainloop.dB = stride_B;

    args.epilogue.thread = {ElementEpi(1.0f), ElementEpi(0.0f)};
    args.epilogue.ptr_C = nullptr;
    args.epilogue.dC = stride_D;
    args.epilogue.ptr_D = static_cast<ElementD*>(ptr_D);
    args.epilogue.dD = stride_D;

    int device_id = 0;
    cudaGetDevice(&device_id);
    args.hw_info.device_id = device_id;
    args.hw_info.sm_count = get_sm_count();
    args.scheduler.max_swizzle_size = 8;

    void* ws_ptr = ensure_workspace<GemmOp>(ws_cache, args);

    GemmOp gemm_op;

    auto status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS cannot implement: ", cutlass::cutlassGetStatusString(status));

    status = gemm_op.initialize(args, ws_ptr, stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS init: ", cutlass::cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS run: ", cutlass::cutlassGetStatusString(status));
}

// ============================================================================
// Python-facing functions
// ============================================================================

void gemm_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda());
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16);
    TORCH_CHECK(out.dtype() == torch::kBFloat16);
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous());

    int M = x.size(0), K = x.size(1), N = w.size(0);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    run_gemm<fwd::Gemm>(x.data_ptr(), w.data_ptr(), out.data_ptr(),
                         M, N, K, g_ws_fwd, stream);
}

void gemm_dgrad(torch::Tensor dy, torch::Tensor w, torch::Tensor dx) {
    TORCH_CHECK(dy.is_cuda() && w.is_cuda() && dx.is_cuda());
    TORCH_CHECK(dy.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16);
    TORCH_CHECK(dx.dtype() == torch::kBFloat16);
    TORCH_CHECK(dy.is_contiguous() && w.is_contiguous() && dx.is_contiguous());

    int M = dy.size(0), K = dy.size(1), N = w.size(1);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    run_gemm<dgrad::Gemm>(dy.data_ptr(), w.data_ptr(), dx.data_ptr(),
                           M, N, K, g_ws_dgrad, stream);
}

void gemm_wgrad(torch::Tensor dy, torch::Tensor x, torch::Tensor dw) {
    TORCH_CHECK(dy.is_cuda() && x.is_cuda() && dw.is_cuda());
    TORCH_CHECK(dy.dtype() == torch::kBFloat16 && x.dtype() == torch::kBFloat16);
    TORCH_CHECK(dw.dtype() == torch::kFloat32);
    TORCH_CHECK(dy.is_contiguous() && x.is_contiguous() && dw.is_contiguous());

    int K = dy.size(0), M = dy.size(1), N = x.size(1);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    run_gemm<wgrad::Gemm>(dy.data_ptr(), x.data_ptr(), dw.data_ptr(),
                           M, N, K, g_ws_wgrad, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_fwd",   &gemm_fwd,   "SM90 cooperative GEMM fwd (cluster 2x1)");
    m.def("gemm_dgrad", &gemm_dgrad, "SM90 cooperative GEMM dgrad (cluster 2x1)");
    m.def("gemm_wgrad", &gemm_wgrad, "SM90 cooperative GEMM wgrad FP32");
}
