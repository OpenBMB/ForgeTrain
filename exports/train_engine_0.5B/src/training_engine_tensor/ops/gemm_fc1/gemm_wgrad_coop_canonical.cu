/*
 * Round 34 (architectural data point #5) — CUTLASS 3.x SM90 cooperative
 * wgrad kernel with the **canonical** tile shape (128, 256, 64) and
 * cluster=(1,1,1).
 *
 *   Direction:   dW = dY^T @ X        M=8192 N=1024 K=40960
 *   Layouts:     A col-major (dY^T), B row-major (X), D row-major
 *   Dtypes:      A/B BF16, C/D FP32 (drop-in compatible with existing wgrad)
 *
 * Architecture: SM90 GEMM with `KernelTmaWarpSpecializedCooperative` mainloop
 * + `TmaWarpSpecializedCooperative` epilogue + the **default**
 * `PersistentScheduler` (data-parallel — no stream-K, no split-K).
 *
 * Rationale:
 *   R30 (stream-K), R31 (PingPong), R32 (tile rotation 256x128) and R33
 *   (cluster=(2,2,1)) all closed off variations on the CUTLASS scheduler
 *   / cluster / tile-rotation axes.  The remaining axis under
 *   cooperative+data-parallel+canonical-tile is the tile **swizzle size**
 *   (`max_swizzle_size`).  CUTLASS's default heuristic picks swizzle=8
 *   for many shapes, while CuTeDSL's wgrad export uses swizzle=4 because
 *   archive R47 measured swizzle=8 was 2× slower on CuTeDSL wgrad.  The
 *   open question: is that swizzle=8 regression CuTeDSL-specific (e.g.,
 *   builder-driven SMEM layout choices), or does it hold for CUTLASS
 *   native cooperative as well?
 *
 *   This kernel exposes `max_swizzle_size` via the
 *   `GEMM_FC1_WGRAD_SWIZZLE` env var, so we can sweep swizzle at the
 *   exact tile/cluster CuTeDSL is using and either confirm CuTeDSL's
 *   choice or find a different optimum on the CUTLASS path.
 *
 *   Secondary purpose: this file establishes a clean apples-to-apples
 *   "CUTLASS native cooperative @ canonical tile" baseline.  R32's
 *   coop_rot.cu had the rotated tile as a confound; this file removes
 *   that confound so any difference vs CuTeDSL is purely framework /
 *   compilation-path noise.
 *
 * Implemented as a separate torch extension parallel to streamk /
 * pingpong / coop_rot / coop_2x2 so it can be selectively engaged via
 * `GEMM_FC1_WGRAD_BACKEND=coop_canonical` while keeping the production
 * CuTeDSL fwd/dgrad/wgrad path unchanged by default.
 */

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

using namespace cute;

namespace wgrad_coop_canonical {

using ElementInput = cutlass::bfloat16_t;
using ElementAcc   = float;
using ElementEpi   = float;
using ElementD     = float;

// Layout choice mirrors the other gemm_wgrad_*.cu variants.
//   A = dy^T : col-major over [M=8192, K=40960]
//   B = X    : row-major over [K=40960, N=1024]
//   D = dW   : row-major over [M=8192, N=1024]
using LayoutA = cutlass::layout::ColumnMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignAB =
    128 / cutlass::sizeof_bits<ElementInput>::value;  // 8
static constexpr int AlignD  =
    128 / cutlass::sizeof_bits<ElementD>::value;      // 4

// Canonical tile = (128, 256, 64).  In cooperative mode the two warpgroups
// co-own the same output tile and split it along M (each WG handles 64
// rows).  Accumulator footprint per CTA = 128*256*4 = 128 KB FP32, split
// across 2 WGs ⇒ 64 KB per WG (~64 regs/thread, no spill risk).
//
// Tile count: 8192/128 × 1024/256 = 64 × 4 = 256 tiles total.
// 256 tiles / 132 SMs ≈ 1.94 waves (same as CuTeDSL default).
using TileShape    = Shape<_128, _256, _64>;
using ClusterShape = Shape<_1, _1, _1>;

using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, ElementEpi,
        ElementD, LayoutD, AlignD,
        ElementD, LayoutD, AlignD,
        EpilogueSchedule>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementInput, LayoutA, AlignAB,
        ElementInput, LayoutB, AlignAB,
        ElementAcc,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
    /* default tile scheduler = PersistentScheduler (data-parallel) */>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}  // namespace wgrad_coop_canonical

// ============================================================================
// Cached workspace + HW info
// ============================================================================

static torch::Tensor g_ws_coop_canonical;
static int g_sm_count = 0;

static int get_sm_count() {
    if (g_sm_count == 0) {
        int device_id = 0;
        cudaGetDevice(&device_id);
        cudaDeviceGetAttribute(&g_sm_count,
                               cudaDevAttrMultiProcessorCount,
                               device_id);
    }
    return g_sm_count;
}

template <typename GemmOp>
static void* ensure_workspace(torch::Tensor& cached,
                              const typename GemmOp::Arguments& args) {
    size_t needed = GemmOp::get_workspace_size(args);
    if (needed == 0) return nullptr;
    if (!cached.defined() || cached.numel() < static_cast<int64_t>(needed)) {
        cached = torch::empty({static_cast<int64_t>(needed)},
            torch::TensorOptions()
                .dtype(torch::kUInt8)
                .device(torch::kCUDA));
    }
    return cached.data_ptr();
}

// ============================================================================
// Cooperative + canonical-tile wgrad — same external contract as gemm_wgrad
// ============================================================================

void gemm_wgrad_coop_canonical(torch::Tensor dy, torch::Tensor x,
                               torch::Tensor dw, int64_t swizzle) {
    TORCH_CHECK(dy.is_cuda() && x.is_cuda() && dw.is_cuda());
    TORCH_CHECK(dy.dtype() == torch::kBFloat16 &&
                x.dtype() == torch::kBFloat16);
    TORCH_CHECK(dw.dtype() == torch::kFloat32);
    TORCH_CHECK(dy.is_contiguous() && x.is_contiguous() && dw.is_contiguous());

    int K = dy.size(0);  // 40960  (S*B)
    int M = dy.size(1);  // 8192
    int N = x.size(1);   // 1024
    TORCH_CHECK(x.size(0) == K, "wgrad shape mismatch on K (dy.size(0) vs x.size(0))");
    TORCH_CHECK(dw.size(0) == M && dw.size(1) == N,
                "wgrad output expects [", M, ",", N, "], got [",
                dw.size(0), ",", dw.size(1), "]");

    using GemmOp = wgrad_coop_canonical::Gemm;
    using Kernel = GemmOp::GemmKernel;
    using StrideA = typename Kernel::StrideA;
    using StrideB = typename Kernel::StrideB;
    using StrideD = typename Kernel::StrideD;

    // dY logical [K_w=K, M_w=M], col-major along M axis (a col-major)
    auto stride_A = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(M, K, 1));
    // X logical [K_w=K, N_w=N], row-major along N axis (b row-major)
    auto stride_B = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(N, K, 1));
    // dW [M, N] row-major
    auto stride_D = cutlass::make_cute_packed_stride(
        StrideD{}, cute::make_shape(M, N, 1));

    using ElementA = typename Kernel::CollectiveMainloop::ElementA;
    using ElementB = typename Kernel::CollectiveMainloop::ElementB;
    using ElementD = typename Kernel::CollectiveEpilogue::ElementD;

    typename GemmOp::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = cute::make_shape(M, N, K, 1);

    args.mainloop.ptr_A = static_cast<const ElementA*>(dy.data_ptr());
    args.mainloop.dA = stride_A;
    args.mainloop.ptr_B = static_cast<const ElementB*>(x.data_ptr());
    args.mainloop.dB = stride_B;

    using ElementEpi = wgrad_coop_canonical::ElementEpi;
    args.epilogue.thread = {ElementEpi(1.0f), ElementEpi(0.0f)};
    args.epilogue.ptr_C = nullptr;  // beta = 0
    args.epilogue.dC = stride_D;
    args.epilogue.ptr_D = static_cast<ElementD*>(dw.data_ptr());
    args.epilogue.dD = stride_D;

    int device_id = 0;
    cudaGetDevice(&device_id);
    args.hw_info.device_id = device_id;
    args.hw_info.sm_count = get_sm_count();

    // Default PersistentScheduler — knob is `max_swizzle_size`.
    //
    // CUTLASS valid values are 1, 2, 4, 8.  Archive R47 measured CuTeDSL
    // swizzle=8 caused a 2× wgrad regression (likely due to CuTeDSL's
    // builder-specific SMEM layout interaction with swizzle), so CuTeDSL
    // settled on swizzle=4.  We make this configurable at runtime so we
    // can sweep on CUTLASS native and either confirm CuTeDSL's choice or
    // find a different optimum.
    int sw = static_cast<int>(swizzle);
    TORCH_CHECK(sw == 1 || sw == 2 || sw == 4 || sw == 8,
                "GEMM_FC1_WGRAD_SWIZZLE must be one of {1,2,4,8}, got ", sw);
    args.scheduler.max_swizzle_size = sw;

    void* ws_ptr = ensure_workspace<GemmOp>(g_ws_coop_canonical, args);

    GemmOp gemm_op;
    auto status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "coop_canonical wgrad cannot implement: ",
                cutlass::cutlassGetStatusString(status));

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    status = gemm_op.initialize(args, ws_ptr, stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "coop_canonical wgrad init: ",
                cutlass::cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "coop_canonical wgrad run: ",
                cutlass::cutlassGetStatusString(status));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_wgrad_coop_canonical",
          &gemm_wgrad_coop_canonical,
          "SM90 cooperative GEMM wgrad with canonical tile (128,256,64), "
          "cluster=(1,1,1), data-parallel persistent scheduler, and "
          "runtime-configurable max_swizzle_size in {1,2,4,8}",
          pybind11::arg("dy"), pybind11::arg("x"), pybind11::arg("dw"),
          pybind11::arg("swizzle") = 4);
}
