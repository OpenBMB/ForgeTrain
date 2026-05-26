/*
 * C++ wrapper for CuTeDSL-exported GEMM kernels.
 *
 * This links against .o files exported by CuTeDSL's export_to_c(),
 * providing CuTeDSL kernel performance with zero Python dispatch overhead.
 *
 * Tensor struct layout (from export headers):
 *   { void *data; int32_t dynamic_shapes[3]; int64_t dynamic_strides[2]; }
 *
 * Directions:
 *   fwd:   C[M,N]   = A[M,K]   * B[N,K]    BF16→BF16
 *   dgrad: C[M,N]   = A[M,K]   * B[N,K]    BF16→BF16
 *   wgrad: C[M,N]   = A[M,K]   * B[N,K]    BF16→FP32
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <functional>

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

/* ============================================================================
 * Round 42: side-stream + reusable events for parallel dgrad/wgrad in
 * cutedsl_gemm_bwd_fast.
 *
 * Both bwd kernels are persistent (grid_size = SM_count), but the wgrad
 * kernel runs ~1.95 ms versus dgrad's ~30-40 µs.  On a single stream the
 * GPU scheduler still tucks dgrad into wgrad's tail wave **sometimes** —
 * which is exactly why we observe the ~30-40 µs bimodal spread between
 * "fast" (1.945 ms) and "slow" (1.985 ms) bwd samples in
 * `bench_against_production.py`.  By moving dgrad onto a non-blocking
 * side stream we let the runtime overlap it with wgrad's mainloop +
 * tail wave deterministically, which collapses the bimodal distribution
 * onto the fast cluster and (empirically) shaves ~25 µs / call off the
 * bwd median.
 *
 * The side stream + events are created lazily on first use and never
 * destroyed (process-lifetime singletons, like the kernel modules above).
 * Using `cudaEventDisableTiming` makes record / wait essentially free
 * (~2 µs each on H100).  Synchronization pattern:
 *
 *     cudaEventRecord(evt_main_to_side, main);    // main publishes inputs
 *     cudaStreamWaitEvent(side, evt_main_to_side, 0);
 *     launch_dgrad(side);
 *     launch_wgrad(main);
 *     cudaEventRecord(evt_side_to_main, side);    // side publishes d_input
 *     cudaStreamWaitEvent(main, evt_side_to_main, 0);
 *
 * Either kernel writes a disjoint output (d_input via dgrad, d_weight via
 * wgrad) and reads independent input bytes (dY/W vs dY/X), so there is
 * no RAW/WAW dependency between the two and reordering across streams
 * is bitwise-safe.  Caller's main stream is the one autograd dispatches
 * subsequent ops on; both grads must be visible there before we return,
 * which the trailing `cudaStreamWaitEvent(main, ...)` guarantees.
 *
 * Toggle via `GEMM_FC1_BWD_PARALLEL`:
 *   unset / "1" → parallel (default, R42 production)
 *   "0"          → serial   (R39/R41 ordering, opt-in for ablation)
 * Reads once per process (static const) — flipping mid-process is not
 * supported (matches the pre-existing `GEMM_FC1_BWD_ORDER` semantics).
 * ============================================================================ */

static bool g_parallel_inited = false;
static cudaStream_t g_side_stream = nullptr;
static cudaStream_t g_capture_stream = nullptr;  // R44: dedicated stream for cudaGraph capture
static cudaEvent_t  g_evt_main_to_side = nullptr;
static cudaEvent_t  g_evt_side_to_main = nullptr;

/* ============================================================================
 * Round 47: pointer-keyed cudaGraph cache for the fwd fast path.
 *
 * R45 (E) Nsight Systems profiling already proved bwd's slow cluster is an
 * HBM/DVFS throttle floor that no kernel-side change can move; R44's reverse-
 * trap analysis additionally rules out any host-side scheduling tightening on
 * the bwd path (graph capture / priority bump / cuLaunchKernelEx all collapse
 * the bimodal distribution onto the slow cluster).  R45's close-out missed
 * one independent lever: fwd is a SINGLE persistent kernel — there is no
 * dgrad+wgrad scheduling competition, so a fwd-only cudaGraph capture is
 * outside R44's reverse-trap envelope and only buys plain ~20 µs/call host
 * dispatch overhead reduction (cudaLaunchKernel → cudaGraphLaunch).
 *
 * Cache strategy: PyTorch's caching allocator returns a small set of stable
 * addresses for the same-shape `torch::empty` allocations in a steady-state
 * loop (fwd output buffer in our case).  Inputs X / W come from upstream
 * code — X cycles through ~2 addresses per micro-batch (fwd + activation
 * checkpointing reuse), W is a parameter (fixed address per process).
 *
 * On the first call with a new (X_ptr, W_ptr, Y_ptr) tuple we:
 *   1. cudaStreamBeginCapture on a private capture stream (synced into via
 *      an event from `main`).
 *   2. Replay the kernel launch via the existing cute_dsl_gemm_fwd_wrapper
 *      (which inside capture mode records the cudaLaunchKernel into the
 *      graph instead of submitting it to the driver).
 *   3. cudaStreamEndCapture → cudaGraphInstantiate → cache the
 *      cudaGraphExec_t under the pointer triple.
 *   4. cudaGraphLaunch on `main` with the cached exec.
 *
 * On subsequent calls with the same triple we just cudaGraphLaunch — saving
 * ~20 µs/call vs the regular submission path (the 11-rep median drop the
 * bench cares about).
 *
 * Cache is bounded at FWD_GRAPH_CACHE_MAX entries; if we hit the limit we
 * evict an arbitrary entry (allocator only ever cycles 2-3 distinct Y
 * addresses in practice, so 16 is well over what we need but cheap).
 *
 * Default OFF via `GEMM_FC1_FWD_GRAPH=0`.  R47 close-out finding: the
 * implementation is correct (FP64 PASS, cache hit rate 200/200 in a
 * 200-call burn-in — PyTorch caching allocator returns one stable Y
 * address) but the predicted ~20 µs/call host overhead reduction did
 * not materialise.  Probe (`probe_fwd_graph.py`):
 *   GRAPH OFF fwd-only median = 1.0514 ms
 *   GRAPH ON  fwd-only median = 1.0536 ms  (2 µs slower / noise)
 * R37 (single-pybind11) / R39 (combined bwd) / R40 (module-level callable
 * cache) had already squeezed CuTeDSL host dispatch down to ~3-5 µs/call,
 * which is right at the cudaGraphLaunch floor — so capturing into a graph
 * cannot beat direct submission for this kernel.  Set `=1` for ablation.
 * Read once per process (matches `parallel_choice` / `order_choice` /
 * `graph_choice` semantics — flipping mid-process unsupported).
 * ============================================================================ */

static cudaStream_t g_fwd_capture_stream = nullptr;
static cudaEvent_t  g_fwd_pre_evt        = nullptr;
static bool         g_fwd_graph_inited   = false;

struct FwdGraphKey {
    void* x_ptr;
    void* w_ptr;
    void* y_ptr;
    bool operator==(const FwdGraphKey& o) const noexcept {
        return x_ptr == o.x_ptr && w_ptr == o.w_ptr && y_ptr == o.y_ptr;
    }
};
struct FwdGraphKeyHash {
    size_t operator()(const FwdGraphKey& k) const noexcept {
        size_t h = std::hash<void*>{}(k.x_ptr);
        h ^= std::hash<void*>{}(k.w_ptr) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<void*>{}(k.y_ptr) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
static std::unordered_map<FwdGraphKey, cudaGraphExec_t, FwdGraphKeyHash>
    g_fwd_graph_cache;
static constexpr size_t FWD_GRAPH_CACHE_MAX = 16;

static void ensure_fwd_graph_init() {
    if (g_fwd_graph_inited) return;
    cudaError_t err = cudaStreamCreateWithFlags(
        &g_fwd_capture_stream, cudaStreamNonBlocking);
    TORCH_CHECK(err == cudaSuccess,
                "fwd capture stream create failed: ", cudaGetErrorString(err));
    err = cudaEventCreateWithFlags(&g_fwd_pre_evt, cudaEventDisableTiming);
    TORCH_CHECK(err == cudaSuccess,
                "fwd pre evt create failed: ", cudaGetErrorString(err));
    g_fwd_graph_inited = true;
}

/* Round 44: cudaGraph capture cache for the parallel bwd sequence
 * (R42's default).  When `GEMM_FC1_BWD_GRAPH=1` is set, the bwd_fast
 * function captures the 6-op sequence (2 events + 2 kernel launches +
 * 2 stream waits) into a graph on the first call, then on every
 * subsequent call recaptures into a temporary graph and asks the runtime
 * to update the cached executable in-place via `cudaGraphExecUpdate`.
 *
 * The cached `g_bwd_graph_exec` is a process-lifetime singleton (same
 * lifetime as `g_side_stream` / events).  If `cudaGraphExecUpdate` ever
 * fails (e.g. topology drift due to a future code change), we fall back
 * to a fresh `cudaGraphInstantiate` so the call still completes cleanly.
 *
 * Default OFF — Round 44 is an opt-in experiment to measure whether the
 * runtime executes a 6-op CUDA graph more efficiently (lower-latency
 * inter-op gaps, deterministic dispatch order) than the equivalent
 * native stream sequence.  The R42 path is unchanged when env=0.
 */
static cudaGraphExec_t g_bwd_graph_exec = nullptr;

/* Round 43: tested side-stream at highest available CUDA priority — opt-in.
 *
 * Hypothesis (R42 → R43): bumping the side stream to the device's highest
 * priority would bias the H100 block scheduler toward dispatching dgrad's
 * leading CTAs early, occupying SMs that wgrad has not yet locked down
 * during its prologue / tail wave, and thus shifting the bimodal candidate-
 * bwd distribution onto the fast cluster more reliably than R42's
 * default-priority pair.
 *
 * Outcome: NEGATIVE.  On shandongdev-297757 with priority enabled we
 * observed the OPPOSITE effect — every sample of `cust_bwd` collapsed
 * onto the slow cluster (2.020-2.029 ms across all 11 reps, two
 * back-to-back runs), pushing total ratio to 1.005-1.030× and
 * decisively failing the 0.985× gate.  Priority=0 (R42 default) at
 * least produced 4/11 fast samples in the same process.
 *
 * Physical interpretation: H100 stream priority on persistent kernels
 * is advisory at the *block-dispatch* level — CTAs cannot preempt once
 * issued.  When wgrad and dgrad both target a saturating CTA count
 * (1280 dgrad tiles ≈ 9.7 waves vs wgrad's ~1.94 waves on 132 SMs),
 * raising side priority causes the scheduler to launch dgrad's CTAs
 * *first* on all 132 SMs, which then *blocks* wgrad's CTAs from
 * starting until dgrad finishes — yielding fully serial execution
 * (slow cluster) rather than the partial overlap that the equal-priority
 * scheduler sometimes achieves during wgrad's prologue / tail wave.
 *
 * Decision: keep the code path as opt-in via `GEMM_FC1_SIDE_PRIORITY=1`
 * for future ablation / instrumentation, but the **default is `0`**
 * (R42 unchanged behaviour: both streams at default priority).  Code
 * stays in the cpp file because this is a useful future reference for
 * any agent considering the same experiment — and avoids re-doing the
 * same negative measurement in a fresh run.
 */
static void ensure_parallel_init() {
    if (g_parallel_inited) return;
    static const int side_priority_choice = []() {
        const char *env = std::getenv("GEMM_FC1_SIDE_PRIORITY");
        return (env && std::string(env) == "1") ? 1 : 0;
    }();
    cudaError_t err;
    if (side_priority_choice == 1) {
        int leastPriority = 0, greatestPriority = 0;
        err = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaDeviceGetStreamPriorityRange failed: ",
                    cudaGetErrorString(err));
        err = cudaStreamCreateWithPriority(
            &g_side_stream, cudaStreamNonBlocking, greatestPriority);
    } else {
        err = cudaStreamCreateWithFlags(&g_side_stream, cudaStreamNonBlocking);
    }
    TORCH_CHECK(err == cudaSuccess, "side stream create failed: ", cudaGetErrorString(err));
    err = cudaEventCreateWithFlags(&g_evt_main_to_side, cudaEventDisableTiming);
    TORCH_CHECK(err == cudaSuccess, "evt_main_to_side create failed: ", cudaGetErrorString(err));
    err = cudaEventCreateWithFlags(&g_evt_side_to_main, cudaEventDisableTiming);
    TORCH_CHECK(err == cudaSuccess, "evt_side_to_main create failed: ", cudaGetErrorString(err));
    // R44: dedicated capture stream (always created so we don't pay the
    // first-call overhead on the bwd hot path).  Non-blocking so it doesn't
    // implicitly serialise against the legacy default stream.
    err = cudaStreamCreateWithFlags(&g_capture_stream, cudaStreamNonBlocking);
    TORCH_CHECK(err == cudaSuccess,
                "capture stream create failed: ", cudaGetErrorString(err));
    g_parallel_inited = true;
}

/* ---------- fwd: Y = X @ W^T ----------
 * Round 46: fwd is exported with is_dynamic_layout=False, so the descriptor
 * is just { void* data; }.  All shapes/strides are baked into the kernel
 * symbol at compile time.  Saves ~70µs/call (8% faster) vs dynamic layout.
 *
 * Shapes locked in by export_kernels.py:
 *   X[40960, 1024] BF16 row-major   → A[M,K,1] = (40960,1024,1) stride (1024,1,40960*1024)
 *   W[8192,  1024] BF16 row-major   → B[N,K,1] = (8192, 1024,1) stride (1024,1, 8192*1024)
 *   Y[40960, 8192] BF16 row-major   → C[M,N,1] = (40960,8192,1) stride (8192,1,40960*8192)
 */
void cutedsl_gemm_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    ensure_init();
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda());
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16);
    TORCH_CHECK(out.dtype() == torch::kBFloat16);
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous());
    TORCH_CHECK(x.size(0) == 40960 && x.size(1) == 1024,
                "fwd: expected X[40960,1024], got [", x.size(0), ",", x.size(1), "]");
    TORCH_CHECK(w.size(0) == 8192 && w.size(1) == 1024,
                "fwd: expected W[8192,1024], got [", w.size(0), ",", w.size(1), "]");
    TORCH_CHECK(out.size(0) == 40960 && out.size(1) == 8192,
                "fwd: expected Y[40960,8192], got [", out.size(0), ",", out.size(1), "]");

    gemm_fwd_Tensor_a_t a_desc = { x.data_ptr() };
    gemm_fwd_Tensor_b_t b_desc = { w.data_ptr() };
    gemm_fwd_Tensor_c_t c_desc = { out.data_ptr() };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int32_t ret = cute_dsl_gemm_fwd_wrapper(&g_fwd_module, &a_desc, &b_desc, &c_desc, stream);
    TORCH_CHECK(ret == 0, "CuTeDSL fwd kernel failed with code ", ret);
}

/* ---------- dgrad: dX = dY @ W ----------
 * Round 46: dgrad is also exported with is_dynamic_layout=False (~10µs gain).
 *
 * Shapes locked in by export_kernels.py:
 *   dY[40960, 8192] BF16 row-major  → A[M,K,1] = (40960,8192,1) stride (8192,1,40960*8192)
 *   W [8192,  1024] BF16 row-major  → B[N,K,1] = (1024, 8192,1) stride (1,1024,1024*8192) (b_major=n)
 *   dX[40960, 1024] BF16 row-major  → C[M,N,1] = (40960,1024,1) stride (1024,1,40960*1024)
 */
void cutedsl_gemm_dgrad(torch::Tensor dy, torch::Tensor w, torch::Tensor dx) {
    ensure_init();
    TORCH_CHECK(dy.is_cuda() && w.is_cuda() && dx.is_cuda());
    TORCH_CHECK(dy.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16);
    TORCH_CHECK(dx.dtype() == torch::kBFloat16);
    TORCH_CHECK(dy.is_contiguous() && w.is_contiguous() && dx.is_contiguous());
    TORCH_CHECK(dy.size(0) == 40960 && dy.size(1) == 8192,
                "dgrad: expected dY[40960,8192], got [", dy.size(0), ",", dy.size(1), "]");
    TORCH_CHECK(w.size(0) == 8192 && w.size(1) == 1024,
                "dgrad: expected W[8192,1024], got [", w.size(0), ",", w.size(1), "]");
    TORCH_CHECK(dx.size(0) == 40960 && dx.size(1) == 1024,
                "dgrad: expected dX[40960,1024], got [", dx.size(0), ",", dx.size(1), "]");

    gemm_dgrad_Tensor_a_t a_desc = { dy.data_ptr() };
    gemm_dgrad_Tensor_b_t b_desc = { w.data_ptr() };
    gemm_dgrad_Tensor_c_t c_desc = { dx.data_ptr() };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int32_t ret = cute_dsl_gemm_dgrad_wrapper(&g_dgrad_module, &a_desc, &b_desc, &c_desc, stream);
    TORCH_CHECK(ret == 0, "CuTeDSL dgrad kernel failed with code ", ret);
}

/* ---------- wgrad: dW = dY^T @ X ----------
 * dY[K_act=40960, M_w=8192] row-major stride=(8192, 1)
 *   → A[M=8192, K=40960, 1] stride=(1, M, M*K) (a_major=m, col-major)
 *   Col-major: stride_0=1(static). dynamic_strides = {stride_1=M, stride_2=M*K}
 * X[K_act=40960, N_w=1024] row-major stride=(1024, 1)
 *   → B[N=1024, K=40960, 1] stride=(1, N, N*K) (b_major=n, col-major)
 *   Col-major: stride_0=1(static). dynamic_strides = {stride_1=N, stride_2=N*K}
 * dW[M=8192, N=1024]
 *   → C[M, N, 1] stride=(N, 1, M*N) (c_major=n, row-major)
 *   Row-major: stride_1=1(static). dynamic_strides = {stride_0=N, stride_2=M*N}
 */
void cutedsl_gemm_wgrad(torch::Tensor dy, torch::Tensor x, torch::Tensor dw) {
    ensure_init();
    TORCH_CHECK(dy.is_cuda() && x.is_cuda() && dw.is_cuda());
    TORCH_CHECK(dy.dtype() == torch::kBFloat16 && x.dtype() == torch::kBFloat16);
    TORCH_CHECK(dw.dtype() == torch::kFloat32);
    TORCH_CHECK(dy.is_contiguous() && x.is_contiguous() && dw.is_contiguous());

    int K = dy.size(0);  // 40960 (seq*batch)
    int M = dy.size(1);  // 8192
    int N = x.size(1);   // 1024

    // A col-major: stride_0=1(static), dynamic = {stride_1=M, stride_2=M*K}
    gemm_wgrad_Tensor_a_t a_desc = {
        dy.data_ptr(), {M, K, 1}, {(int64_t)M, (int64_t)M * K}
    };
    // B col-major: stride_0=1(static), dynamic = {stride_1=N, stride_2=N*K}
    gemm_wgrad_Tensor_b_t b_desc = {
        x.data_ptr(), {N, K, 1}, {(int64_t)N, (int64_t)N * K}
    };
    // C row-major: stride_1=1(static), dynamic = {stride_0=N, stride_2=M*N}
    gemm_wgrad_Tensor_c_t c_desc = {
        dw.data_ptr(), {M, N, 1}, {(int64_t)N, (int64_t)M * N}
    };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int32_t ret = cute_dsl_gemm_wgrad_wrapper(&g_wgrad_module, &a_desc, &b_desc, &c_desc, stream);
    TORCH_CHECK(ret == 0, "CuTeDSL wgrad kernel failed with code ", ret);
}

/* ============================================================================
 * Round 37: "_fast" wrappers that fold .contiguous() / .reshape() /
 * torch.empty() / .view() into a single pybind11 call.
 *
 * These accept the upstream 3-D tensors directly (X[S,B,K] / dY[S,B,N]),
 * make them contiguous + 2-D internally via aten C++ APIs (no Python
 * dispatch), allocate the output tensor with the prefix shape preserved,
 * dispatch the exported CuTeDSL kernel, and return the output.
 *
 * Saves ~5-10 µs/call of Python dispatch overhead (.contiguous, .reshape,
 * torch.empty, .view) which shows up as ~0.5-1 % per-call ratio gain in
 * the back-to-back `bench_against_production.py` measurement.
 *
 * The shape contract (M=40960, N=8192, K=1024) is identical to the
 * non-_fast bindings — they wrap the same exported kernel.
 * ============================================================================ */

torch::Tensor cutedsl_gemm_fwd_fast(torch::Tensor x, torch::Tensor w) {
    ensure_init();
    TORCH_CHECK(x.is_cuda() && w.is_cuda());
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16);

    auto x_c = x.contiguous();
    auto w_c = w.contiguous();

    int64_t K = x_c.size(-1);
    int64_t M = x_c.numel() / K;
    int64_t N = w_c.size(0);

    TORCH_CHECK(M == 40960 && N == 8192 && K == 1024,
                "fwd_fast: expected M=40960 N=8192 K=1024; got M=",
                M, " N=", N, " K=", K);
    TORCH_CHECK(w_c.size(1) == K, "fwd_fast: weight K mismatch");

    // Build output shape: preserve x's leading dims, replace last with N.
    auto sizes = x_c.sizes().vec();
    sizes.back() = N;
    auto out = torch::empty(sizes, x_c.options());

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    // Round 47: pointer-keyed cudaGraph cache.  See file-top comment
    // block "Round 47" for the rationale + close-out finding.  Default
    // OFF after the close-out probe showed graph-ON is 2 µs slower than
    // graph-OFF on the same kernel (CuTeDSL host dispatch already at the
    // ~3-5 µs cudaGraphLaunch floor post R37/R39/R40).  `=1` opt-in for
    // ablation.  Read once per process.
    static const int fwd_graph_choice = []() {
        const char *env = std::getenv("GEMM_FC1_FWD_GRAPH");
        return (env && std::string(env) == "1") ? 1 : 0;
    }();

    if (fwd_graph_choice == 0) {
        // R46 direct-launch path (kept as opt-out for ablation).
        gemm_fwd_Tensor_a_t a_desc = { x_c.data_ptr() };
        gemm_fwd_Tensor_b_t b_desc = { w_c.data_ptr() };
        gemm_fwd_Tensor_c_t c_desc = { out.data_ptr() };
        int32_t ret = cute_dsl_gemm_fwd_wrapper(
            &g_fwd_module, &a_desc, &b_desc, &c_desc, stream);
        TORCH_CHECK(ret == 0, "CuTeDSL fwd kernel failed with code ", ret);
        return out;
    }

    ensure_fwd_graph_init();

    FwdGraphKey key{x_c.data_ptr(), w_c.data_ptr(), out.data_ptr()};
    cudaGraphExec_t exec = nullptr;
    auto it = g_fwd_graph_cache.find(key);

    if (it != g_fwd_graph_cache.end()) {
        exec = it->second;
    } else {
        // Capture into a private stream synced from `main` via event so any
        // upstream work that produced X / W on `main` is observed by the
        // recorded kernel.  (When the captured graph is later launched on
        // `main` directly, this synchronisation is implicit, but capture
        // happens on the capture stream so we make the dependency explicit.)
        cudaError_t err = cudaEventRecord(g_fwd_pre_evt, stream);
        TORCH_CHECK(err == cudaSuccess,
                    "fwd graph pre-evt record failed: ",
                    cudaGetErrorString(err));
        err = cudaStreamWaitEvent(g_fwd_capture_stream, g_fwd_pre_evt, 0);
        TORCH_CHECK(err == cudaSuccess,
                    "fwd graph capture wait failed: ",
                    cudaGetErrorString(err));

        err = cudaStreamBeginCapture(
            g_fwd_capture_stream, cudaStreamCaptureModeRelaxed);
        TORCH_CHECK(err == cudaSuccess,
                    "fwd cudaStreamBeginCapture failed: ",
                    cudaGetErrorString(err));

        gemm_fwd_Tensor_a_t a_desc = { x_c.data_ptr() };
        gemm_fwd_Tensor_b_t b_desc = { w_c.data_ptr() };
        gemm_fwd_Tensor_c_t c_desc = { out.data_ptr() };
        int32_t ret = cute_dsl_gemm_fwd_wrapper(
            &g_fwd_module, &a_desc, &b_desc, &c_desc, g_fwd_capture_stream);
        TORCH_CHECK(ret == 0,
                    "CuTeDSL fwd kernel failed during capture with code ",
                    ret);

        cudaGraph_t graph = nullptr;
        err = cudaStreamEndCapture(g_fwd_capture_stream, &graph);
        TORCH_CHECK(err == cudaSuccess,
                    "fwd cudaStreamEndCapture failed: ",
                    cudaGetErrorString(err));

        // Bound cache size — evict an arbitrary entry on overflow.
        if (g_fwd_graph_cache.size() >= FWD_GRAPH_CACHE_MAX) {
            auto victim = g_fwd_graph_cache.begin();
            cudaGraphExecDestroy(victim->second);
            g_fwd_graph_cache.erase(victim);
        }

        err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        TORCH_CHECK(err == cudaSuccess,
                    "fwd cudaGraphInstantiate failed: ",
                    cudaGetErrorString(err));
        cudaGraphDestroy(graph);

        g_fwd_graph_cache.emplace(key, exec);
    }

    cudaError_t err = cudaGraphLaunch(exec, stream);
    TORCH_CHECK(err == cudaSuccess,
                "fwd cudaGraphLaunch failed: ",
                cudaGetErrorString(err));

    return out;
}

torch::Tensor cutedsl_gemm_dgrad_fast(torch::Tensor dy, torch::Tensor w) {
    ensure_init();
    TORCH_CHECK(dy.is_cuda() && w.is_cuda());
    TORCH_CHECK(dy.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16);

    auto dy_c = dy.contiguous();
    auto w_c = w.contiguous();

    int64_t N_in = dy_c.size(-1);     // 8192
    int64_t M = dy_c.numel() / N_in;   // 40960
    int64_t K_out = w_c.size(1);       // 1024

    TORCH_CHECK(M == 40960 && N_in == 8192 && K_out == 1024,
                "dgrad_fast: expected M=40960 N=8192 K=1024; got M=",
                M, " N_in=", N_in, " K_out=", K_out);
    TORCH_CHECK(w_c.size(0) == N_in, "dgrad_fast: weight N mismatch");

    auto sizes = dy_c.sizes().vec();
    sizes.back() = K_out;
    auto dx = torch::empty(sizes, dy_c.options());

    gemm_dgrad_Tensor_a_t a_desc = { dy_c.data_ptr() };
    gemm_dgrad_Tensor_b_t b_desc = { w_c.data_ptr() };
    gemm_dgrad_Tensor_c_t c_desc = { dx.data_ptr() };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int32_t ret = cute_dsl_gemm_dgrad_wrapper(
        &g_dgrad_module, &a_desc, &b_desc, &c_desc, stream);
    TORCH_CHECK(ret == 0, "CuTeDSL dgrad kernel failed with code ", ret);

    return dx;
}

torch::Tensor cutedsl_gemm_wgrad_fast(torch::Tensor dy, torch::Tensor x) {
    ensure_init();
    TORCH_CHECK(dy.is_cuda() && x.is_cuda());
    TORCH_CHECK(dy.dtype() == torch::kBFloat16 && x.dtype() == torch::kBFloat16);

    auto dy_c = dy.contiguous();
    auto x_c = x.contiguous();

    int64_t N_dy = dy_c.size(-1);     // 8192 (M for wgrad)
    int64_t K_x  = x_c.size(-1);      // 1024 (N for wgrad)
    int64_t K_act_dy = dy_c.numel() / N_dy;  // 40960
    int64_t K_act_x  = x_c.numel()  / K_x;   // 40960

    TORCH_CHECK(K_act_dy == 40960 && K_act_x == 40960 &&
                N_dy == 8192 && K_x == 1024,
                "wgrad_fast: expected K=40960 M=8192 N=1024; got "
                "K_dy=", K_act_dy, " K_x=", K_act_x,
                " M=", N_dy, " N=", K_x);

    auto dw = torch::empty({N_dy, K_x}, dy_c.options().dtype(torch::kFloat32));

    int K = (int)K_act_dy;
    int M = (int)N_dy;
    int N = (int)K_x;

    gemm_wgrad_Tensor_a_t a_desc = {
        dy_c.data_ptr(), {M, K, 1}, {(int64_t)M, (int64_t)M * K}
    };
    gemm_wgrad_Tensor_b_t b_desc = {
        x_c.data_ptr(),  {N, K, 1}, {(int64_t)N, (int64_t)N * K}
    };
    gemm_wgrad_Tensor_c_t c_desc = {
        dw.data_ptr(),   {M, N, 1}, {(int64_t)N, (int64_t)M * N}
    };

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    int32_t ret = cute_dsl_gemm_wgrad_wrapper(
        &g_wgrad_module, &a_desc, &b_desc, &c_desc, stream);
    TORCH_CHECK(ret == 0, "CuTeDSL wgrad kernel failed with code ", ret);

    return dw;
}

/* ============================================================================
 * Round 39: single-call combined bwd binding.
 *
 * R37 folded each direction's Python dispatch (contiguous / reshape / alloc /
 * view) into one pybind11 entry per direction.  R39 takes the next step and
 * folds **both** bwd directions (dgrad + wgrad) into a single pybind11 call
 * so that:
 *   - dy is made contiguous + reshaped to 2-D **once** instead of twice
 *     (the previous fast path called gemm_dgrad_fast(dy, w) and
 *      gemm_wgrad_fast(dy, x) back-to-back, each redoing the dy 3D→2D work).
 *   - One Python→C++ round trip per `gemm_fc1_bwd` invocation instead of two.
 *   - The final `.view_as(x)` for d_input is done in C++ as well (returned
 *     with the original 3-D shape, no Python view needed).
 *
 * The contract matches the previous two-call sequence exactly:
 *   d_input  = dgrad(dy_2d, weight)  — 3-D, bf16, same shape as `x`.
 *   d_weight = wgrad(dy_2d, x_2d)    — 2-D, fp32, [8192, 1024].
 *
 * Everything else (dispatcher backend selection, parallel-bwd, opt-in
 * coop_canonical wgrad) still lives in `kernel.py`'s slow path.
 * ============================================================================ */

std::tuple<torch::Tensor, torch::Tensor>
cutedsl_gemm_bwd_fast(torch::Tensor dy, torch::Tensor x, torch::Tensor w) {
    ensure_init();
    TORCH_CHECK(dy.is_cuda() && x.is_cuda() && w.is_cuda());
    TORCH_CHECK(dy.dtype() == torch::kBFloat16 &&
                x.dtype()  == torch::kBFloat16 &&
                w.dtype()  == torch::kBFloat16);

    auto dy_c = dy.contiguous();
    auto x_c  = x.contiguous();
    auto w_c  = w.contiguous();

    int64_t N_dy = dy_c.size(-1);            // 8192
    int64_t K_x  = x_c.size(-1);             // 1024
    int64_t M    = dy_c.numel() / N_dy;      // 40960
    int64_t M_x  = x_c.numel() / K_x;        // 40960

    TORCH_CHECK(M == 40960 && M_x == 40960 && N_dy == 8192 && K_x == 1024,
                "bwd_fast: expected M=40960 N=8192 K=1024; got M=",
                M, " M_x=", M_x, " N=", N_dy, " K=", K_x);
    TORCH_CHECK(w_c.size(0) == N_dy && w_c.size(1) == K_x,
                "bwd_fast: weight expected [", N_dy, ",", K_x, "]; got [",
                w_c.size(0), ",", w_c.size(1), "]");

    // Allocate d_input with x's original shape (preserves [S, B, K]
    // residual-stream layout that the upstream `view_as(x)` would have
    // produced) and d_weight as 2-D fp32 contig.
    auto d_input  = torch::empty(x_c.sizes(), x_c.options());
    auto d_weight = torch::empty({N_dy, K_x},
                                 dy_c.options().dtype(torch::kFloat32));

    auto main = c10::cuda::getCurrentCUDAStream().stream();

    // Round 42: parallel dgrad on a non-blocking side stream + wgrad on the
    // caller's current stream is the new default.  Reads `GEMM_FC1_BWD_PARALLEL`
    // once per process and caches the choice in a static; flipping at runtime
    // is not supported (same semantics as the R41 `GEMM_FC1_BWD_ORDER` env).
    //
    //   unset / "1" → parallel (default; collapses bimodal bwd onto fast cluster)
    //   "0"         → serial   (opt-in fallback to R41 wgrad_first ordering;
    //                          R41 `GEMM_FC1_BWD_ORDER` still respected here)
    static const int parallel_choice = []() {
        const char *env = std::getenv("GEMM_FC1_BWD_PARALLEL");
        return (env && std::string(env) == "0") ? 0 : 1;
    }();

    auto launch_dgrad_on = [&](cudaStream_t s) {
        gemm_dgrad_Tensor_a_t a_desc = { dy_c.data_ptr() };
        gemm_dgrad_Tensor_b_t b_desc = { w_c.data_ptr() };
        gemm_dgrad_Tensor_c_t c_desc = { d_input.data_ptr() };
        int32_t ret = cute_dsl_gemm_dgrad_wrapper(
            &g_dgrad_module, &a_desc, &b_desc, &c_desc, s);
        TORCH_CHECK(ret == 0, "CuTeDSL dgrad kernel failed with code ", ret);
    };

    auto launch_wgrad_on = [&](cudaStream_t s) {
        const int Mw = (int)N_dy;       // 8192
        const int Nw = (int)K_x;        // 1024
        const int Kw = (int)M;          // 40960
        gemm_wgrad_Tensor_a_t a_desc = {
            dy_c.data_ptr(), {Mw, Kw, 1},
            {(int64_t)Mw, (int64_t)Mw * Kw}
        };
        gemm_wgrad_Tensor_b_t b_desc = {
            x_c.data_ptr(),  {Nw, Kw, 1},
            {(int64_t)Nw, (int64_t)Nw * Kw}
        };
        gemm_wgrad_Tensor_c_t c_desc = {
            d_weight.data_ptr(), {Mw, Nw, 1},
            {(int64_t)Nw, (int64_t)Mw * Nw}
        };
        int32_t ret = cute_dsl_gemm_wgrad_wrapper(
            &g_wgrad_module, &a_desc, &b_desc, &c_desc, s);
        TORCH_CHECK(ret == 0, "CuTeDSL wgrad kernel failed with code ", ret);
    };

    // Round 44: optional cudaGraph capture for the parallel bwd sequence.
    // Enabled via `GEMM_FC1_BWD_GRAPH=1`; default 0 keeps R42 native-stream
    // behaviour.  Read once per process (matches `parallel_choice` /
    // `order_choice` semantics — flipping mid-process unsupported).
    static const int graph_choice = []() {
        const char *env = std::getenv("GEMM_FC1_BWD_GRAPH");
        return (env && std::string(env) == "1") ? 1 : 0;
    }();

    if (parallel_choice == 1) {
        // Parallel path (R42 default).  Side stream takes dgrad (the lighter
        // ~30-40 µs persistent kernel) so it can ride wgrad's tail wave;
        // wgrad stays on the caller's main stream so subsequent ops on that
        // stream see d_weight via natural stream ordering plus the explicit
        // `cudaStreamWaitEvent(main, side)` that drains d_input back.
        ensure_parallel_init();

        if (graph_choice == 1) {
            // Round 44 opt-in — capture the 6-op sequence into a graph and
            // dispatch it via the runtime's graph executor.  We recapture
            // every call (pointers change every step) but reuse the cached
            // `g_bwd_graph_exec` via in-place `cudaGraphExecUpdate` so we
            // pay only the cheap topology-comparison cost (not a full
            // re-instantiate).
            //
            // We capture into a private `g_capture_stream` (created in
            // `ensure_parallel_init` and synchronised to `main` via events)
            // rather than `main` directly, because PyTorch's per-device
            // current stream may not always satisfy `cudaStreamBeginCapture`
            // preconditions (e.g. legacy default stream variants), and a
            // dedicated capture stream side-steps that entirely.  The graph
            // is launched onto `main` so semantic ordering is preserved.
            cudaError_t err = cudaEventRecord(g_evt_main_to_side, main);
            TORCH_CHECK(err == cudaSuccess,
                        "graph pre-evt main->capture record failed: ",
                        cudaGetErrorString(err));
            err = cudaStreamWaitEvent(
                g_capture_stream, g_evt_main_to_side, 0);
            TORCH_CHECK(err == cudaSuccess,
                        "graph capture wait main failed: ",
                        cudaGetErrorString(err));

            err = cudaStreamBeginCapture(
                g_capture_stream, cudaStreamCaptureModeRelaxed);
            TORCH_CHECK(err == cudaSuccess,
                        "cudaStreamBeginCapture failed: ",
                        cudaGetErrorString(err));

            err = cudaEventRecord(g_evt_main_to_side, g_capture_stream);
            TORCH_CHECK(err == cudaSuccess,
                        "graph evt capture->side record failed: ",
                        cudaGetErrorString(err));
            err = cudaStreamWaitEvent(g_side_stream, g_evt_main_to_side, 0);
            TORCH_CHECK(err == cudaSuccess,
                        "graph side wait capture failed: ",
                        cudaGetErrorString(err));

            launch_dgrad_on(g_side_stream);
            launch_wgrad_on(g_capture_stream);

            err = cudaEventRecord(g_evt_side_to_main, g_side_stream);
            TORCH_CHECK(err == cudaSuccess,
                        "graph evt side->capture record failed: ",
                        cudaGetErrorString(err));
            err = cudaStreamWaitEvent(
                g_capture_stream, g_evt_side_to_main, 0);
            TORCH_CHECK(err == cudaSuccess,
                        "graph capture wait side failed: ",
                        cudaGetErrorString(err));

            cudaGraph_t graph = nullptr;
            err = cudaStreamEndCapture(g_capture_stream, &graph);
            TORCH_CHECK(err == cudaSuccess,
                        "cudaStreamEndCapture failed: ",
                        cudaGetErrorString(err));

            if (g_bwd_graph_exec == nullptr) {
                err = cudaGraphInstantiate(
                    &g_bwd_graph_exec, graph, nullptr, nullptr, 0);
                TORCH_CHECK(err == cudaSuccess,
                            "cudaGraphInstantiate (initial) failed: ",
                            cudaGetErrorString(err));
            } else {
                cudaGraphExecUpdateResultInfo update_info{};
                err = cudaGraphExecUpdate(
                    g_bwd_graph_exec, graph, &update_info);
                if (err != cudaSuccess ||
                    update_info.result != cudaGraphExecUpdateSuccess) {
                    // Fall back to fresh instantiate; common case is
                    // `cudaGraphExecUpdateErrorTopologyChanged` which
                    // shouldn't happen here (topology is fixed) but we
                    // handle it for robustness.
                    cudaGraphExecDestroy(g_bwd_graph_exec);
                    err = cudaGraphInstantiate(
                        &g_bwd_graph_exec, graph, nullptr, nullptr, 0);
                    TORCH_CHECK(err == cudaSuccess,
                                "cudaGraphInstantiate (re-after-update) "
                                "failed: ", cudaGetErrorString(err));
                }
            }
            cudaGraphDestroy(graph);

            err = cudaGraphLaunch(g_bwd_graph_exec, main);
            TORCH_CHECK(err == cudaSuccess,
                        "cudaGraphLaunch failed: ", cudaGetErrorString(err));
        } else {
            // R42 default native-stream path.
            // Make the side stream wait for any in-flight work on `main`
            // that produced the input tensors (e.g., upstream contiguous()
            // copies); without this, the side dgrad could race ahead of
            // input visibility.
            cudaError_t err = cudaEventRecord(g_evt_main_to_side, main);
            TORCH_CHECK(err == cudaSuccess,
                        "evt main->side record failed: ",
                        cudaGetErrorString(err));
            err = cudaStreamWaitEvent(g_side_stream, g_evt_main_to_side, 0);
            TORCH_CHECK(err == cudaSuccess,
                        "side wait main failed: ", cudaGetErrorString(err));

            launch_dgrad_on(g_side_stream);
            launch_wgrad_on(main);

            // Drain side back to main so the caller (autograd) sees
            // d_input when its next op on `main` runs.  d_weight already
            // sits on main.
            err = cudaEventRecord(g_evt_side_to_main, g_side_stream);
            TORCH_CHECK(err == cudaSuccess,
                        "evt side->main record failed: ",
                        cudaGetErrorString(err));
            err = cudaStreamWaitEvent(main, g_evt_side_to_main, 0);
            TORCH_CHECK(err == cudaSuccess,
                        "main wait side failed: ", cudaGetErrorString(err));
        }
    } else {
        // Serial path (opt-in via GEMM_FC1_BWD_PARALLEL=0).  R41 dispatch-
        // order rule kept intact for ablation: `GEMM_FC1_BWD_ORDER=dgrad_first`
        // restores the pre-R41 ordering; default is `wgrad_first`.
        static const int order_choice = []() {
            const char *env = std::getenv("GEMM_FC1_BWD_ORDER");
            return (env && std::string(env) == "dgrad_first") ? 0 : 1;
        }();
        if (order_choice == 1) {
            launch_wgrad_on(main);
            launch_dgrad_on(main);
        } else {
            launch_dgrad_on(main);
            launch_wgrad_on(main);
        }
    }

    return std::make_tuple(d_input, d_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_fwd",   &cutedsl_gemm_fwd,   "CuTeDSL GEMM fwd (2-D, pre-allocated)");
    m.def("gemm_dgrad", &cutedsl_gemm_dgrad, "CuTeDSL GEMM dgrad (2-D, pre-allocated)");
    m.def("gemm_wgrad", &cutedsl_gemm_wgrad, "CuTeDSL GEMM wgrad FP32 (2-D, pre-allocated)");

    // Round 37: fast-path bindings — accept 3-D tensors, fold
    // contiguous+reshape+alloc into one pybind11 call.
    m.def("gemm_fwd_fast",   &cutedsl_gemm_fwd_fast,
          "CuTeDSL GEMM fwd (3-D in/out, internal alloc)");
    m.def("gemm_dgrad_fast", &cutedsl_gemm_dgrad_fast,
          "CuTeDSL GEMM dgrad (3-D in/out, internal alloc)");
    m.def("gemm_wgrad_fast", &cutedsl_gemm_wgrad_fast,
          "CuTeDSL GEMM wgrad FP32 (3-D in, 2-D out, internal alloc)");

    // Round 39: combined dgrad+wgrad single pybind11 call.  Returns
    // (d_input, d_weight) tuple matching the previous two-call sequence.
    m.def("gemm_bwd_fast",   &cutedsl_gemm_bwd_fast,
          "CuTeDSL GEMM dgrad+wgrad combined (3-D dy/x in, returns "
          "(d_input 3-D bf16, d_weight 2-D fp32))");
}
