// Gradient checks for the ops LeWM added to ggml on the `lewm` branch:
// norm_back (LayerNorm), gelu_erf_back, and the reprelu straight-through op.
// Each analytic gradient from ggml_build_backward_expand is compared against central finite
// differences of the forward pass, on the CPU backend.

#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace {

struct Ctx {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ~Ctx() { ggml_backend_free(backend); }
};

using OpFn = std::function<ggml_tensor*(ggml_context*, ggml_tensor*)>;

// loss(x) = sum(op(x) * w) for a fixed random weighting w, so the gradient is dense and well scaled.
struct Result {
    std::vector<float> analytic;
    std::vector<float> numeric;
    std::vector<float> forward;
};

Result gradient_check(const OpFn& op, const std::vector<float>& x0, int64_t ne0, int64_t ne1, bool numeric_from_relu = false) {
    Ctx c;
    const int64_t n = ne0 * ne1;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> w(static_cast<size_t>(n));
    for (float& v : w) v = nd(rng);

    auto build = [&](const std::vector<float>& xv, bool backward) -> std::pair<float, std::vector<float>> {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(256, true) * 2 + 1 << 20;
        ip.no_alloc = true;
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        ggml_set_name(x, "x");
        ggml_set_param(x);
        ggml_tensor* wt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        ggml_set_name(wt, "w");
        ggml_set_input(wt);
        ggml_tensor* y = op(ctx, x);
        ggml_tensor* loss = ggml_sum(ctx, ggml_mul(ctx, y, wt));
        ggml_set_loss(loss);
        ggml_set_output(y);

        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 256, backward);
        ggml_build_forward_expand(gf, loss);
        if (backward) ggml_build_backward_expand(ctx, gf, nullptr);

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c.backend));
        REQUIRE(ggml_gallocr_alloc_graph(galloc, gf));
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(wt, w.data(), 0, ggml_nbytes(wt));
        if (backward) ggml_graph_reset(gf);  // zero gradient accumulators, seed loss grad with 1
        REQUIRE(ggml_backend_graph_compute(c.backend, gf) == GGML_STATUS_SUCCESS);

        float l = 0;
        ggml_backend_tensor_get(loss, &l, 0, sizeof(float));
        std::vector<float> g;
        if (backward) {
            ggml_tensor* gx = ggml_graph_get_grad(gf, x);
            REQUIRE(gx != nullptr);
            g.resize(static_cast<size_t>(n));
            ggml_backend_tensor_get(gx, g.data(), 0, ggml_nbytes(gx));
        } else {
            g.resize(static_cast<size_t>(n));
            ggml_backend_tensor_get(y, g.data(), 0, ggml_nbytes(y));
        }
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return {l, g};
    };

    Result r;
    auto [l0, analytic] = build(x0, true);
    r.analytic = analytic;
    r.forward = build(x0, false).second;

    // central differences in double precision on the loss
    r.numeric.resize(static_cast<size_t>(n));
    const float h = 1e-2f;
    for (int64_t i = 0; i < n; ++i) {
        std::vector<float> xp = x0, xm = x0;
        xp[static_cast<size_t>(i)] += h;
        xm[static_cast<size_t>(i)] -= h;
        const double lp = build(xp, false).first;
        const double lm = build(xm, false).first;
        r.numeric[static_cast<size_t>(i)] = static_cast<float>((lp - lm) / (2.0 * h));
    }
    (void)numeric_from_relu;
    return r;
}

std::vector<float> random_input(int64_t n, unsigned seed, float scale = 1.5f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> x(static_cast<size_t>(n));
    for (float& v : x) v = nd(rng);
    return x;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    return m;
}

}  // namespace

TEST_CASE("ggml: norm_back matches finite differences of ggml_norm") {
    const int64_t ne0 = 16, ne1 = 3;
    auto x = random_input(ne0 * ne1, 1);
    Result r = gradient_check([](ggml_context* ctx, ggml_tensor* t) { return ggml_norm(ctx, t, 1e-6f); }, x, ne0, ne1);
    INFO("max abs diff analytic vs numeric: ", max_abs_diff(r.analytic, r.numeric));
    CHECK(max_abs_diff(r.analytic, r.numeric) < 2e-3);
}

TEST_CASE("ggml: gelu_erf_back matches finite differences of ggml_gelu_erf") {
    const int64_t ne0 = 8, ne1 = 4;
    auto x = random_input(ne0 * ne1, 2);
    Result r = gradient_check([](ggml_context* ctx, ggml_tensor* t) { return ggml_gelu_erf(ctx, t); }, x, ne0, ne1);
    INFO("max abs diff analytic vs numeric: ", max_abs_diff(r.analytic, r.numeric));
    CHECK(max_abs_diff(r.analytic, r.numeric) < 2e-3);
}

TEST_CASE("ggml: reprelu forward is relu, backward is gelu_erf's gradient") {
    const int64_t ne0 = 8, ne1 = 4;
    auto x = random_input(ne0 * ne1, 3);
    Result rep = gradient_check([](ggml_context* ctx, ggml_tensor* t) { return ggml_reprelu(ctx, t); }, x, ne0, ne1);
    Result gelu = gradient_check([](ggml_context* ctx, ggml_tensor* t) { return ggml_gelu_erf(ctx, t); }, x, ne0, ne1);

    // forward value equals relu(x) exactly
    for (size_t i = 0; i < x.size(); ++i) CHECK(rep.forward[i] == std::max(x[i], 0.0f));
    // analytic gradient equals gelu_erf's analytic gradient (straight-through)
    CHECK(max_abs_diff(rep.analytic, gelu.analytic) < 1e-6);
    // and therefore differs from relu's own finite-difference gradient where x < 0
    bool differs_somewhere = false;
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] < -0.5f && std::fabs(rep.analytic[i]) > 1e-3f) differs_somewhere = true;
    }
    CHECK(differs_somewhere);
}

// repeat_back on the GPU backend must match a double-precision host reference for the shapes
// training produces: bias gradients (reduce every row), positional-embedding gradients (reduce the
// batch dim only), repeats along every dim at once, and a strided (transposed-view) source. The
// CPU backend is checked too where it applies (it asserts on strided sources). Skipped without a GPU.
TEST_CASE("ggml: repeat_back on the GPU backend matches a host reference") {
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu == nullptr) {
        MESSAGE("no GPU backend; skipping");
        return;
    }
    struct Case { std::array<int64_t, 4> src; std::array<int64_t, 4> dst; bool transpose; };
    const Case cases[] = {
        {{384, 257, 64, 1}, {384, 1, 1, 1}, false},
        {{384, 257, 64, 1}, {384, 257, 1, 1}, false},
        {{16, 6, 4, 2}, {8, 3, 4, 1}, false},
        {{300, 70, 1, 1}, {300, 1, 1, 1}, true},   // src is the transposed view of a {70, 300} tensor
    };
    for (const Case& c : cases) {
        const int64_t n_src = c.src[0] * c.src[1] * c.src[2] * c.src[3];
        const std::vector<float> x = random_input(n_src, 11);
        // host reference in double; x is stored in the base tensor's layout
        std::vector<double> ref(static_cast<size_t>(c.dst[0] * c.dst[1] * c.dst[2] * c.dst[3]), 0.0);
        for (int64_t i3 = 0; i3 < c.src[3]; ++i3)
            for (int64_t i2 = 0; i2 < c.src[2]; ++i2)
                for (int64_t i1 = 0; i1 < c.src[1]; ++i1)
                    for (int64_t i0 = 0; i0 < c.src[0]; ++i0) {
                        const int64_t flat = c.transpose ? (i0 * c.src[1] + i1)
                                                         : (((i3 * c.src[2] + i2) * c.src[1] + i1) * c.src[0] + i0);
                        const int64_t o = ((i3 % c.dst[3]) * c.dst[2] + i2 % c.dst[2]) * c.dst[1] * c.dst[0] +
                                          (i1 % c.dst[1]) * c.dst[0] + i0 % c.dst[0];
                        ref[static_cast<size_t>(o)] += x[static_cast<size_t>(flat)];
                    }
        auto run = [&](ggml_backend_t backend) {
            ggml_init_params ip{};
            ip.mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
            ip.no_alloc = true;
            ggml_context* ctx = ggml_init(ip);
            ggml_tensor* base = c.transpose ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.src[1], c.src[0])
                                            : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.src[0], c.src[1], c.src[2], c.src[3]);
            ggml_set_input(base);
            ggml_tensor* src = c.transpose ? ggml_transpose(ctx, base) : base;
            ggml_tensor* dst = ggml_repeat_back(ctx, src, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.dst[0], c.dst[1], c.dst[2], c.dst[3]));
            ggml_set_output(dst);
            ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, dst);
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            REQUIRE(ggml_gallocr_alloc_graph(ga, gf));
            ggml_backend_tensor_set(base, x.data(), 0, x.size() * sizeof(float));
            REQUIRE(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
            std::vector<float> out(static_cast<size_t>(ggml_nelements(dst)));
            ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
            ggml_gallocr_free(ga);
            ggml_free(ctx);
            return out;
        };
        auto worst_abs = [&](const std::vector<float>& got) {
            REQUIRE(got.size() == ref.size());
            double w = 0;
            for (size_t i = 0; i < ref.size(); ++i) w = std::max(w, std::fabs(got[i] - ref[i]));
            return w;
        };
        // fp32 accumulation of R terms of scale ~1.5: allow a few ulps per term
        const double R = static_cast<double>(n_src) / static_cast<double>(ref.size());
        const double tol = 4e-7 * R * 1.5 + 1e-6;
        CAPTURE(c.src[0]); CAPTURE(c.src[1]); CAPTURE(c.src[2]); CAPTURE(c.src[3]);
        CAPTURE(c.dst[0]); CAPTURE(c.dst[1]); CAPTURE(c.dst[2]); CAPTURE(c.dst[3]);

        ggml_backend_t dev = ggml_backend_dev_init(gpu, nullptr);
        REQUIRE(dev != nullptr);
        const std::vector<float> got = run(dev);
        ggml_backend_free(dev);
        const double gpu_err = worst_abs(got);
        CAPTURE(gpu_err); CAPTURE(tol);
        CHECK(gpu_err < tol);

        if (!c.transpose) {
            ggml_backend_t cpu = ggml_backend_cpu_init();
            const double cpu_err = worst_abs(run(cpu));
            ggml_backend_free(cpu);
            CAPTURE(cpu_err);
            CHECK(cpu_err < tol);
        }
        // deterministic: a second run gives bit-identical output
        ggml_backend_t dev2 = ggml_backend_dev_init(gpu, nullptr);
        const std::vector<float> again = run(dev2);
        ggml_backend_free(dev2);
        CHECK(again == got);
    }
}
