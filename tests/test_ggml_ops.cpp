// Gradient checks for the ops LeWM added to ggml on the `lewm` branch:
// norm_back (LayerNorm), gelu_erf_back, and the reprelu straight-through op.
// Each analytic gradient from ggml_build_backward_expand is compared against central finite
// differences of the forward pass, on the CPU backend.

#include <doctest/doctest.h>

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
