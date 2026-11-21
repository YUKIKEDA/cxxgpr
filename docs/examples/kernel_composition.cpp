// SPDX-License-Identifier: Apache-2.0
// How-to: compose kernels with + and *, inspect the parameter vector, evaluate Gram matrices and gradients,
// and hand the kernel to Gpr. Every printed value is checked against a formula written out by hand.
#include <array>
#include <cmath>
#include <cstdlib>
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/transform/target.hpp>
#include <iostream>
#include <string_view>
#include <vector>

namespace
{

    template <typename T> T unwrap(std::expected<T, cxxgpr::GprError> result)
    {
        if (!result)
        {
            std::cerr << "error: " << result.error().message() << '\n';
            std::exit(1);
        }
        return std::move(*result);
    }

    void check(const std::expected<void, cxxgpr::GprError>& result)
    {
        if (!result)
        {
            std::cerr << "error: " << result.error().message() << '\n';
            std::exit(1);
        }
    }

    bool all_ok = true;

    void expect(bool condition, std::string_view what)
    {
        all_ok = all_ok && condition;
        if (!condition)
        {
            std::cout << "CHECK FAILED: " << what << '\n';
        }
    }

    std::string_view to_string(cxxgpr::CoordMode mode)
    {
        switch (mode)
        {
            case cxxgpr::CoordMode::Dist:
                return "Dist";
            case cxxgpr::CoordMode::Points:
                return "Points";
            case cxxgpr::CoordMode::Either:
                return "Either";
            case cxxgpr::CoordMode::Mixed:
                return "Mixed";
        }
        return "?";
    }

}  // namespace

int main()
{
    using namespace cxxgpr;

    // [compose]
    // k(x, x') = 2.0 * RBF(l = 1.5) + 0.1 * delta(x, x')
    const auto signal = unwrap(ConstantKernel::create(2.0));
    const auto rbf = unwrap(RbfKernel::create(1.5));
    const auto white = unwrap(WhiteKernel::create(0.1));
    KernelSpec spec = KernelSpec(signal) * KernelSpec(rbf) + KernelSpec(white);
    // [end]

    // The same tree through the named factories: sum(product(signal, rbf), white).
    const KernelSpec same =
        KernelSpec::sum(KernelSpec::product(KernelSpec(signal), KernelSpec(rbf)), KernelSpec(white));
    std::array<double, 3> theta{};
    std::array<double, 3> theta_same{};
    check(spec.get_params(theta));
    check(same.get_params(theta_same));
    expect(theta == theta_same, "operators and factories build the same tree");

    // [parameters]
    std::cout << "num_params = " << spec.num_params() << '\n';
    std::cout << "log-parameters = [" << theta[0] << ", " << theta[1] << ", " << theta[2] << "]\n";
    for (const ParameterBinding& b : spec.parameter_bindings())
    {
        std::cout << "  parameter " << b.index << " -> leaf " << b.leaf_id << ", local slot " << b.local_index << '\n';
    }
    std::array<Interval, 3> bounds{
        Interval::default_positive(), Interval::default_positive(), Interval::default_positive()
    };
    check(spec.bounds_into(bounds));
    std::cout << "coord_mode = " << to_string(spec.coord_mode()) << '\n';
    // [end]
    expect(spec.num_params() == 3, "three parameters");
    expect(
        std::abs(theta[0] - std::log(2.0)) < 1e-15 && std::abs(theta[1] - std::log(1.5)) < 1e-15
            && std::abs(theta[2] - std::log(0.1)) < 1e-15,
        "parameters are stored as logarithms in depth-first order"
    );
    expect(spec.coord_mode() == CoordMode::Dist, "constant * rbf + white needs only distances");

    // set_params is atomic: log(0.1) -> 100 pushes the white variance to e^100, far outside (1e-5, 1e5).
    const std::array<double, 3> bad{std::log(3.0), std::log(1.5), 100.0};
    const auto rejected = spec.set_params(bad);
    std::array<double, 3> after{};
    check(spec.get_params(after));
    std::cout << "set_params(bad) -> " << (rejected ? "ok" : std::string(rejected.error().message())) << '\n';
    expect(
        !rejected && rejected.error().code() == GprErrorCode::InvalidInterval && after == theta,
        "failed set_params leaves every leaf unchanged"
    );

    // Compile for double precision and look at the execution-layer properties.
    // [compile]
    CompiledKernel<double> kernel = spec.compile<double>();
    std::cout << "leaves = " << kernel.num_leaves() << ", nested_depth = " << kernel.nested_depth()
              << ", reads_distances = " << std::boolalpha << kernel.reads_distances()
              << ", needs_ard_sq_diff = " << kernel.needs_ard_sq_diff() << '\n';
    // [end]
    expect(kernel.num_leaves() == 3 && kernel.reads_distances() && !kernel.needs_ard_sq_diff(), "compiled properties");

    // Three one-dimensional training points and their pairwise squared distances.
    const std::array<double, 3> x_data{0.0, 1.0, 2.0};
    const std::array<double, 9> d2_data{0, 1, 4, 1, 0, 1, 4, 1, 0};
    const MatrixRef<double> x(x_data.data(), 3, 1);
    const MatrixRef<double> d2(d2_data.data(), 3, 3);

    // [gram]
    std::array<double, 9> gram{};
    std::array<double, 9> scratch{};
    check(
        kernel.apply(d2, MatrixMut<double>(gram.data(), 3, 3), Triangle::Full, MatrixMut<double>(scratch.data(), 3, 3))
    );
    // The same matrix from coordinates instead of precomputed distances.
    std::array<double, 9> gram_pts{};
    check(kernel.apply_points(
        x, MatrixMut<double>(gram_pts.data(), 3, 3), Triangle::Full, MatrixMut<double>(scratch.data(), 3, 3)
    ));
    std::cout << "K(0,0) = " << gram[0] << "  K(1,0) = " << gram[1] << "  K(2,0) = " << gram[2] << '\n';
    // [end]
    const double k10 = 2.0 * std::exp(-1.0 / (2.0 * 1.5 * 1.5));
    const double k20 = 2.0 * std::exp(-4.0 / (2.0 * 1.5 * 1.5));
    expect(std::abs(gram[0] - 2.1) < 1e-14, "diagonal = 2.0 + 0.1");
    expect(std::abs(gram[1] - k10) < 1e-14 && std::abs(gram[2] - k20) < 1e-14, "off-diagonal = 2 exp(-d2 / 2l^2)");
    expect(gram == gram_pts, "distance and coordinate paths agree");

    // Cross-covariance with two query points: white noise contributes nothing between distinct point sets.
    const std::array<double, 2> q_data{0.5, 1.5};
    std::array<double, 6> cross{};
    std::array<double, 6> cross_scratch{};
    check(kernel.apply_cross_points(
        MatrixRef<double>(q_data.data(), 2, 1), x, MatrixMut<double>(cross.data(), 2, 3),
        MatrixMut<double>(cross_scratch.data(), 2, 3)
    ));
    // Column-major 2 x 3: cross(i, j) = cross[i + 2 j]; entry (0, 0) is k(0.5, 0).
    expect(std::abs(cross[0] - 2.0 * std::exp(-0.25 / 4.5)) < 1e-14, "cross-covariance k(0.5, 0)");
    expect(std::abs(cross[1] - 2.0 * std::exp(-2.25 / 4.5)) < 1e-14, "cross-covariance k(1.5, 0)");

    // Prior variance at the training points: signal + noise.
    std::array<double, 3> diag{};
    check(kernel.fill_diag(x, diag));
    expect(diag[0] == diag[1] && std::abs(diag[0] - 2.1) < 1e-14, "fill_diag");

    // Derivative with respect to a log-parameter. Parameter 1 is log(lengthscale):
    //   d/d(log l) [2 exp(-d2 / (2 l^2))] = 2 exp(-d2 / (2 l^2)) * d2 / l^2.
    // [gradient]
    std::array<double, 9> d_k{};
    std::array<double, 9> work_b{};
    check(kernel.grad_points(
        x, MatrixMut<double>(d_k.data(), 3, 3), 1, Triangle::Full, MatrixMut<double>(scratch.data(), 3, 3),
        MatrixMut<double>(work_b.data(), 3, 3)
    ));
    // [end]
    expect(
        std::abs(d_k[1] - k10 * 1.0 / 2.25) < 1e-13 && std::abs(d_k[2] - k20 * 4.0 / 2.25) < 1e-13,
        "gradient with respect to log(lengthscale)"
    );
    std::cout << "dK/dlog(l): (1,0) = " << d_k[1] << " (expected " << k10 / 2.25 << ")\n";

    // A tree that needs coordinates: an ARD leaf makes the kernel Points-only, so apply() with distances fails.
    const auto ard = unwrap(RbfArdKernel::create(std::array<double, 3>{1.0, 2.0, 3.0}));
    const KernelSpec ard_spec(ard);
    const auto ard_kernel = ard_spec.compile<double>();
    const auto no_dist = ard_kernel.apply(
        d2, MatrixMut<double>(gram.data(), 3, 3), Triangle::Full, MatrixMut<double>(scratch.data(), 3, 3)
    );
    std::cout << "ARD kernel coord_mode = " << to_string(ard_kernel.coord_mode()) << ", apply(distances) -> "
              << (no_dist ? "ok" : std::string(no_dist.error().message())) << '\n';
    expect(!no_dist && no_dist.error().code() == GprErrorCode::UnsupportedKernelOperation, "ARD needs coordinates");
    // x has one column, the ARD leaf three lengthscales.
    const auto wrong_dims = ard_kernel.apply_points(
        x, MatrixMut<double>(gram.data(), 3, 3), Triangle::Full, MatrixMut<double>(scratch.data(), 3, 3)
    );
    expect(!wrong_dims && wrong_dims.error().code() == GprErrorCode::DimensionMismatch, "ARD dimension check");

    // Single precision: the same tree compiled for float agrees to about 1e-7.
    const auto kernel_f = spec.compile<float>();
    const std::array<float, 3> xf_data{0.0F, 1.0F, 2.0F};
    std::array<float, 9> gram_f{};
    std::array<float, 9> scratch_f{};
    check(kernel_f.apply_points(
        MatrixRef<float>(xf_data.data(), 3, 1), MatrixMut<float>(gram_f.data(), 3, 3), Triangle::Full,
        MatrixMut<float>(scratch_f.data(), 3, 3)
    ));
    expect(std::abs(static_cast<double>(gram_f[1]) - k10) < 1e-6, "float kernel");

    // [leaf-level]
    // Evaluate each leaf alone, then rebuild the composite without touching the leaves again.
    auto nested = kernel.make_nested_buffers(3, 3);
    std::array<std::array<double, 9>, 3> leaf_grams{};
    for (std::size_t leaf = 0; leaf < kernel.num_leaves(); ++leaf)
    {
        check(
            kernel.apply_leaf_points(leaf, x, MatrixMut<double>(leaf_grams[leaf].data(), 3, 3), Triangle::Full, nested)
        );
    }
    const std::array<MatrixRef<double>, 3> leaf_views{
        MatrixRef<double>(leaf_grams[0].data(), 3, 3), MatrixRef<double>(leaf_grams[1].data(), 3, 3),
        MatrixRef<double>(leaf_grams[2].data(), 3, 3)
    };
    std::array<double, 9> recombined{};
    check(kernel.combine_from_leaf_grams(
        leaf_views, MatrixMut<double>(recombined.data(), 3, 3), Triangle::Full, MatrixMut<double>(scratch.data(), 3, 3),
        nested
    ));
    const std::size_t owner = unwrap(kernel.leaf_index_for_param(2));
    std::cout << "recombined == gram: " << std::boolalpha << (recombined == gram) << ", parameter 2 belongs to leaf "
              << owner << '\n';
    // [end]
    expect(recombined == gram && owner == 2, "combine_from_leaf_grams reproduces the Gram matrix");

    // Second derivative: d2K / (d log c  d log l) equals dK / d log l because K is linear in c.
    std::array<double, 9> hess_buf{};
    std::array<double, 9> scratch_c{};
    check(kernel.hess_points(
        x, MatrixMut<double>(hess_buf.data(), 3, 3), 0, 1, Triangle::Full, MatrixMut<double>(scratch.data(), 3, 3),
        MatrixMut<double>(work_b.data(), 3, 3), MatrixMut<double>(scratch_c.data(), 3, 3)
    ));
    expect(std::abs(hess_buf[1] - d_k[1]) < 1e-13, "mixed second derivative");

    // One tree walk for all parameters: g_p = sum over the lower triangle of W(i, j) dK_ij / dtheta_p.
    // [weighted]
    std::array<double, 9> weight{};  // W: ones on and below the diagonal
    for (std::size_t col = 0; col < 3; ++col)
    {
        for (std::size_t row = col; row < 3; ++row)
        {
            weight[row + 3 * col] = 1.0;
        }
    }
    const std::size_t kept = 0;  // keep no product factors; see the API summary for kept_products()
    NestedBuffers<double> weighted_bufs(kernel.weighted_buffers(kept), 3, 3);
    std::array<double, 3> grads{};
    std::array<double, 3> col_partials{};
    check(kernel.weighted_grads(
        x, nullptr, MatrixRef<double>(weight.data(), 3, 3), grads, MatrixMut<double>(scratch.data(), 3, 3), nested,
        weighted_bufs, kept, col_partials
    ));
    // [end]
    std::cout << "weighted gradients = [" << grads[0] << ", " << grads[1] << ", " << grads[2] << "]\n";
    // Reference: contract every parameter's derivative matrix by hand. The lower triangle is mirrored, so each
    // off-diagonal entry counts twice and each diagonal entry once.
    for (std::size_t p = 0; p < 3; ++p)
    {
        check(kernel.grad_points(
            x, MatrixMut<double>(d_k.data(), 3, 3), p, Triangle::Full, MatrixMut<double>(scratch.data(), 3, 3),
            MatrixMut<double>(work_b.data(), 3, 3)
        ));
        const double reference = d_k[0] + d_k[4] + d_k[8] + 2.0 * (d_k[1] + d_k[2] + d_k[5]);
        expect(std::abs(grads[p] - reference) < 1e-12, "weighted_grads equals the hand contraction");
    }

    // [fit]
    // Hand the composite to a trainer: Gpr adds its own observation noise, so the white leaf is left out here.
    const KernelSpec model_kernel =
        KernelSpec(unwrap(ConstantKernel::create(1.0))) * KernelSpec(unwrap(RbfKernel::create(1.0)));
    std::vector<double> x_train;
    std::vector<double> y_train;
    for (int i = 0; i < 12; ++i)
    {
        x_train.push_back(0.5 * i);
        y_train.push_back(std::sin(0.5 * i));
    }
    auto trainer = unwrap(
        Gpr<optimizer::Lbfgs>(model_kernel)
            .with_target_transform(transform::StandardizeTarget{})
            .with_noise_variance(1e-4)
    );
    auto fitted = unwrap(trainer.fit(x_train, 12, 1, y_train));
    const std::array<double, 2> x_query{1.25, 3.75};
    const auto prediction = unwrap(fitted.predict(x_query, 2, 1));
    std::cout << "predict(1.25) = " << prediction.mean[0] << " (sin = " << std::sin(1.25) << ")\n";
    std::cout << "predict(3.75) = " << prediction.mean[1] << " (sin = " << std::sin(3.75) << ")\n";
    // [end]
    expect(
        std::abs(prediction.mean[0] - std::sin(1.25)) < 0.02 && std::abs(prediction.mean[1] - std::sin(3.75)) < 0.02,
        "the fitted model reproduces sin(x) between training points"
    );

    return all_ok ? 0 : 1;
}
