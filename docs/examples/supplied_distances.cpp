// SPDX-License-Identifier: Apache-2.0
// How-to: evaluate kernels from squared distances that you compute yourself (here: arc length on a circle),
// mix them with coordinate kernels, and reuse a DistanceCache for coordinate kernels.
#include <cxxgpr/kernel/kernel.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <optional>
#include <string>
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

    // Borrow the value of an expected without moving it (a CheckedDistanceStore is built by the library and kept in place).
    template <typename T> const T& borrow(const std::expected<T, cxxgpr::GprError>& result)
    {
        if (!result)
        {
            std::cerr << "error: " << result.error().message() << '\n';
            std::exit(1);
        }
        return *result;
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

    // Geodesic distance between two angles on the unit circle.
    double arc(double a, double b)
    {
        const double delta = std::abs(a - b);
        return std::min(delta, 2.0 * std::numbers::pi - delta);
    }

    // [distance-table]
    // Column-major table of squared arc lengths between two sets of angles (rows: a, columns: b).
    std::vector<double> squared_arc_table(const std::vector<double>& a, const std::vector<double>& b)
    {
        std::vector<double> d2(a.size() * b.size());
        for (std::size_t j = 0; j < b.size(); ++j)
        {
            for (std::size_t i = 0; i < a.size(); ++i)
            {
                d2[i + j * a.size()] = arc(a[i], b[j]) * arc(a[i], b[j]);
            }
        }
        return d2;
    }
    // [end]

}  // namespace

int main()
{
    using namespace cxxgpr;

    const std::vector<double> angles{0.0, 0.5, 1.5, 3.0, 5.5};
    const std::size_t n = angles.size();
    const std::vector<double> d2 = squared_arc_table(angles, angles);

    // [kernel]
    // k(a, b) = 2 * exp(-arc(a, b) / 1.0) + 0.1 * delta: an exponential (Matern 1/2) kernel on the circle.
    const ScalarDistance arc_slot;  // a fresh slot; its identity is what binds a table to the kernel
    const DistanceKernel<DistanceOnly> kernel =
        DistanceKernel<DistanceOnly>(unwrap(ConstantKernel::create(2.0))) *
            arc_slot.matern(unwrap(MaternKernel::create(1.0, MaternNu::OneHalf))) +
        DistanceKernel<DistanceOnly>(unwrap(WhiteKernel::create(0.1)));
    // [end]
    std::cout << "num_params = " << kernel.num_params() << ", scalar slots = " << kernel.num_scalar_slots()
              << ", ARD slots = " << kernel.num_ard_slots() << '\n';
    expect(kernel.num_params() == 3 && kernel.num_scalar_slots() == 1 && kernel.num_ard_slots() == 0, "slot counts");
    const std::vector<SlotDescriptor> slots = kernel.slots();
    expect(slots.size() == 1 && slots[0].id == arc_slot.slot() && !slots[0].ard_dims.has_value() && slots[0].at == 0,
           "slot descriptor");

    // [prepare]
    const CompiledDistanceKernel<double> compiled = kernel.compile<double>();
    DistanceSource source;
    source.bind_scalar(arc_slot.slot(), MatrixRef<double>(d2.data(), n, n));
    const auto prepared = compiled.prepare_square(source);
    const CheckedDistanceStore<double>& store = borrow(prepared);
    // [end]
    std::cout << "store: " << store.rows() << " x " << store.cols() << ", square = " << std::boolalpha
              << store.is_square() << ", requires_points = " << compiled.requires_points() << '\n';
    expect(store.rows() == n && store.cols() == n && store.is_square() && !compiled.requires_points(), "store shape");

    // [gram]
    std::vector<double> gram(n * n);
    std::vector<double> scratch(n * n);
    NestedBuffers<double> nested = compiled.make_nested_buffers(n, n);
    check(compiled.apply(store, std::nullopt, MatrixMut<double>(gram.data(), n, n), Triangle::Full,
                         MatrixMut<double>(scratch.data(), n, n), nested));
    // [end]
    std::cout << "K(0,0) = " << gram[0] << "  K(1,0) = " << gram[1] << "  K(3,0) = " << gram[3] << "  K(4,0) = " << gram[4]
              << '\n';
    // Hand values: angles 0.0/0.5 are 0.5 apart, 0.0/3.0 are 3.0 apart, 0.0/5.5 are 2 pi - 5.5 apart.
    expect(std::abs(gram[0] - 2.1) < 1e-14, "diagonal");
    expect(std::abs(gram[1] - 2.0 * std::exp(-0.5)) < 1e-14, "arc 0.5");
    expect(std::abs(gram[3] - 2.0 * std::exp(-3.0)) < 1e-14, "arc 3.0 (the short way round is 3.0, not 3.28)");
    expect(std::abs(gram[4] - 2.0 * std::exp(-(2.0 * std::numbers::pi - 5.5))) < 1e-14, "wrap-around");
    expect(nested.depth() == compiled.nested_depth() && nested.rows() == n && nested.cols() == n, "nested buffers");
    expect(nested.level(0, n, n).rows() == n, "nested level view");

    // [gradient]
    // Gradient with respect to log(lengthscale), parameter 1: d/d(log l) [2 exp(-r / l)] = 2 exp(-r / l) r / l.
    std::vector<double> d_k(n * n);
    std::vector<double> scratch_b(n * n);
    check(compiled.grad(store, std::nullopt, MatrixMut<double>(d_k.data(), n, n), 1, Triangle::Full,
                        MatrixMut<double>(scratch.data(), n, n), MatrixMut<double>(scratch_b.data(), n, n), nested));
    // [end]
    expect(std::abs(d_k[1] - 2.0 * std::exp(-0.5) * 0.5) < 1e-14, "gradient");

    // [cross]
    // Cross-covariance between two query angles and the training angles.
    const std::vector<double> queries{1.0, 4.0};
    const std::vector<double> d2_cross = squared_arc_table(queries, angles);
    DistanceSource cross_source;
    cross_source.bind_scalar(arc_slot.slot(), MatrixRef<double>(d2_cross.data(), 2, n));
    const auto prepared_cross = compiled.prepare_cross(cross_source);
    const CheckedDistanceStore<double>& cross_store = borrow(prepared_cross);
    std::vector<double> cross(2 * n);
    std::vector<double> cross_scratch(2 * n);
    check(compiled.apply_cross(cross_store, std::nullopt, std::nullopt, MatrixMut<double>(cross.data(), 2, n),
                               MatrixMut<double>(cross_scratch.data(), 2, n)));
    // [end]
    std::cout << "k(1.0, 0.0) = " << cross[0] << " (hand " << 2.0 * std::exp(-1.0) << ")\n";
    expect(std::abs(cross[0] - 2.0 * std::exp(-1.0)) < 1e-14, "cross-covariance without white noise");

    // Prior variance without any table: stationary leaves are evaluated at zero distance.
    std::vector<double> diag(n);
    check(compiled.fill_diag(n, std::nullopt, diag));
    expect(std::abs(diag[2] - 2.1) < 1e-14, "fill_diag");

    // [tidy]
    // A table with rounding noise: a nonzero diagonal and a slightly asymmetric pair.
    std::vector<double> noisy = d2;
    noisy[0] = 1e-12;
    noisy[1] += 1e-12;
    DistanceSource strict;
    strict.bind_scalar(arc_slot.slot(), MatrixRef<double>(noisy.data(), n, n));
    const auto rejected = compiled.prepare_square(strict);
    std::cout << "strict: " << (rejected ? "accepted" : std::string(rejected.error().message())) << '\n';
    DistanceSource lenient;
    lenient.bind_scalar(arc_slot.slot(), MatrixRef<double>(noisy.data(), n, n)).tidy(1e-9);
    const auto repaired = compiled.prepare_square(lenient);
    std::cout << "tidy(1e-9): " << (repaired ? "accepted" : std::string(repaired.error().message()))
              << ", tidy_tolerance = " << *lenient.tidy_tolerance() << '\n';
    // [end]
    expect(!rejected && rejected.error().code() == GprErrorCode::InvalidDistance, "strict validation");
    expect(repaired.has_value() && !strict.tidy_tolerance().has_value(), "tidy repair");

    // A missing table is reported as a slot error.
    const DistanceSource empty;
    const auto missing = compiled.prepare_square(empty);
    expect(!missing && missing.error().code() == GprErrorCode::DistanceSlot, "missing slot");

    // [ard]
    // ARD: one squared-difference table per feature. Two features, three points.
    const std::array<double, 3> t{0.0, 1.0, 3.0};
    const std::array<double, 3> s{0.0, 2.0, 1.0};
    std::array<double, 9> d2_t{};
    std::array<double, 9> d2_s{};
    for (std::size_t j = 0; j < 3; ++j)
    {
        for (std::size_t i = 0; i < 3; ++i)
        {
            d2_t[i + 3 * j] = (t[i] - t[j]) * (t[i] - t[j]);
            d2_s[i + 3 * j] = (s[i] - s[j]) * (s[i] - s[j]);
        }
    }
    const ArdDistance ard_slot = unwrap(ArdDistance::create(2));
    const auto ard_kernel =
        unwrap(ard_slot.rbf_ard(unwrap(RbfArdKernel::create(std::array<double, 2>{1.0, 2.0}))));
    const auto ard_compiled = ard_kernel.compile<double>();
    const std::array<MatrixRef<double>, 2> tables{MatrixRef<double>(d2_t.data(), 3, 3),
                                                  MatrixRef<double>(d2_s.data(), 3, 3)};
    DistanceSource ard_source;
    ard_source.bind_ard(ard_slot.slot(), tables);
    const auto prepared_ard = ard_compiled.prepare_square(ard_source);
    const CheckedDistanceStore<double>& ard_store = borrow(prepared_ard);
    std::array<double, 9> ard_gram{};
    std::array<double, 9> ard_scratch{};
    check(ard_compiled.apply(ard_store, std::nullopt, MatrixMut<double>(ard_gram.data(), 3, 3), Triangle::Full,
                             MatrixMut<double>(ard_scratch.data(), 3, 3)));
    // [end]
    std::cout << "ARD K(1,0) = " << ard_gram[1] << " (hand " << std::exp(-1.0) << "), dims = " << ard_slot.dims() << '\n';
    expect(std::abs(ard_gram[1] - std::exp(-1.0)) < 1e-14 && ard_slot.dims() == 2, "ARD from per-feature tables");
    const auto wrong = ard_slot.matern_ard(unwrap(MaternArdKernel::create(std::array<double, 3>{1.0, 1.0, 1.0}, MaternNu::FiveHalves)));
    expect(!wrong && wrong.error().code() == GprErrorCode::DimensionMismatch, "ARD leaf must match the slot");
    expect(!ArdDistance::create(0) && ArdDistance::create(0).error().code() == GprErrorCode::EmptyInput, "dims > 0");
    // The other leaf factories of the two slot types.
    const auto rbf_k = arc_slot.rbf(unwrap(RbfKernel::create(1.0)));
    const auto per_k = arc_slot.periodic(unwrap(PeriodicKernel::create(1.0, 6.0)));
    const auto rq_k = arc_slot.rational_quadratic(unwrap(RationalQuadraticKernel::create(1.0, 1.0)));
    const auto rq_ard_k = ard_slot.rational_quadratic_ard(unwrap(RationalQuadraticArdKernel::create(std::array<double, 2>{1.0, 1.0}, 1.0)));
    expect(rbf_k.num_params() == 1 && per_k.num_params() == 2 && rq_k.num_params() == 2 && rq_ard_k.has_value(),
           "slot leaf factories");

    // [with-points]
    // Mix the supplied arc distance with a coordinate kernel on a numeric covariate (here the index of each point).
    const std::vector<double> index{0.0, 1.0, 2.0, 3.0, 4.0};
    const DistanceKernel<WithPoints> mixed =
        arc_slot.matern(unwrap(MaternKernel::create(1.0, MaternNu::OneHalf))) +
        KernelSpec(unwrap(LinearKernel::create(0.5)));
    const auto mixed_compiled = mixed.compile<double>();
    std::vector<double> mixed_gram(n * n);
    check(mixed_compiled.apply(store, MatrixRef<double>(index.data(), n, 1), MatrixMut<double>(mixed_gram.data(), n, n),
                               Triangle::Full, MatrixMut<double>(scratch.data(), n, n)));
    // [end]
    // k(0, 1) = exp(-0.5) + 0.5 * (0 * 1) and k(1, 2) = exp(-1.0) + 0.5 * (1 * 2).
    std::cout << "mixed requires_points = " << mixed_compiled.requires_points() << ", K(2,1) = " << mixed_gram[7] << '\n';
    expect(mixed_compiled.requires_points() && std::abs(mixed_gram[1] - std::exp(-0.5)) < 1e-14 &&
               std::abs(mixed_gram[7] - (std::exp(-1.0) + 0.5 * 2.0)) < 1e-14,
           "supplied distance plus coordinate kernel");
    const auto promoted = kernel.with_points();
    expect(promoted.num_params() == kernel.num_params(), "with_points keeps the parameters");

    // The coordinate evaluation of the same tree without the supplied table is rejected.
    const auto no_points = mixed_compiled.apply(store, std::nullopt, MatrixMut<double>(mixed_gram.data(), n, n),
                                                Triangle::Full, MatrixMut<double>(scratch.data(), n, n));
    expect(!no_points && no_points.error().code() == GprErrorCode::UnsupportedKernelOperation, "points required");

    // [cache]
    // DistanceCache: precompute squared distances and per-dimension squared differences of coordinates once.
    const std::array<double, 6> x_data{0.0, 1.0, 3.0, 0.0, 2.0, 1.0};  // the same t and s as above, 3 rows x 2 cols
    const MatrixRef<double> x(x_data.data(), 3, 2);
    DistanceCache<double> cache = unwrap(DistanceCache<double>::build(x, true, true, DistanceCachePolicy::Cached));
    std::cout << "cache: points = " << cache.num_points() << ", dims = " << cache.num_dims()
              << ", has_dist = " << cache.has_dist() << ", has_ard = " << cache.has_ard()
              << ", packed size = " << cache.ard_packed().size() << '\n';
    const auto cached_kernel = KernelSpec(unwrap(RbfArdKernel::create(std::array<double, 2>{1.0, 2.0}))).compile<double>();
    std::array<double, 9> cached_gram{};
    check(cached_kernel.apply_cached(x, cache, MatrixMut<double>(cached_gram.data(), 3, 3), Triangle::Lower,
                                     MatrixMut<double>(ard_scratch.data(), 3, 3)));
    // [end]
    expect(cache.has_dist() && cache.has_ard() && cache.ard_packed().size() == 2 * 3 * 4 / 2, "cache sizes");
    expect(std::abs(cache.dist_view()(1, 0) - (1.0 + 4.0)) < 1e-15, "cached squared distance");
    // Packed layout: dimension block 1 (feature s), column 0, row 1 -> (s1 - s0)^2 = 4 at offset 6 + 0 + 1.
    expect(std::abs(cache.ard_packed()[6 + 1] - 4.0) < 1e-15, "packed ARD layout");
    expect(std::abs(cached_gram[1] - ard_gram[1]) < 1e-14, "cached evaluation equals distance-table evaluation");
    // Uncached policy: ensure() clears the buffers again and reset() empties the cache.
    check(cache.ensure(x, true, true, DistanceCachePolicy::Uncached));
    expect(!cache.has_dist() && !cache.has_ard() && cache.policy() == DistanceCachePolicy::Uncached, "uncached");
    cache.reset();
    expect(cache.num_points() == 0 && cache.num_dims() == 0 && cache.dist_mut().is_empty(), "reset");

    return all_ok ? 0 : 1;
}
