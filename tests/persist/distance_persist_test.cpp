// SPDX-License-Identifier: Apache-2.0
// Spec: FMT-07, FMT-08, REQ-05
/// @file distance_persist_test.cpp
/// @brief Persistence tests for supplied-distance models (save, save_with_factor, LoadedDistanceGpr).
/// @details Verifies bit-identical predictions after a round trip for every precision policy and both distance
/// modes, the refactor-on-load path, fresh slot issuance, typed extraction, and the rejection of foreign, malformed
/// and tampered checkpoints.

#include "cxxgpr/core/rng.hpp"
#include "cxxgpr/gpr/distance_gpr.hpp"
#include "cxxgpr/gpr/gpr.hpp"
#include "cxxgpr/kernel/kernel.hpp"
#include "cxxgpr/persist/persist.hpp"
#include "cxxgpr/transform/transform.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

namespace
{

    /// @brief RAII temporary directory removed on destruction.
    struct TempDir
    {
        /// @brief Managed directory.
        std::filesystem::path path;

        /// @brief Creates a fresh directory named after the test.
        /// @param[in] name Unique suffix.
        explicit TempDir(const std::string& name)
        {
            path = std::filesystem::temp_directory_path() / ("cxxgpr_distance_persist_" + name);
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
            std::filesystem::create_directories(path, ec);
        }

        /// @brief Removes the directory.
        ~TempDir()
        {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }

        /// @brief Not copyable.
        TempDir(const TempDir&) = delete;
        /// @brief Not copyable.
        /// @return Never returns; deleted.
        TempDir& operator=(const TempDir&) = delete;
    };

    /// @brief Training, cross and query squared-distance tables of one data set, in total and per dimension.
    struct Fixture
    {
        /// @brief Number of training points.
        std::size_t n{0};
        /// @brief Number of query points.
        std::size_t m{0};
        /// @brief Coordinate dimension.
        std::size_t d{2};
        /// @brief Training coordinates, column-major n x d.
        std::vector<double> x;
        /// @brief Query coordinates, column-major m x d.
        std::vector<double> q;
        /// @brief Training targets.
        std::vector<double> y;
        /// @brief Tables indexed by [which][dim], where which is 0 train, 1 cross, 2 query and dim == d is the total.
        std::vector<std::vector<std::vector<double>>> tables;
    };

    /// @brief Squared distance table between two point sets for one dimension, or for all of them.
    /// @param[in] a Row points, column-major na x d.
    /// @param[in] na Row count.
    /// @param[in] b Column points, column-major nb x d.
    /// @param[in] nb Column count.
    /// @param[in] d Dimension count.
    /// @param[in] dim Dimension to use, or d for all.
    /// @return Column-major na x nb table.
    std::vector<double> make_table(
        const std::vector<double>& a,
        std::size_t na,
        const std::vector<double>& b,
        std::size_t nb,
        std::size_t d,
        std::size_t dim
    )
    {
        std::vector<double> out(na * nb, 0.0);
        for (std::size_t j = 0; j < nb; ++j)
        {
            for (std::size_t i = 0; i < na; ++i)
            {
                double acc = 0.0;
                for (std::size_t k = 0; k < d; ++k)
                {
                    if (dim == d || dim == k)
                    {
                        const double diff = a[(k * na) + i] - b[(k * nb) + j];
                        acc += diff * diff;
                    }
                }
                out[(j * na) + i] = acc;
            }
        }
        return out;
    }

    /// @brief Builds the data set and all distance tables.
    /// @param[in] n Training points.
    /// @param[in] m Query points.
    /// @return Filled fixture.
    Fixture make_fixture(std::size_t n, std::size_t m)
    {
        Fixture f;
        f.n = n;
        f.m = m;
        f.x.resize(n * f.d);
        f.q.resize(m * f.d);
        f.y.resize(n);
        cxxgpr::Xoshiro256PlusPlus rng(23);
        for (double& v : f.x)
        {
            v = 2.0 * rng.unit();
        }
        for (double& v : f.q)
        {
            v = 2.0 * rng.unit();
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            f.y[i] = std::sin(1.4 * f.x[i]) + (0.5 * std::cos(2.0 * f.x[n + i])) + (0.05 * rng.unit_normal());
        }
        f.tables.resize(3);
        for (std::size_t dim = 0; dim <= f.d; ++dim)
        {
            f.tables[0].push_back(make_table(f.x, n, f.x, n, f.d, dim));
            f.tables[1].push_back(make_table(f.x, n, f.q, m, f.d, dim));
            f.tables[2].push_back(make_table(f.q, m, f.q, m, f.d, dim));
        }
        return f;
    }

    /// @brief Binds the tables of one kind to the slots of a kernel, in slot order.
    /// @details A scalar slot receives the total table and an ARD slot the per-dimension tables.
    class Binder
    {
    public:
        /// @brief Binds the tables.
        /// @param[in] f Fixture holding the tables.
        /// @param[in] which 0 for training, 1 for cross, 2 for query tables.
        /// @param[in] slots Slot descriptors of the kernel to bind.
        Binder(const Fixture& f, std::size_t which, const std::vector<cxxgpr::SlotDescriptor>& slots)
        {
            const std::size_t rows = which == 2 ? f.m : f.n;
            const std::size_t cols = which == 0 ? f.n : f.m;
            refs_.reserve(slots.size());
            for (const auto& slot : slots)
            {
                if (slot.ard_dims.has_value())
                {
                    const std::size_t ard_dims = slot.ard_dims.value_or(0U);
                    std::vector<cxxgpr::MatrixRef<double>> dims;
                    dims.reserve(ard_dims);
                    for (std::size_t k = 0; k < ard_dims; ++k)
                    {
                        dims.emplace_back(f.tables[which][k].data(), rows, cols);
                    }
                    refs_.push_back(std::move(dims));
                    source_.bind_ard(slot.id, refs_.back());
                }
                else
                {
                    refs_.push_back({cxxgpr::MatrixRef<double>(f.tables[which][f.d].data(), rows, cols)});
                    source_.bind_scalar(slot.id, refs_.back().front());
                }
            }
        }

        /// @brief Returns the bound source.
        /// @return Source referring to the fixture tables.
        [[nodiscard]] const cxxgpr::DistanceSource& source() const noexcept { return source_; }

    private:
        /// @brief Matrix views kept alive for the source.
        std::vector<std::vector<cxxgpr::MatrixRef<double>>> refs_;
        /// @brief Source bound to the slots.
        cxxgpr::DistanceSource source_;
    };

    /// @brief Kernel of the DistanceOnly tests: a scaled scalar RBF plus white noise.
    /// @return Kernel with one scalar slot.
    cxxgpr::DistanceKernel<cxxgpr::DistanceOnly> only_kernel()
    {
        const cxxgpr::ScalarDistance slot;
        return (cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::ConstantKernel::create(1.2).value())
                   * slot.rbf(cxxgpr::RbfKernel::create(0.9).value()))
            + cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::WhiteKernel::create(0.02).value());
    }

    /// @brief Kernel of the WithPoints tests: scalar and ARD distance leaves plus a coordinate Matern leaf.
    /// @return Kernel with a scalar slot, an ARD slot and a coordinate leaf.
    cxxgpr::DistanceKernel<cxxgpr::WithPoints> points_kernel()
    {
        const cxxgpr::ScalarDistance scalar;
        const auto ard = cxxgpr::ArdDistance::create(2).value();
        const std::array<double, 2> ls{0.8, 1.3};
        const auto distance_part =
            (cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::ConstantKernel::create(0.9).value())
                * scalar.matern(cxxgpr::MaternKernel::create(1.1, cxxgpr::MaternNu::FiveHalves).value()))
            + ard.rbf_ard(cxxgpr::RbfArdKernel::create(ls).value()).value();
        return distance_part
            + cxxgpr::KernelSpec(cxxgpr::MaternKernel::create(0.7, cxxgpr::MaternNu::ThreeHalves).value());
    }

    /// @brief Builds a trainer of the requested mode.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode.
    /// @return Trainer with a noise variance, and transforms for WithPoints.
    template <typename P, typename C> auto make_trainer()
    {
        if constexpr (std::same_as<C, cxxgpr::DistanceOnly>)
        {
            return cxxgpr::DistanceGpr<cxxgpr::optimizer::Lbfgs, P, C>(only_kernel()).with_noise_variance(0.05).value();
        }
        else
        {
            return cxxgpr::DistanceGpr<cxxgpr::optimizer::Lbfgs, P, C>(points_kernel())
                .with_noise_variance(0.05)
                .value()
                .with_input_transform(cxxgpr::transform::StandardizeInput{})
                .with_target_transform(cxxgpr::transform::StandardizeTarget{});
        }
    }

    /// @brief Fits a model of the requested policy and mode on the fixture.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode.
    /// @param[in] f Fixture.
    /// @return Fitted model.
    template <typename P, typename C> auto fit_model(const Fixture& f)
    {
        const auto trainer = make_trainer<P, C>();
        const Binder train(f, 0, trainer.kernel().slots());
        if constexpr (std::same_as<C, cxxgpr::DistanceOnly>)
        {
            return trainer.fit(train.source(), f.y);
        }
        else
        {
            return trainer.fit(train.source(), cxxgpr::MatrixRef<double>(f.x.data(), f.n, f.d), f.y);
        }
    }

    /// @brief Compares two sequences for exact equality after widening to double.
    /// @tparam A First element type.
    /// @tparam B Second element type.
    /// @param[in] a First sequence.
    /// @param[in] b Second sequence.
    /// @return True when sizes match and every element is bit-identical after widening.
    template <typename A, typename B> bool identical(const std::vector<A>& a, const std::vector<B>& b)
    {
        if (a.size() != b.size())
        {
            return false;
        }
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            const auto x = std::bit_cast<std::uint64_t>(static_cast<double>(a[i]));
            const auto y = std::bit_cast<std::uint64_t>(static_cast<double>(b[i]));
            if (x != y)
            {
                return false;
            }
        }
        return true;
    }

    /// @brief Compares two sequences with a relative tolerance after widening to double.
    /// @tparam A First element type.
    /// @tparam B Second element type.
    /// @param[in] a First sequence.
    /// @param[in] b Second sequence.
    /// @return Largest absolute difference scaled by the larger of one and the largest magnitude.
    template <typename A, typename B> double rel_diff(const std::vector<A>& a, const std::vector<B>& b)
    {
        double scale = 1.0;
        double diff = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            scale = std::max(scale, std::abs(static_cast<double>(a[i])));
            diff = std::max(diff, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
        }
        return diff / scale;
    }

    /// @brief Saves a model with its factor, loads it, and requires bit-identical results of every query.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode.
    /// @param[in] name Directory suffix.
    template <typename P, typename C> void expect_bit_identical_round_trip(const std::string& name)
    {
        const Fixture f = make_fixture(26, 6);
        auto model = fit_model<P, C>(f);
        ASSERT_TRUE(model.has_value()) << model.error().message();
        const TempDir dir(name);
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, dir.path).has_value());
        auto loaded = cxxgpr::persist::LoadedDistanceGpr<C>::load(dir.path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error().message();
        EXPECT_TRUE(loaded->has_factor());
        EXPECT_EQ(loaded->n(), f.n);
        constexpr bool with_points = std::same_as<C, cxxgpr::WithPoints>;
        EXPECT_EQ(loaded->d(), with_points ? f.d : 0U);
        EXPECT_EQ(loaded->precision_kind(), P::persist_kind());
        EXPECT_EQ(loaded->factor_jitter(), model->factor_jitter());

        const Binder cross_old(f, 1, model->kernel().slots());
        const Binder query_old(f, 2, model->kernel().slots());
        const Binder cross_new(f, 1, loaded->kernel().slots());
        const Binder query_new(f, 2, loaded->kernel().slots());
        const cxxgpr::MatrixRef<double> xq(f.q.data(), f.m, f.d);
        auto call = [&](const auto& m, const Binder& c, const Binder& q)
        {
            if constexpr (std::same_as<C, cxxgpr::DistanceOnly>)
            {
                return std::tuple(
                    m.predict(c.source()), m.predict_covariance(c.source(), q.source()),
                    m.sample(c.source(), q.source(), 3U, 17U), m.loo_predict()
                );
            }
            else
            {
                return std::tuple(
                    m.predict(c.source(), xq), m.predict_covariance(c.source(), q.source(), xq),
                    m.sample(c.source(), q.source(), xq, 3U, 17U), m.loo_predict()
                );
            }
        };
        const auto [p0, c0, s0, l0] = call(*model, cross_old, query_old);
        const auto [p1, c1, s1, l1] = call(*loaded, cross_new, query_new);
        ASSERT_TRUE(p0.has_value() && c0.has_value() && s0.has_value() && l0.has_value());
        ASSERT_TRUE(p1.has_value()) << p1.error().message();
        ASSERT_TRUE(c1.has_value() && s1.has_value() && l1.has_value());
        EXPECT_TRUE(identical(p0->mean, p1->mean));
        EXPECT_TRUE(identical(p0->variance, p1->variance));
        EXPECT_TRUE(identical(c0->mean, c1->mean));
        EXPECT_TRUE(identical(c0->covariance, c1->covariance));
        EXPECT_TRUE(identical(*s0, *s1));
        EXPECT_TRUE(identical(l0->mean, l1->mean));
        EXPECT_TRUE(identical(l0->variance, l1->variance));
    }

    /// @brief Round trip of a double-precision DistanceOnly model gives bit-identical results.
    TEST(DistancePersistTest, DoubleDistanceOnlyRoundTripIsBitIdentical)
    {
        // Spec: FMT-07
        expect_bit_identical_round_trip<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>("double_only");
    }

    /// @brief Round trip of a double-precision WithPoints model (scalar, ARD and coordinate leaves, transforms).
    TEST(DistancePersistTest, DoubleWithPointsRoundTripIsBitIdentical)
    {
        // Spec: FMT-07
        expect_bit_identical_round_trip<cxxgpr::DoublePrecision, cxxgpr::WithPoints>("double_points");
    }

    /// @brief Round trip of a single-precision model stores the float tables and restores the float state.
    TEST(DistancePersistTest, SinglePrecisionRoundTripIsBitIdentical)
    {
        // Spec: FMT-07
        expect_bit_identical_round_trip<cxxgpr::SinglePrecision, cxxgpr::DistanceOnly>("single_only");
        expect_bit_identical_round_trip<cxxgpr::SinglePrecision, cxxgpr::WithPoints>("single_points");
    }

    /// @brief Round trip of both mixed-precision residual formulas.
    TEST(DistancePersistTest, MixedPrecisionRoundTripIsBitIdentical)
    {
        // Spec: FMT-07
        expect_bit_identical_round_trip<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>, cxxgpr::DistanceOnly>(
            "mixed_p"
        );
        expect_bit_identical_round_trip<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>, cxxgpr::WithPoints>(
            "mixed_r"
        );
    }

    /// @brief A checkpoint saved without the factor is factored again on load and agrees to rounding.
    TEST(DistancePersistTest, SaveWithoutFactorRefactorsOnLoad)
    {
        // Spec: FMT-07
        const Fixture f = make_fixture(24, 5);
        auto model = fit_model<cxxgpr::DoublePrecision, cxxgpr::WithPoints>(f);
        ASSERT_TRUE(model.has_value());
        const TempDir dir("no_factor");
        ASSERT_TRUE(cxxgpr::persist::save(*model, dir.path).has_value());
        auto loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::WithPoints>::load(dir.path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error().message();
        EXPECT_FALSE(loaded->has_factor());
        const Binder cross_old(f, 1, model->kernel().slots());
        const Binder cross_new(f, 1, loaded->kernel().slots());
        const cxxgpr::MatrixRef<double> xq(f.q.data(), f.m, f.d);
        auto a = model->predict(cross_old.source(), xq);
        auto b = loaded->predict(cross_new.source(), xq);
        ASSERT_TRUE(a.has_value() && b.has_value());
        EXPECT_LT(rel_diff(a->mean, b->mean), 1e-10);
        EXPECT_LT(rel_diff(a->variance, b->variance), 1e-10);
    }

    /// @brief Loading issues fresh slots, so tables bound to the slots used before saving are rejected.
    TEST(DistancePersistTest, LoadedKernelHasFreshSlots)
    {
        // Spec: FMT-07
        const Fixture f = make_fixture(20, 4);
        auto model = fit_model<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(f);
        ASSERT_TRUE(model.has_value());
        const TempDir dir("fresh_slots");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, dir.path).has_value());
        auto loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path);
        ASSERT_TRUE(loaded.has_value());
        const auto old_slots = model->kernel().slots();
        const auto new_slots = loaded->kernel().slots();
        ASSERT_EQ(old_slots.size(), new_slots.size());
        ASSERT_FALSE(old_slots.empty());
        EXPECT_NE(old_slots[0].id, new_slots[0].id);
        const Binder stale(f, 1, old_slots);
        auto rejected = loaded->predict(stale.source());
        ASSERT_FALSE(rejected.has_value());
        // A slot that belongs to another kernel is not accepted.
        EXPECT_EQ(rejected.error().code(), cxxgpr::GprErrorCode::UnsupportedKernelOperation);
        // Loading the same checkpoint twice issues different slots again.
        auto again = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path);
        ASSERT_TRUE(again.has_value());
        EXPECT_NE(again->kernel().slots()[0].id, new_slots[0].id);
    }

    /// @brief into_fitted shares the restored state, refits, and rejects a different precision policy.
    TEST(DistancePersistTest, IntoFittedRestoresATypedModel)
    {
        // Spec: FMT-07
        const Fixture f = make_fixture(22, 5);
        auto model = fit_model<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(f);
        ASSERT_TRUE(model.has_value());
        const TempDir dir("into_fitted");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, dir.path).has_value());
        auto loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path);
        ASSERT_TRUE(loaded.has_value());

        auto typed = loaded->into_fitted<cxxgpr::optimizer::Lbfgs, cxxgpr::DoublePrecision>();
        ASSERT_TRUE(typed.has_value()) << typed.error().message();
        const Binder cross(f, 1, typed->kernel().slots());
        const Binder cross_old(f, 1, model->kernel().slots());
        auto a = typed->predict(cross.source());
        auto b = model->predict(cross_old.source());
        ASSERT_TRUE(a.has_value() && b.has_value());
        EXPECT_TRUE(identical(a->mean, b->mean));
        EXPECT_EQ(typed->n_train(), model->n_train());
        EXPECT_NEAR(typed->neg_log_marginal_likelihood(), model->neg_log_marginal_likelihood(), 1e-9);

        const Binder train(f, 0, typed->kernel().slots());
        EXPECT_TRUE(typed->refit(train.source(), f.y).has_value());

        auto wrong = loaded->into_fitted<cxxgpr::optimizer::Lbfgs, cxxgpr::SinglePrecision>();
        ASSERT_FALSE(wrong.has_value());
        EXPECT_EQ(wrong.error().persist_error_kind(), cxxgpr::PersistErrorKind::WrongModel);
    }

    /// @brief Copies of a loaded model answer the same and own independent query buffers.
    TEST(DistancePersistTest, CopiesOfALoadedModelAreIndependent)
    {
        // Spec: FMT-07
        const Fixture f = make_fixture(20, 4);
        auto model = fit_model<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(f);
        ASSERT_TRUE(model.has_value());
        const TempDir dir("copies");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, dir.path).has_value());
        auto loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path);
        ASSERT_TRUE(loaded.has_value());
        auto copy = *loaded;
        const Binder cross(f, 1, loaded->kernel().slots());
        auto a = loaded->predict(cross.source());
        auto b = copy.predict(cross.source());
        ASSERT_TRUE(a.has_value() && b.has_value());
        EXPECT_TRUE(identical(a->mean, b->mean));
    }

    /// @brief A checkpoint of one kind is rejected by the loaders of the others with WrongModel.
    TEST(DistancePersistTest, ForeignCheckpointsAreRejectedWithWrongModel)
    {
        // Spec: FMT-07, FMT-08
        const Fixture f = make_fixture(20, 4);
        auto dist_only = fit_model<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(f);
        auto dist_points = fit_model<cxxgpr::DoublePrecision, cxxgpr::WithPoints>(f);
        ASSERT_TRUE(dist_only.has_value() && dist_points.has_value());
        const TempDir only_dir("foreign_only");
        const TempDir points_dir("foreign_points");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*dist_only, only_dir.path).has_value());
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*dist_points, points_dir.path).has_value());

        auto as_coordinate = cxxgpr::persist::LoadedGpr::load(only_dir.path);
        ASSERT_FALSE(as_coordinate.has_value());
        EXPECT_EQ(as_coordinate.error().persist_error_kind(), cxxgpr::PersistErrorKind::WrongModel);
        auto points_as_only = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(points_dir.path);
        ASSERT_FALSE(points_as_only.has_value());
        EXPECT_EQ(points_as_only.error().persist_error_kind(), cxxgpr::PersistErrorKind::WrongModel);
        auto only_as_points = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::WithPoints>::load(only_dir.path);
        ASSERT_FALSE(only_as_points.has_value());
        EXPECT_EQ(only_as_points.error().persist_error_kind(), cxxgpr::PersistErrorKind::WrongModel);

        const auto coordinate =
            cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value()))
                .factor(f.x, f.n, f.d, f.y);
        ASSERT_TRUE(coordinate.has_value());
        const TempDir coord_dir("foreign_coord");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*coordinate, coord_dir.path).has_value());
        auto coord_as_distance = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(coord_dir.path);
        ASSERT_FALSE(coord_as_distance.has_value());
        EXPECT_EQ(coord_as_distance.error().persist_error_kind(), cxxgpr::PersistErrorKind::WrongModel);
    }

    /// @brief Reads a whole file.
    /// @param[in] path File to read.
    /// @return File content.
    std::string slurp(const std::filesystem::path& path)
    {
        std::ifstream in(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }

    /// @brief Overwrites a whole file.
    /// @param[in] path File to write.
    /// @param[in] text New content.
    void spit(const std::filesystem::path& path, const std::string& text)
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << text;
    }

    /// @brief Replaces the first occurrence of a text in config.json.
    /// @param[in] dir Checkpoint directory.
    /// @param[in] from Text to find.
    /// @param[in] to Replacement.
    /// @return True when the text was found.
    bool edit_config(const std::filesystem::path& dir, const std::string& from, const std::string& to)
    {
        std::string text = slurp(dir / "config.json");
        const auto at = text.find(from);
        if (at == std::string::npos)
        {
            return false;
        }
        text.replace(at, from.size(), to);
        spit(dir / "config.json", text);
        return true;
    }

    /// @brief Overwrites one double inside the data of a named tensor of model.safetensors.
    /// @param[in] dir Checkpoint directory.
    /// @param[in] tensor Tensor name.
    /// @param[in] index Element index inside the tensor.
    /// @param[in] value New value.
    /// @return True when the tensor was found and patched.
    bool patch_tensor(const std::filesystem::path& dir, const std::string& tensor, std::size_t index, double value)
    {
        std::string bytes = slurp(dir / "model.safetensors");
        std::uint64_t header_len = 0;
        std::memcpy(&header_len, bytes.data(), sizeof(header_len));
        const std::string header = bytes.substr(8, header_len);
        const auto key = header.find("\"" + tensor + "\"");
        if (key == std::string::npos)
        {
            return false;
        }
        const auto offsets = header.find("\"data_offsets\":[", key);
        if (offsets == std::string::npos)
        {
            return false;
        }
        const std::size_t begin = std::stoull(header.substr(offsets + 16));
        const std::size_t at = 8 + header_len + begin + (index * sizeof(double));
        std::memcpy(bytes.data() + at, &value, sizeof(value));
        spit(dir / "model.safetensors", bytes);
        return true;
    }

    /// @brief Malformed distance sections of config.json are rejected with Config.
    TEST(DistancePersistTest, MalformedConfigIsRejected)
    {
        // Spec: FMT-07, FMT-08
        const Fixture f = make_fixture(18, 4);
        auto model = fit_model<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(f);
        ASSERT_TRUE(model.has_value());
        auto expect_config_error = [&](const std::string& name, const std::string& from, const std::string& to)
        {
            const TempDir dir("bad_" + name);
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, dir.path).has_value());
            ASSERT_TRUE(edit_config(dir.path, from, to)) << name;
            auto loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path);
            ASSERT_FALSE(loaded.has_value()) << name;
            EXPECT_EQ(loaded.error().persist_error_kind(), cxxgpr::PersistErrorKind::Config) << name;
        };
        // A distance-only model cannot claim coordinates.
        expect_config_error("d", R"("d": 0)", R"("d": 2)");
        // A leaf that reads a slot which does not exist.
        expect_config_error("slot", R"("slot": 0)", R"("slot": 5)");
        // A slot list that disagrees with the kernel tree.
        expect_config_error("slot_kind", R"("kind": "scalar")", R"("kind": "ard", "dims": 2)");
        // An unknown mode.
        expect_config_error("points", R"("distance_only")", R"("sideways")");
    }

    /// @brief Tampered distance tables are rejected: invalid values by table validation, non-finite ones by the reader.
    TEST(DistancePersistTest, TamperedDistanceTablesAreRejected)
    {
        // Spec: FMT-07, FMT-08
        const Fixture f = make_fixture(18, 4);
        auto model = fit_model<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(f);
        ASSERT_TRUE(model.has_value());
        auto load_tampered = [&](const std::string& name, std::size_t index, double value)
        {
            const TempDir dir("tamper_" + name);
            EXPECT_TRUE(cxxgpr::persist::save_with_factor(*model, dir.path).has_value());
            EXPECT_TRUE(patch_tensor(dir.path, "d2.0", index, value)) << name;
            return cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path);
        };
        // Packed position 0 is the diagonal entry (0, 0); position 1 is the entry (1, 0).
        auto diagonal = load_tampered("diagonal", 0, 0.5);
        ASSERT_FALSE(diagonal.has_value());
        EXPECT_EQ(diagonal.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
        auto negative = load_tampered("negative", 1, -1.0);
        ASSERT_FALSE(negative.has_value());
        EXPECT_EQ(negative.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
        // A non-finite value never reaches the distance validation: the tensor reader rejects it first.
        for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
        {
            auto non_finite = load_tampered("non_finite", 1, bad);
            ASSERT_FALSE(non_finite.has_value());
            EXPECT_EQ(non_finite.error().persist_error_kind(), cxxgpr::PersistErrorKind::Tensor);
        }
    }

    /// @brief A missing distance tensor is a Tensor error.
    TEST(DistancePersistTest, MissingDistanceTensorIsRejected)
    {
        // Spec: FMT-07, FMT-08
        const Fixture f = make_fixture(18, 4);
        auto model = fit_model<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(f);
        ASSERT_TRUE(model.has_value());
        const TempDir dir("missing_tensor");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, dir.path).has_value());
        // Rename the tensor in the header; the header length is unchanged because the name keeps its length.
        std::string bytes = slurp(dir.path / "model.safetensors");
        const auto at = bytes.find("\"d2.0\"");
        ASSERT_NE(at, std::string::npos);
        bytes.replace(at, 6, "\"d9.0\"");
        spit(dir.path / "model.safetensors", bytes);
        auto loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path);
        ASSERT_FALSE(loaded.has_value());
        EXPECT_EQ(loaded.error().persist_error_kind(), cxxgpr::PersistErrorKind::Tensor);
    }

    /// @brief Saving into an unwritable location reports an Io error and leaves the model usable.
    TEST(DistancePersistTest, SaveReportsIoFailure)
    {
        // Spec: FMT-08
        const Fixture f = make_fixture(16, 4);
        auto model = fit_model<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(f);
        ASSERT_TRUE(model.has_value());
        const TempDir dir("io_failure");
        // A regular file where a directory is required.
        spit(dir.path / "blocker", "x");
        auto saved = cxxgpr::persist::save(*model, dir.path / "blocker" / "child");
        ASSERT_FALSE(saved.has_value());
        EXPECT_EQ(saved.error().persist_error_kind(), cxxgpr::PersistErrorKind::Io);
        const Binder cross(f, 1, model->kernel().slots());
        EXPECT_TRUE(model->predict(cross.source()).has_value());
    }

}  // namespace
