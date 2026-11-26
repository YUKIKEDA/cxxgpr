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
#include <expected>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "small_stack.hpp"

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

    /// @brief Copies of a loaded model answer the same and own independent query buffers.
    /// @details Interleaves batches of different sizes on a copy-assigned copy and on the original; each answer
    /// must equal (bitwise) that of a model loaded fresh from the same directory.
    TEST(DistancePersistTest, CopiesOfALoadedModelAreIndependent)
    {
        // Spec: FMT-07
        const Fixture f = make_fixture(20, 4);
        // The same training points (x is drawn before the queries) with a single query point.
        const Fixture small = make_fixture(f.n, 1);
        auto model = fit_model<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(f);
        ASSERT_TRUE(model.has_value());
        const TempDir dir("copies");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, dir.path).has_value());
        auto loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path);
        auto fresh = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path);
        auto other = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path);
        if (!loaded.has_value() || !fresh.has_value() || !other.has_value())
        {
            FAIL() << "load failed";
        }
        auto copy = *other;
        copy = *loaded;
        const Binder cross(f, 1, loaded->kernel().slots());
        const Binder cross_small(small, 1, loaded->kernel().slots());
        // Every load issues new slots, so the fresh model needs tables bound to its own slots.
        const Binder fresh_cross(f, 1, fresh->kernel().slots());
        const Binder fresh_small(small, 1, fresh->kernel().slots());
        auto expected_large = fresh->predict(fresh_cross.source());
        auto expected_small = fresh->predict(fresh_small.source());
        ASSERT_TRUE(expected_large.has_value()) << expected_large.error().message();
        ASSERT_TRUE(expected_small.has_value()) << expected_small.error().message();
        for (int round = 0; round < 2; ++round)
        {
            auto a = copy.predict(cross_small.source());
            auto b = loaded->predict(cross.source());
            auto c = copy.predict(cross.source());
            auto d = loaded->predict(cross_small.source());
            ASSERT_TRUE(a.has_value() && b.has_value() && c.has_value() && d.has_value());
            EXPECT_TRUE(identical(a->mean, expected_small->mean));
            EXPECT_TRUE(identical(a->variance, expected_small->variance));
            EXPECT_TRUE(identical(b->mean, expected_large->mean));
            EXPECT_TRUE(identical(c->variance, expected_large->variance));
            EXPECT_TRUE(identical(d->mean, expected_small->mean));
        }
    }

    /// @brief Overwrites, in place and without truncating, one double of a named tensor of model.safetensors.
    /// @details Unlike patch_tensor(), the file keeps its inode and size, so a live read-only mapping observes the
    /// change (Linux shows file writes through unmodified MAP_PRIVATE pages).
    /// @param[in] dir Checkpoint directory.
    /// @param[in] tensor Tensor name.
    /// @param[in] index Element index inside the tensor.
    /// @param[in] value New value.
    /// @return True when the tensor was found and written.
    bool write_tensor_in_place(
        const std::filesystem::path& dir, const std::string& tensor, std::size_t index, double value
    )
    {
        const std::string bytes = slurp(dir / "model.safetensors");
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
        const auto at = static_cast<std::streamoff>(8 + header_len + begin + (index * sizeof(double)));
        std::fstream io(dir / "model.safetensors", std::ios::binary | std::ios::in | std::ios::out);
        io.seekp(at);
        io.write(reinterpret_cast<const char*>(&value), sizeof(value));
        return static_cast<bool>(io);
    }

    /// @brief The double factor of a save_with_factor checkpoint is memory-mapped; into_fitted copies it.
    /// @details has_mmap_factor() is true only for a stored double factor. Writing a NaN onto the stored diagonal
    /// after load (a breach of the mapping contract) must reach the loaded model, proving it reads the mapping, and
    /// must not reach a model extracted with into_fitted() before the write, proving the extraction copied.
    TEST(DistancePersistTest, DoubleFactorIsMemoryMappedAndIntoFittedCopiesIt)
    {
        // Spec: FMT-05, FMT-07
        const Fixture f = make_fixture(18, 4);
        auto model = fit_model<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(f);
        auto single = fit_model<cxxgpr::SinglePrecision, cxxgpr::DistanceOnly>(f);
        ASSERT_TRUE(model.has_value() && single.has_value());
        const TempDir with_factor("mmap_with_factor");
        const TempDir without_factor("mmap_without_factor");
        const TempDir single_dir("mmap_single");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, with_factor.path).has_value());
        ASSERT_TRUE(cxxgpr::persist::save(*model, without_factor.path).has_value());
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*single, single_dir.path).has_value());
        auto mapped = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(with_factor.path);
        auto refactored = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(without_factor.path);
        auto single_loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(single_dir.path);
        if (!mapped.has_value() || !refactored.has_value() || !single_loaded.has_value())
        {
            FAIL() << "load failed";
        }
        EXPECT_TRUE(mapped->has_mmap_factor());
        EXPECT_FALSE(refactored->has_mmap_factor());
        EXPECT_FALSE(single_loaded->has_mmap_factor());

        auto typed = mapped->into_fitted<cxxgpr::optimizer::Lbfgs, cxxgpr::DoublePrecision>();
        ASSERT_TRUE(typed.has_value());
        const Binder cross(f, 1, mapped->kernel().slots());
        const Binder typed_cross(f, 1, typed->kernel().slots());
        auto before = mapped->predict(cross.source());
        ASSERT_TRUE(before.has_value());
#ifdef __linux__
        ASSERT_TRUE(write_tensor_in_place(with_factor.path, "l", 0, std::numeric_limits<double>::quiet_NaN()));
        auto after = mapped->predict(cross.source());
        EXPECT_FALSE(after.has_value()) << "the loaded model does not read the mapped factor";
        auto extracted = typed->predict(typed_cross.source());
        ASSERT_TRUE(extracted.has_value());
        EXPECT_TRUE(identical(extracted->mean, before->mean));
        EXPECT_TRUE(identical(extracted->variance, before->variance));
#endif
    }

    /// @brief save refuses a distance model whose config.json would be nested deeper than load accepts.
    /// @details The kernel tree bound counts nodes, but each composite node adds two JSON levels, so a tree that the
    /// node bound accepts can still produce a document deeper than 128 levels. The deepest chain that saves must
    /// load, and the next one must fail at save with Config and leave an existing directory byte for byte as it was.
    TEST(DistancePersistTest, SaveRejectsAConfigNestedDeeperThanLoadAccepts)
    {
        // Spec: FMT-02, FMT-07, FMT-08
        const Fixture f = make_fixture(4, 2);
        const cxxgpr::ScalarDistance slot;
        const auto white = cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::WhiteKernel::create(0.01).value());
        auto chain = [&](std::size_t m)
        {
            auto k = cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(slot.rbf(cxxgpr::RbfKernel::create(1.0).value()));
            for (std::size_t i = 0; i < m; ++i)
            {
                k = k + white;
            }
            const Binder train(f, 0, k.slots());
            return cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed>(k).factor(train.source(), f.y);
        };
        const TempDir parent("distance_nesting_limit");
        const auto good_dir = parent.path / "good";
        // Below the root object a composite adds two levels and the supplied leaf
        // {"distance": {"leaf": {"rbf": {"length_scale": {"value": ...}}}}} adds five: depth 2m + 6.
        constexpr std::size_t deepest = 61;
        auto deep = chain(deepest);
        ASSERT_TRUE(deep.has_value());
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*deep, good_dir).has_value());
        EXPECT_TRUE(cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(good_dir).has_value());
        const std::string config_before = slurp(good_dir / "config.json");
        const std::string tensors_before = slurp(good_dir / "model.safetensors");

        auto too_deep = chain(deepest + 1U);
        ASSERT_TRUE(too_deep.has_value());
        for (const bool factor : {false, true})
        {
            auto saved = factor ? cxxgpr::persist::save_with_factor(*too_deep, good_dir)
                                : cxxgpr::persist::save(*too_deep, good_dir);
            ASSERT_FALSE(saved.has_value());
            EXPECT_EQ(saved.error().persist_error_kind(), cxxgpr::PersistErrorKind::Config);
            auto fresh = cxxgpr::persist::save(*too_deep, parent.path / "fresh");
            ASSERT_FALSE(fresh.has_value());
            EXPECT_FALSE(std::filesystem::exists(parent.path / "fresh"));
        }
        EXPECT_EQ(slurp(good_dir / "config.json"), config_before);
        EXPECT_EQ(slurp(good_dir / "model.safetensors"), tensors_before);
    }

    /// @brief Property: save and load of the deepest accepted supplied-distance model, and save of the shallowest
    /// refused one, run on a thread with a small stack.
    /// @details The config.json codec and the rebuild of the distance kernel on load recurse once per composite. The
    /// models are fitted on the main thread; every save, load, and destruction of the loaded model runs on the small
    /// stack. The chains are the ones of SaveRejectsAConfigNestedDeeperThanLoadAccepts.
    TEST(DistancePersistTest, SaveAndLoadOfTheDeepestAcceptedModelFitInASmallStack)
    {
        // Spec: FMT-02, FMT-07, FMT-08
        const Fixture f = make_fixture(4, 2);
        const cxxgpr::ScalarDistance slot;
        const auto white = cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::WhiteKernel::create(0.01).value());
        auto chain = [&](std::size_t m)
        {
            auto k = cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(slot.rbf(cxxgpr::RbfKernel::create(1.0).value()));
            for (std::size_t i = 0; i < m; ++i)
            {
                k = k + white;
            }
            const Binder train(f, 0, k.slots());
            return cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed>(k).factor(train.source(), f.y);
        };
        const TempDir parent("distance_small_stack");
        // Depth 2m + 6: 61 composites are accepted and 62 are refused.
        const auto deep = chain(61U);
        const auto too_deep = chain(62U);
        ASSERT_TRUE(deep.has_value());
        ASSERT_TRUE(too_deep.has_value());

        // Outcomes are recorded on the small stack and checked on the main thread.
        bool saved = false;
        bool loaded = false;
        std::optional<cxxgpr::GprError> refused;
        const bool ran = cxxgpr::test_support::run_with_stack_size(
            cxxgpr::test_support::deep_model_stack_bytes(), [&]
        {
            const auto dir = parent.path / "accepted";
            saved = cxxgpr::persist::save_with_factor(*deep, dir).has_value();
            loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir).has_value();
            auto result = cxxgpr::persist::save_with_factor(*too_deep, parent.path / "refused");
            if (!result.has_value())
            {
                refused = std::move(result).error();
            }
        }
        );
        ASSERT_TRUE(ran);
        EXPECT_TRUE(saved);
        EXPECT_TRUE(loaded);
        if (!refused.has_value())
        {
            FAIL() << "save accepted a model nested deeper than load accepts";
        }
        EXPECT_EQ(refused->code(), cxxgpr::GprErrorCode::PersistFailed);
        EXPECT_EQ(refused->persist_error_kind(), cxxgpr::PersistErrorKind::Config);
    }

    /// @brief Slot numbers out of depth-first first-appearance order, and coordinate transforms on a distance-only
    /// model, are rejected with Config.
    /// @details The loader binds d2.<k> to the k-th slot of the rebuilt kernel; a permuted numbering would pair each
    /// table with another leaf without any other error. A distance-only model has no coordinates, so both of its
    /// coordinate transforms must be the identity.
    TEST(DistancePersistTest, PermutedSlotsAndCoordinateTransformsAreRejected)
    {
        // Spec: FMT-07, FMT-08
        const Fixture f = make_fixture(14, 3);
        const cxxgpr::ScalarDistance s0;
        const cxxgpr::ScalarDistance s1;
        // Two scalar slots with different length scales, so that swapping the tables would change the model.
        const auto kernel = cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(s0.rbf(cxxgpr::RbfKernel::create(0.6).value()))
            + cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(s1.rbf(cxxgpr::RbfKernel::create(2.0).value()));
        const Binder train(f, 0, kernel.slots());
        auto model = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed>(kernel).with_noise_variance(0.05).value().factor(
            train.source(), f.y
        );
        ASSERT_TRUE(model.has_value()) << model.error().message();

        const TempDir dir("permuted_slots");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, dir.path).has_value());
        ASSERT_TRUE(cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path).has_value());
        ASSERT_TRUE(edit_config(dir.path, R"("slot": 0)", R"("slot": 9)"));
        ASSERT_TRUE(edit_config(dir.path, R"("slot": 1)", R"("slot": 0)"));
        ASSERT_TRUE(edit_config(dir.path, R"("slot": 9)", R"("slot": 1)"));
        auto permuted = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path);
        ASSERT_FALSE(permuted.has_value());
        EXPECT_EQ(permuted.error().persist_error_kind(), cxxgpr::PersistErrorKind::Config);

        auto expect_config = [&](const std::string& name, const std::string& from, const std::string& to)
        {
            const TempDir bad("transform_" + name);
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, bad.path).has_value());
            ASSERT_TRUE(edit_config(bad.path, from, to)) << name;
            auto loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(bad.path);
            ASSERT_FALSE(loaded.has_value()) << name;
            EXPECT_EQ(loaded.error().persist_error_kind(), cxxgpr::PersistErrorKind::Config) << name;
        };
        expect_config("x_unfitted", R"("x_unfitted": "identity")", R"("x_unfitted": {"standardize": {}})");
        expect_config("x_transform", R"("x_transform": "identity")", R"("x_transform": {"pipeline": {"steps": []}})");
    }

    /// @brief Optional string fields of the wrong JSON type are a Config error on the distance loader too.
    TEST(DistancePersistTest, OptionalStringFieldsOfTheWrongTypeAreRejectedWithConfig)
    {
        // Spec: FMT-03, FMT-07, FMT-08
        const Fixture f = make_fixture(10, 2);
        auto model = fit_model<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>, cxxgpr::DistanceOnly>(f);
        ASSERT_TRUE(model.has_value());

        struct Case
        {
            /// @brief Text present in the saved config.json.
            std::string from;
            /// @brief Replacement giving one field a non-string value.
            std::string to;
        };

        const std::array<Case, 5> cases{
            Case{         .from = R"("precision": "mixed")",               .to = R"("precision": 1)"},
            Case{.from = R"("residual": "promote_storage")",             .to = R"("residual": null)"},
            Case{           .from = R"("math": "accurate")",                 .to = R"("math": true)"},
            Case{   .from = R"("distance_cache": "always")", .to = R"("distance_cache": ["always"])"},
            Case{    .from = R"("points": "distance_only")",                  .to = R"("points": 0)"},
        };
        for (const auto& c : cases)
        {
            SCOPED_TRACE(c.to);
            const TempDir dir("optional_type");
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, dir.path).has_value());
            ASSERT_TRUE(edit_config(dir.path, c.from, c.to));
            auto loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path);
            ASSERT_FALSE(loaded.has_value());
            EXPECT_EQ(loaded.error().persist_error_kind(), cxxgpr::PersistErrorKind::Config);
        }
    }

    /// @brief Fitted target transform that changes nothing and persists a caller-supplied identifier and state.
    class TextFittedTarget final : public cxxgpr::transform::FittedTargetTransform
    {
    public:
        /// @brief Constructs the transform.
        /// @param[in] id Persistence identifier.
        /// @param[in] state State text.
        TextFittedTarget(std::string id, std::string state) : id_(std::move(id)), state_(std::move(state)) {}

        /// @brief Leaves the targets unchanged.
        /// @param[in,out] y Targets.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> transform(
            [[maybe_unused]] std::span<double> y
        ) const override
        {
            return {};
        }

        /// @brief Leaves the means unchanged.
        /// @param[in,out] mean Means.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> inverse_transform_mean(
            [[maybe_unused]] std::span<double> mean
        ) const override
        {
            return {};
        }

        /// @brief Leaves the variances unchanged.
        /// @param[in,out] var Variances.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> inverse_transform_variance(
            [[maybe_unused]] std::span<double> var
        ) const override
        {
            return {};
        }

        /// @brief Clones this transform.
        /// @return Owning copy.
        [[nodiscard]] std::unique_ptr<cxxgpr::transform::FittedTargetTransform> clone() const override
        {
            return std::make_unique<TextFittedTarget>(id_, state_);
        }

        /// @brief Returns the identifier.
        /// @return Identifier.
        [[nodiscard]] std::string_view persist_id() const override { return id_; }

        /// @brief Returns the state text.
        /// @return State text.
        [[nodiscard]] std::expected<std::string, cxxgpr::GprError> persist_state_json() const override
        {
            return state_;
        }

    private:
        /// @brief Persistence identifier.
        std::string id_;
        /// @brief State text.
        std::string state_;
    };

    /// @brief Unfitted target transform persisting its own text and fitting to a TextFittedTarget.
    class TextUnfittedTarget final : public cxxgpr::transform::UnfittedTargetTransform
    {
    public:
        /// @brief Constructs the transform.
        /// @param[in] id Persistence identifier.
        /// @param[in] state State text.
        /// @param[in] fitted Fitted transform returned by fit().
        TextUnfittedTarget(std::string id, std::string state, TextFittedTarget fitted) :
            id_(std::move(id)),
            state_(std::move(state)),
            fitted_(std::move(fitted))
        {
        }

        /// @brief Returns a copy of the fitted transform.
        /// @param[in] y Targets, ignored.
        /// @return Owning fitted transform.
        [[nodiscard]] std::expected<std::unique_ptr<cxxgpr::transform::FittedTargetTransform>, cxxgpr::GprError> fit(
            [[maybe_unused]] std::span<const double> y
        ) const override
        {
            return fitted_.clone();
        }

        /// @brief Clones this transform.
        /// @return Owning copy.
        [[nodiscard]] std::unique_ptr<cxxgpr::transform::UnfittedTargetTransform> clone() const override
        {
            return std::make_unique<TextUnfittedTarget>(id_, state_, fitted_);
        }

        /// @brief Returns the identifier.
        /// @return Identifier.
        [[nodiscard]] std::string_view persist_id() const override { return id_; }

        /// @brief Returns the state text.
        /// @return State text.
        [[nodiscard]] std::expected<std::string, cxxgpr::GprError> persist_state_json() const override
        {
            return state_;
        }

    private:
        /// @brief Persistence identifier.
        std::string id_;
        /// @brief State text.
        std::string state_;
        /// @brief Result of fit().
        TextFittedTarget fitted_;
    };

    /// @brief Factors a distance-only model whose target transforms persist the given identifiers and states.
    /// @param[in] f Fixture.
    /// @param[in] ids Identifiers of the unfitted and the fitted target transform.
    /// @param[in] states States of the unfitted and the fitted target transform.
    /// @return Factored model.
    auto factor_custom_target(
        const Fixture& f, const std::array<std::string, 2>& ids, const std::array<std::string, 2>& states
    )
    {
        const auto kernel = only_kernel();
        const Binder train(f, 0, kernel.slots());
        return cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed>(kernel)
            .with_target_transform(TextUnfittedTarget(ids[0], states[0], TextFittedTarget(ids[1], states[1])))
            .factor(train.source(), f.y);
    }

    /// @brief Custom state is stored verbatim as a string, and text that config.json cannot hold is refused at save.
    /// @details Through the distance serializer and loader: a state that is a JSON string literal comes back with
    /// its quotes; a state or identifier that is not valid UTF-8 fails at save as a value (no exception) and leaves
    /// an existing checkpoint untouched; a state nested deeper than the JSON bound is refused with Config.
    TEST(DistancePersistTest, CustomStateIsVerbatimAndUnstorableTextIsRefused)
    {
        // Spec: FMT-02, FMT-06, FMT-07, FMT-08
        const Fixture f = make_fixture(8, 2);
        const std::array<std::string, 2> ids{"user.text.y_unfitted", "user.text.y_fitted"};
        std::array<std::string, 2> received{"unset", "unset"};
        cxxgpr::persist::PersistRegistry reg;
        ASSERT_TRUE(
            reg.register_unfitted_target(
                   ids[0],
                   [&received, ids](std::string_view state)
                       -> std::expected<std::unique_ptr<cxxgpr::transform::UnfittedTargetTransform>, cxxgpr::GprError>
        {
            received[0] = std::string(state);
            return std::make_unique<TextUnfittedTarget>(ids[0], std::string(state), TextFittedTarget(ids[1], ""));
        }
            ).has_value()
        );
        ASSERT_TRUE(
            reg.register_fitted_target(
                   ids[1],
                   [&received, ids](std::string_view state)
                       -> std::expected<std::unique_ptr<cxxgpr::transform::FittedTargetTransform>, cxxgpr::GprError>
        {
            received[1] = std::string(state);
            return std::make_unique<TextFittedTarget>(ids[1], std::string(state));
        }
            ).has_value()
        );

        // A JSON string literal and text that is not JSON both come back exactly.
        for (const std::array<std::string, 2>& sent : {
                 std::array<std::string, 2>{ "\"abc\"", "\"def\""},
                  std::array<std::string, 2>{"raw text",   "{open"}
        })
        {
            auto model = factor_custom_target(f, ids, sent);
            ASSERT_TRUE(model.has_value());
            const TempDir dir("custom_verbatim");
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, dir.path, reg).has_value());
            ASSERT_TRUE(cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path, reg).has_value());
            EXPECT_EQ(received[0], sent[0]);
            EXPECT_EQ(received[1], sent[1]);
        }

        // Text that cannot be stored fails at save and leaves the existing checkpoint as it was.
        const TempDir dir("custom_refused");
        auto good = factor_custom_target(f, ids, {"1", "2"});
        ASSERT_TRUE(good.has_value());
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*good, dir.path, reg).has_value());
        const std::string config_before = slurp(dir.path / "config.json");
        const std::string tensors_before = slurp(dir.path / "model.safetensors");

        struct Bad
        {
            /// @brief Identifiers to use.
            std::array<std::string, 2> ids;
            /// @brief States to use.
            std::array<std::string, 2> states;
            /// @brief Expected error kind.
            cxxgpr::PersistErrorKind kind;
        };

        const std::string deep = std::string(200, '[') + std::string(200, ']');
        const std::array<Bad, 4> bads{
            Bad{                  .ids = ids, .states = {"\xff\xfe", "2"},   .kind = cxxgpr::PersistErrorKind::NotPersistable},
            Bad{                  .ids = ids, .states = {"1", "\"\xff\""},   .kind = cxxgpr::PersistErrorKind::NotPersistable},
            Bad{.ids = {"user.\xff", ids[1]},        .states = {"1", "2"}, .kind = cxxgpr::PersistErrorKind::InvalidPersistId},
            Bad{                  .ids = ids,       .states = {deep, "2"},           .kind = cxxgpr::PersistErrorKind::Config},
        };
        for (std::size_t i = 0; i < bads.size(); ++i)
        {
            SCOPED_TRACE("case " + std::to_string(i));
            auto model = factor_custom_target(f, bads[i].ids, bads[i].states);
            ASSERT_TRUE(model.has_value());
            for (const bool factor : {false, true})
            {
                auto saved = factor ? cxxgpr::persist::save_with_factor(*model, dir.path, reg)
                                    : cxxgpr::persist::save(*model, dir.path, reg);
                ASSERT_FALSE(saved.has_value());
                EXPECT_EQ(saved.error().persist_error_kind(), bads[i].kind);
            }
        }
        EXPECT_EQ(slurp(dir.path / "config.json"), config_before);
        EXPECT_EQ(slurp(dir.path / "model.safetensors"), tensors_before);
    }

    /// @brief A loaded mixed-precision model recomputes its objective from the stored factor and has no summary.
    /// @details The checkpoint stores the float factor and the refined double weights but not the refinement
    /// summary. Training reports the objective of the float solve (the same value as a SinglePrecision model at the
    /// same hyperparameters), while the loader recomputes it from the stored factor and the refined weights, so the
    /// two differ by (1/2) y^T (alpha_refined - alpha_float) plus the float rounding of the training objective. The
    /// test predicts the difference from a SinglePrecision fit of the same data and allows 1e-6 relative for that
    /// rounding (observed 1.2e-7 relative; the difference itself is about 4e-6 relative on this fixture).
    /// refinement_summary() is empty after load and after into_fitted(), as for the coordinate LoadedGpr.
    TEST(DistancePersistTest, LoadedMixedModelRecomputesTheObjectiveAndHasNoSummary)
    {
        // Spec: FMT-05, FMT-07
        const Fixture f = make_fixture(26, 6);
        auto single = fit_model<cxxgpr::SinglePrecision, cxxgpr::DistanceOnly>(f);
        ASSERT_TRUE(single.has_value());
        auto check = [&]<typename P>(const std::string& name)
        {
            SCOPED_TRACE(name);
            auto model = fit_model<P, cxxgpr::DistanceOnly>(f);
            ASSERT_TRUE(model.has_value());
            ASSERT_TRUE(model->refinement_summary().has_value());
            const TempDir dir(name);
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, dir.path).has_value());
            auto loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir.path);
            ASSERT_TRUE(loaded.has_value());
            auto typed = loaded->into_fitted<cxxgpr::optimizer::Lbfgs, P>();
            ASSERT_TRUE(typed.has_value());
            EXPECT_FALSE(typed->refinement_summary().has_value());
            EXPECT_TRUE(identical(
                std::vector<double>(typed->alpha().begin(), typed->alpha().end()),
                std::vector<double>(model->alpha().begin(), model->alpha().end())
            ));
            const double trained = model->neg_log_marginal_likelihood();
            EXPECT_EQ(trained, single->neg_log_marginal_likelihood());
            double predicted = 0.0;
            for (std::size_t i = 0; i < f.n; ++i)
            {
                predicted +=
                    0.5 * f.y[i] * (static_cast<double>(model->alpha()[i]) - static_cast<double>(single->alpha()[i]));
            }
            const double delta = typed->neg_log_marginal_likelihood() - trained;
            EXPECT_NE(delta, 0.0);
            EXPECT_LT(std::abs(delta - predicted), 1e-6 * std::abs(trained));
        };
        check.template operator()<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>("summary_promote");
        check.template operator()<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>("summary_reevaluate");
    }

    /// @brief Verifies FMT-09 save_id determinism, mismatch rejection, one-sided rejection, and legacy compatibility
    /// for supplied-distance checkpoints (DistanceOnly and WithPoints).
    TEST(DistancePersistTest, SaveIdDeterminismMismatchAndLegacyCompatibility)
    {
        // Spec: FMT-07, FMT-08, FMT-09
        const Fixture f_a = make_fixture(26, 6);
        const Fixture f_b = make_fixture(27, 6);
        auto m_a = fit_model<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(f_a);
        auto m_b = fit_model<cxxgpr::DoublePrecision, cxxgpr::DistanceOnly>(f_b);
        ASSERT_TRUE(m_a.has_value());
        ASSERT_TRUE(m_b.has_value());

        const TempDir dir_a1("dist_save_id_a1");
        const TempDir dir_a2("dist_save_id_a2");
        const TempDir dir_b("dist_save_id_b");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*m_a, dir_a1.path).has_value());
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*m_a, dir_a2.path).has_value());
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*m_b, dir_b.path).has_value());

        auto read_bytes = [](const std::filesystem::path& p)
        {
            std::ifstream ifs(p, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
        };
        const std::string cfg_a1 = read_bytes(dir_a1.path / "config.json");
        EXPECT_EQ(cfg_a1, read_bytes(dir_a2.path / "config.json"));
        EXPECT_NE(cfg_a1, read_bytes(dir_b.path / "config.json"));

        // Cross-save pair is rejected with PersistFailed(Tensor).
        {
            const TempDir mix("dist_save_id_mix");
            std::filesystem::copy_file(dir_a1.path / "config.json", mix.path / "config.json");
            std::filesystem::copy_file(dir_b.path / "model.safetensors", mix.path / "model.safetensors");
            const auto res = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(mix.path);
            ASSERT_FALSE(res.has_value());
            EXPECT_EQ(res.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(res.error().persist_error_kind(), cxxgpr::PersistErrorKind::Tensor);
        }

        // Legacy WithPoints pair without save_id / __metadata__ loads cleanly, while one-sided is rejected.
        auto m_wp = fit_model<cxxgpr::DoublePrecision, cxxgpr::WithPoints>(f_a);
        ASSERT_TRUE(m_wp.has_value());
        const TempDir dir_wp("dist_save_id_wp");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*m_wp, dir_wp.path).has_value());
        {
            std::string cfg = read_bytes(dir_wp.path / "config.json");
            const std::size_t kpos = cfg.find("\"save_id\":");
            ASSERT_NE(kpos, std::string::npos);
            const std::size_t lstart = cfg.rfind('\n', kpos);
            const std::size_t lend = cfg.find('\n', kpos);
            const bool trailing = (lend != std::string::npos && cfg[lend - 1U] == ',');
            cfg.erase(lstart, lend - lstart);
            if (!trailing && lstart > 0U && cfg[lstart - 1U] == ',')
            {
                cfg.erase(lstart - 1U, 1U);
            }
            std::ofstream ofs(dir_wp.path / "config.json", std::ios::binary | std::ios::trunc);
            ofs << cfg;
        }
        // Only safetensors has cxxgpr.save_id -> rejected.
        {
            const auto one_sided = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::WithPoints>::load(dir_wp.path);
            ASSERT_FALSE(one_sided.has_value());
            EXPECT_EQ(one_sided.error().persist_error_kind(), cxxgpr::PersistErrorKind::Tensor);
        }
        // Strip __metadata__ too -> legacy pair loads cleanly.
        {
            const std::string raw = read_bytes(dir_wp.path / "model.safetensors");
            std::uint64_t hlen = 0U;
            std::memcpy(&hlen, raw.data(), sizeof(hlen));
            std::string hstr = raw.substr(8U, static_cast<std::size_t>(hlen));
            while (!hstr.empty() && hstr.back() == ' ')
            {
                hstr.pop_back();
            }
            const std::size_t mpos = hstr.find("\"__metadata__\":{");
            ASSERT_NE(mpos, std::string::npos);
            const std::size_t mend = hstr.find("},", mpos);
            ASSERT_NE(mend, std::string::npos);
            hstr.erase(mpos, (mend + 2U) - mpos);
            const std::size_t padded = (hstr.size() + 7U) & ~std::size_t{7U};
            hstr.append(padded - hstr.size(), ' ');
            const auto new_hlen = static_cast<std::uint64_t>(hstr.size());
            std::ofstream ofs(dir_wp.path / "model.safetensors", std::ios::binary | std::ios::trunc);
            ofs.write(reinterpret_cast<const char*>(&new_hlen), sizeof(new_hlen));
            ofs.write(hstr.data(), static_cast<std::streamsize>(hstr.size()));
            ofs.write(raw.data() + 8U + hlen, static_cast<std::streamsize>(raw.size() - (8U + hlen)));
            ofs.flush();
            ofs.close();
            const auto leg_wp = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::WithPoints>::load(dir_wp.path);
            EXPECT_TRUE(leg_wp.has_value()) << (leg_wp.has_value() ? std::string_view{} : leg_wp.error().message());
        }
    }

    /// @brief Verifies saving a FittedDistanceGpr built with with_retain_distances(false) fails with
    /// PersistFailed(NotPersistable) before creating any output directory.
    TEST(DistancePersistTest, NonRetainedModelRefusesPersistBeforeCreatingDirectory)
    {
        // Spec: FMT-07, DES-21
        const Fixture f = make_fixture(10, 3);
        const TempDir parent("non_retained_guard");

        // DistanceOnly: both save() and save_with_factor() refuse and leave destination absent.
        {
            const auto kernel = only_kernel();
            const Binder train(f, 0, kernel.slots());
            const auto model = cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed>(kernel)
                                   .with_noise_variance(0.05)
                                   .value()
                                   .with_retain_distances(false)
                                   .factor(train.source(), f.y);
            ASSERT_TRUE(model.has_value());
            EXPECT_FALSE(model->retains_distances());

            const auto dir_save = parent.path / "dist_only_save_missing";
            const auto err_save = cxxgpr::persist::save(*model, dir_save);
            ASSERT_FALSE(err_save.has_value());
            EXPECT_EQ(err_save.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(err_save.error().persist_error_kind(), cxxgpr::PersistErrorKind::NotPersistable);
            EXPECT_FALSE(std::filesystem::exists(dir_save));

            const auto dir_factor = parent.path / "dist_only_factor_missing";
            const auto err_factor = cxxgpr::persist::save_with_factor(*model, dir_factor);
            ASSERT_FALSE(err_factor.has_value());
            EXPECT_EQ(err_factor.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(err_factor.error().persist_error_kind(), cxxgpr::PersistErrorKind::NotPersistable);
            EXPECT_FALSE(std::filesystem::exists(dir_factor));
        }

        // WithPoints: both save() and save_with_factor() refuse and leave destination absent.
        {
            const auto kernel = points_kernel();
            const Binder train(f, 0, kernel.slots());
            const auto model =
                cxxgpr::DistanceGpr<cxxgpr::optimizer::Fixed, cxxgpr::DoublePrecision, cxxgpr::WithPoints>(kernel)
                    .with_noise_variance(0.05)
                    .value()
                    .with_retain_distances(false)
                    .factor(train.source(), cxxgpr::MatrixRef<double>(f.x.data(), f.n, f.d), f.y);
            ASSERT_TRUE(model.has_value());
            EXPECT_FALSE(model->retains_distances());

            const auto dir_save = parent.path / "wp_save_missing";
            const auto err_save = cxxgpr::persist::save(*model, dir_save);
            ASSERT_FALSE(err_save.has_value());
            EXPECT_EQ(err_save.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(err_save.error().persist_error_kind(), cxxgpr::PersistErrorKind::NotPersistable);
            EXPECT_FALSE(std::filesystem::exists(dir_save));

            const auto dir_factor = parent.path / "wp_factor_missing";
            const auto err_factor = cxxgpr::persist::save_with_factor(*model, dir_factor);
            ASSERT_FALSE(err_factor.has_value());
            EXPECT_EQ(err_factor.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(err_factor.error().persist_error_kind(), cxxgpr::PersistErrorKind::NotPersistable);
            EXPECT_FALSE(std::filesystem::exists(dir_factor));
        }
    }

    /// @brief Verifies LoadedDistanceGpr<WithPoints> float query overloads match widened doubles and scalar packed
    /// stores roundtrip identically.
    TEST(DistancePersistTest, LoadedDistanceWithPointsFloatQueriesAndPackedRoundtrip)
    {
        // Spec: FMT-07, FMT-08, DES-16
        const Fixture f = make_fixture(12, 4);

        // LoadedDistanceGpr<WithPoints> float query overloads match double-widened queries bit-for-bit.
        {
            const auto model = fit_model<cxxgpr::DoublePrecision, cxxgpr::WithPoints>(f);
            ASSERT_TRUE(model.has_value());
            const TempDir dir("wp_float_queries");
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, dir.path).has_value());
            const auto loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::WithPoints>::load(dir.path);
            ASSERT_TRUE(loaded.has_value());

            const Binder cross(f, 1, loaded->kernel().slots());
            const Binder query(f, 2, loaded->kernel().slots());
            std::vector<float> q_f(f.q.size());
            for (std::size_t idx = 0; idx < f.q.size(); ++idx)
            {
                q_f[idx] = static_cast<float>(f.q[idx]);
            }
            const std::vector<double> q_w(q_f.begin(), q_f.end());
            const cxxgpr::MatrixRef<float> q_ref_f(q_f.data(), f.m, f.d);
            const cxxgpr::MatrixRef<double> q_ref_w(q_w.data(), f.m, f.d);
            const cxxgpr::PredictOptions opts{cxxgpr::VarianceKind::Latent};

            const auto p_w = loaded->predict(cross.source(), q_ref_w, opts);
            const auto p_f = loaded->predict(cross.source(), q_ref_f, opts);
            ASSERT_TRUE(p_w.has_value());
            ASSERT_TRUE(p_f.has_value());
            EXPECT_TRUE(identical(p_w->mean, p_f->mean));
            EXPECT_TRUE(identical(p_w->variance, p_f->variance));

            const auto cov_w = loaded->predict_covariance(cross.source(), query.source(), q_ref_w, opts);
            const auto cov_f = loaded->predict_covariance(cross.source(), query.source(), q_ref_f, opts);
            ASSERT_TRUE(cov_w.has_value());
            ASSERT_TRUE(cov_f.has_value());
            EXPECT_TRUE(identical(cov_w->covariance, cov_f->covariance));

            const auto s_w = loaded->sample(cross.source(), query.source(), q_ref_w, 3U, 91U, opts);
            const auto s_f = loaded->sample(cross.source(), query.source(), q_ref_f, 3U, 91U, opts);
            ASSERT_TRUE(s_w.has_value());
            ASSERT_TRUE(s_f.has_value());
            EXPECT_TRUE(identical(*s_w, *s_f));
        }

        // Packed built-in scalar store roundtrip via save() and save_with_factor().
        {
            const auto check_packed = [&]<cxxgpr::PrecisionPolicyConcept P>(const std::string& tag)
            {
                const auto model = fit_model<P, cxxgpr::DistanceOnly>(f);
                ASSERT_TRUE(model.has_value()) << tag;
                const TempDir dir_no_factor("packed_roundtrip_nf_" + tag);
                const TempDir dir_with_factor("packed_roundtrip_wf_" + tag);
                ASSERT_TRUE(cxxgpr::persist::save(*model, dir_no_factor.path).has_value());
                ASSERT_TRUE(cxxgpr::persist::save_with_factor(*model, dir_with_factor.path).has_value());

                for (const bool with_factor : {false, true})
                {
                    const auto& dir_p = with_factor ? dir_with_factor.path : dir_no_factor.path;
                    const auto loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir_p);
                    ASSERT_TRUE(loaded.has_value()) << tag;
                    const Binder cross_m(f, 1, model->kernel().slots());
                    const Binder cross_l(f, 1, loaded->kernel().slots());
                    const Binder query_m(f, 2, model->kernel().slots());
                    const Binder query_l(f, 2, loaded->kernel().slots());

                    const auto pm = model->predict(cross_m.source());
                    const auto pl = loaded->predict(cross_l.source());
                    ASSERT_TRUE(pm.has_value());
                    ASSERT_TRUE(pl.has_value());
                    const auto cm = model->predict_covariance(cross_m.source(), query_m.source());
                    const auto cl = loaded->predict_covariance(cross_l.source(), query_l.source());
                    ASSERT_TRUE(cm.has_value());
                    ASSERT_TRUE(cl.has_value());
                    const auto lm = model->loo_predict();
                    const auto ll = loaded->loo_predict();
                    ASSERT_TRUE(lm.has_value());
                    ASSERT_TRUE(ll.has_value());

                    if (with_factor)
                    {
                        EXPECT_TRUE(identical(pm->mean, pl->mean));
                        EXPECT_TRUE(identical(pm->variance, pl->variance));
                        EXPECT_TRUE(identical(cm->covariance, cl->covariance));
                        EXPECT_TRUE(identical(lm->mean, ll->mean));
                        EXPECT_TRUE(identical(lm->variance, ll->variance));
                    }
                    else
                    {
                        const double tol = std::same_as<P, cxxgpr::DoublePrecision> ? 1e-10 : 1e-5;
                        EXPECT_LT(rel_diff(pm->mean, pl->mean), tol);
                        EXPECT_LT(rel_diff(pm->variance, pl->variance), tol);
                        EXPECT_LT(rel_diff(cm->covariance, cl->covariance), tol);
                        EXPECT_LT(rel_diff(lm->mean, ll->mean), tol);
                        EXPECT_LT(rel_diff(lm->variance, ll->variance), tol);
                    }
                }
            };
            check_packed.template operator()<cxxgpr::DoublePrecision>("f64");
            check_packed.template operator()<cxxgpr::SinglePrecision>("f32");
            check_packed.template operator()<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>("mixed");
        }
    }

}  // namespace
