// SPDX-License-Identifier: Apache-2.0
// Spec: FMT-01, FMT-02, FMT-03, FMT-04, FMT-05, FMT-06, FMT-08
/// @file persist_test.cpp
/// @brief Unit and integration tests for GPR model persistence, mmap loading, and registry hooks.
/// @details Verifies save_with_factor() mmap roundtrip and bit-identical predictions/LOO, save()
/// refactor-on-load parity, into_fitted() precision checks, MixedPrecision<ReevaluateKernel> and
/// CustomKernel PersistRegistry roundtrips, and negative validation against corrupt or unsupported
/// config.json and model.safetensors artifacts.

#include "cxxgpr/persist/persist.hpp"

#include "cxxgpr/gpr/gpr.hpp"
#include "cxxgpr/kernel/kernel.hpp"
#include "cxxgpr/transform/transform.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

#include "allocation_audit.hpp"

namespace
{

    /// @brief RAII temporary directory guard creating an isolated per-test directory under temp_directory_path.
    /// @details Removes any leftover directory on construction and cleans up recursively on destruction
    /// so parallel CTest runs never collide on persistence artifacts.
    struct TempDirGuard
    {
        /// @brief Filesystem path of the managed temporary directory.
        std::filesystem::path path;

        /// @brief Creates a fresh temporary directory named with prefix `cxxgpr_test_`.
        /// @param[in] name Unique test identifier suffix appended to the directory name.
        explicit TempDirGuard(const std::string& name)
        {
            path = std::filesystem::temp_directory_path() / ("cxxgpr_test_" + name);
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
            std::filesystem::create_directories(path, ec);
        }

        /// @brief Recursively deletes the temporary directory while suppressing cleanup exceptions.
        ~TempDirGuard()
        {
            std::error_code ec;
            try
            {
                std::filesystem::remove_all(path, ec);
            }
            catch (...)
            {
                ec.clear();
            }
        }
    };

    /// @brief Double-precision custom scaled exponential kernel implementing state JSON persistence.
    /// @details Evaluates k(d2) = scale_ * exp(-d2) and emits serialized scale_ state via
    /// persist_state_json() under persist ID `"user.custom_scale"`.
    class CustomScaleKernel final : public cxxgpr::KernelTerm<double>
    {
    public:
        /// @brief Constructs the double-precision custom kernel with initial scale factor.
        /// @param[in] scale Positive kernel output scale parameter.
        explicit CustomScaleKernel(double scale) : scale_(scale) {}

        /// @brief Clones this custom kernel preserving its scale_ value.
        /// @return Owning unique_ptr copy of this CustomScaleKernel.
        [[nodiscard]] std::unique_ptr<cxxgpr::KernelTerm<double>> clone() const override
        {
            return std::make_unique<CustomScaleKernel>(scale_);
        }

        /// @brief Returns the number of trainable hyperparameters (1 for scale_).
        /// @return Hyperparameter count (always 1).
        [[nodiscard]] std::size_t num_params() const noexcept override { return 1U; }

        /// @brief Writes log(scale_) into out[0].
        /// @param[out] out Destination span of length 1 receiving log-transformed scale.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> get_params(std::span<double> out) const override
        {
            out[0] = std::log(scale_);
            return {};
        }

        /// @brief Updates scale_ from log-parameter params[0].
        /// @param[in] params Input span containing log(scale) at index 0.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> set_params(std::span<const double> params) override
        {
            scale_ = std::exp(params[0]);
            return {};
        }

        /// @brief Populates default positive bounds for the scale parameter.
        /// @param[out] out Destination Interval span of length 1.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> bounds_into(std::span<cxxgpr::Interval> out) const override
        {
            out[0] = cxxgpr::Interval::default_positive();
            return {};
        }

        /// @brief Evaluates scale_ * exp(-d2) over the requested square matrix triangle.
        /// @param[in] dist Pairwise squared Euclidean distance matrix.
        /// @param[out] out Output kernel Gram matrix buffer.
        /// @param[in] uplo Triangle mask controlling which entries are written.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply(
            cxxgpr::MatrixRef<double> dist, cxxgpr::MatrixMut<double> out, cxxgpr::Triangle uplo
        ) const override
        {
            const std::size_t n = dist.rows();
            for (std::size_t col = 0; col < n; ++col)
            {
                const std::size_t rb = (uplo == cxxgpr::Triangle::Lower) ? col : 0U;
                const std::size_t re = (uplo == cxxgpr::Triangle::Upper) ? (col + 1U) : n;
                for (std::size_t row = rb; row < re; ++row)
                {
                    out(row, col) = scale_ * std::exp(-dist(row, col));
                }
            }
            return {};
        }

        /// @brief Evaluates rectangular cross-covariance scale_ * exp(-d2).
        /// @param[in] dist Rectangular pairwise squared Euclidean distance matrix.
        /// @param[out] out Output cross-covariance matrix buffer.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply_cross(
            cxxgpr::MatrixRef<double> dist, cxxgpr::MatrixMut<double> out
        ) const override
        {
            for (std::size_t col = 0; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0; row < dist.rows(); ++row)
                {
                    out(row, col) = scale_ * std::exp(-dist(row, col));
                }
            }
            return {};
        }

        /// @brief Fills the prior kernel diagonal with scale_.
        /// @param[out] out Destination diagonal span of length n.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> fill_diag(std::span<double> out) const override
        {
            std::ranges::fill(out, scale_);
            return {};
        }

        /// @brief Computes derivative dk / d(log scale), which equals k itself.
        /// @param[in] dist Pairwise squared Euclidean distance matrix.
        /// @param[out] d_k Output gradient matrix buffer.
        /// @param[in] param_idx Local parameter index (unused because num_params() == 1).
        /// @param[in] uplo Triangle mask controlling which entries are written.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> grad(
            cxxgpr::MatrixRef<double> dist,
            cxxgpr::MatrixMut<double> d_k,
            [[maybe_unused]] std::size_t param_idx,
            cxxgpr::Triangle uplo
        ) const override
        {
            return apply(dist, d_k, uplo);
        }

        /// @brief Computes second derivative d^2 k / d(log scale)^2, which equals k itself.
        /// @param[in] dist Pairwise squared Euclidean distance matrix.
        /// @param[out] d2_k Output Hessian matrix buffer.
        /// @param[in] i First local parameter index (unused because num_params() == 1).
        /// @param[in] j Second local parameter index (unused because num_params() == 1).
        /// @param[in] uplo Triangle mask controlling which entries are written.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> hess(
            cxxgpr::MatrixRef<double> dist,
            cxxgpr::MatrixMut<double> d2_k,
            [[maybe_unused]] std::size_t i,
            [[maybe_unused]] std::size_t j,
            cxxgpr::Triangle uplo
        ) const override
        {
            return apply(dist, d2_k, uplo);
        }

        /// @brief Returns the persistence identifier matching CustomScaleKernelFloat and PersistRegistry.
        /// @return Static string view `"user.custom_scale"`.
        [[nodiscard]] std::string_view persist_id() const noexcept override { return "user.custom_scale"; }

        /// @brief Serializes the current scale_ parameter as decimal string state for persistence.
        /// @return Decimal string representation of scale_.
        [[nodiscard]] std::expected<std::string, cxxgpr::GprError> persist_state_json() const override
        {
            return std::to_string(scale_);
        }

    private:
        /// @brief Positive output scale factor in linear space.
        double scale_;
    };

    /// @brief Single-precision companion of CustomScaleKernel required by CustomKernel::create().
    /// @details Evaluates float-precision k(d2) = scale_ * exp(-d2) and shares persist ID
    /// `"user.custom_scale"` with CustomScaleKernel.
    class CustomScaleKernelFloat final : public cxxgpr::KernelTerm<float>
    {
    public:
        /// @brief Constructs the single-precision custom kernel with initial scale factor.
        /// @param[in] scale Positive kernel output scale parameter.
        explicit CustomScaleKernelFloat(float scale) : scale_(scale) {}

        /// @brief Clones this float custom kernel preserving its scale_ value.
        /// @return Owning unique_ptr copy of this CustomScaleKernelFloat.
        [[nodiscard]] std::unique_ptr<cxxgpr::KernelTerm<float>> clone() const override
        {
            return std::make_unique<CustomScaleKernelFloat>(scale_);
        }

        /// @brief Returns the number of trainable hyperparameters (1 for scale_).
        /// @return Hyperparameter count (always 1).
        [[nodiscard]] std::size_t num_params() const noexcept override { return 1U; }

        /// @brief Writes log(scale_) cast to double into out[0].
        /// @param[out] out Destination span of length 1 receiving log-transformed scale.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> get_params(std::span<double> out) const override
        {
            out[0] = std::log(static_cast<double>(scale_));
            return {};
        }

        /// @brief Updates float scale_ from log-parameter params[0].
        /// @param[in] params Input span containing log(scale) at index 0.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> set_params(std::span<const double> params) override
        {
            scale_ = static_cast<float>(std::exp(params[0]));
            return {};
        }

        /// @brief Populates default positive bounds for the scale parameter.
        /// @param[out] out Destination Interval span of length 1.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> bounds_into(std::span<cxxgpr::Interval> out) const override
        {
            out[0] = cxxgpr::Interval::default_positive();
            return {};
        }

        /// @brief Evaluates float scale_ * exp(-d2) over the requested square matrix triangle.
        /// @param[in] dist Pairwise squared Euclidean distance matrix.
        /// @param[out] out Output kernel Gram matrix buffer.
        /// @param[in] uplo Triangle mask controlling which entries are written.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply(
            cxxgpr::MatrixRef<float> dist, cxxgpr::MatrixMut<float> out, cxxgpr::Triangle uplo
        ) const override
        {
            const std::size_t n = dist.rows();
            for (std::size_t col = 0; col < n; ++col)
            {
                const std::size_t rb = (uplo == cxxgpr::Triangle::Lower) ? col : 0U;
                const std::size_t re = (uplo == cxxgpr::Triangle::Upper) ? (col + 1U) : n;
                for (std::size_t row = rb; row < re; ++row)
                {
                    out(row, col) = scale_ * std::exp(-dist(row, col));
                }
            }
            return {};
        }

        /// @brief Evaluates float rectangular cross-covariance scale_ * exp(-d2).
        /// @param[in] dist Rectangular pairwise squared Euclidean distance matrix.
        /// @param[out] out Output cross-covariance matrix buffer.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply_cross(
            cxxgpr::MatrixRef<float> dist, cxxgpr::MatrixMut<float> out
        ) const override
        {
            for (std::size_t col = 0; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0; row < dist.rows(); ++row)
                {
                    out(row, col) = scale_ * std::exp(-dist(row, col));
                }
            }
            return {};
        }

        /// @brief Fills the prior float kernel diagonal with scale_.
        /// @param[out] out Destination diagonal span of length n.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> fill_diag(std::span<float> out) const override
        {
            std::ranges::fill(out, scale_);
            return {};
        }

        /// @brief Computes derivative dk / d(log scale) in single precision.
        /// @param[in] dist Pairwise squared Euclidean distance matrix.
        /// @param[out] d_k Output gradient matrix buffer.
        /// @param[in] param_idx Local parameter index (unused because num_params() == 1).
        /// @param[in] uplo Triangle mask controlling which entries are written.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> grad(
            cxxgpr::MatrixRef<float> dist,
            cxxgpr::MatrixMut<float> d_k,
            [[maybe_unused]] std::size_t param_idx,
            cxxgpr::Triangle uplo
        ) const override
        {
            return apply(dist, d_k, uplo);
        }

        /// @brief Computes second derivative d^2 k / d(log scale)^2 in single precision.
        /// @param[in] dist Pairwise squared Euclidean distance matrix.
        /// @param[out] d2_k Output Hessian matrix buffer.
        /// @param[in] i First local parameter index (unused because num_params() == 1).
        /// @param[in] j Second local parameter index (unused because num_params() == 1).
        /// @param[in] uplo Triangle mask controlling which entries are written.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> hess(
            cxxgpr::MatrixRef<float> dist,
            cxxgpr::MatrixMut<float> d2_k,
            [[maybe_unused]] std::size_t i,
            [[maybe_unused]] std::size_t j,
            cxxgpr::Triangle uplo
        ) const override
        {
            return apply(dist, d2_k, uplo);
        }

        /// @brief Returns the shared persistence identifier `"user.custom_scale"`.
        /// @return Static string view `"user.custom_scale"`.
        [[nodiscard]] std::string_view persist_id() const noexcept override { return "user.custom_scale"; }

    private:
        /// @brief Positive output scale factor stored in float precision.
        float scale_;
    };

    /// @brief Verifies save_with_factor() mmap roundtrip and save() refactor-on-load parity.
    /// @details Saves an L-BFGS model fitted with input/target transforms, checks bit-exact
    /// mmap predictions/LOO and typed extraction rejection/success, then checks refactor-on-load
    /// accuracy when saved without factors.
    TEST(PersistModelsTest, SaveWithFactorMmapAndSaveRefactorRoundtrip)
    {
        // Spec: FMT-01, FMT-02, FMT-03, FMT-04, FMT-05
        const auto spec =
            (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.2).value())
                * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array<double, 2>{0.9, 1.3}).value()))
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.02).value());

        const std::size_t n = 6;
        const std::size_t d = 2;
        const std::vector<double> x_train{
            -1.0, -0.4, 0.1, 0.6, 1.2, 1.7,  // col 0
            0.3, -0.7, 0.5, -0.2, 0.9, -0.5  // col 1
        };
        const std::vector<double> y_train{5.2, 4.1, 6.0, 5.8, 7.1, 4.9};
        const std::vector<double> x_query{-0.5, 0.8, 0.0, 0.4};

        const auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(spec)
                                 .with_input_transform(cxxgpr::transform::StandardizeInput())
                                 .with_target_transform(cxxgpr::transform::StandardizeTarget())
                                 .with_noise_variance(0.05)
                                 .value();
        auto fitted = trainer.fit(x_train, n, d, y_train);
        ASSERT_TRUE(fitted.has_value());

        const auto xq_ref = cxxgpr::MatrixRef<double>(x_query.data(), 2U, d, 2U);
        auto orig_pred = fitted->predict(xq_ref);
        auto orig_loo = fitted->loo_predict();
        ASSERT_TRUE(orig_pred.has_value());
        ASSERT_TRUE(orig_loo.has_value());

        // 1. save_with_factor -> mmap F64 factor L.
        const TempDirGuard dir_factor("factor_roundtrip");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fitted, dir_factor.path).has_value());
        EXPECT_TRUE(std::filesystem::exists(dir_factor.path / "config.json"));
        EXPECT_TRUE(std::filesystem::exists(dir_factor.path / "model.safetensors"));

        {
            auto loaded = cxxgpr::persist::LoadedGpr::load(dir_factor.path);
            ASSERT_TRUE(loaded.has_value());
            EXPECT_EQ(loaded->n(), n);
            EXPECT_EQ(loaded->d(), d);
            EXPECT_EQ(loaded->precision_kind(), cxxgpr::PersistPrecisionKind::Double);
            EXPECT_TRUE(loaded->has_factor());
            EXPECT_TRUE(loaded->has_mmap_factor());

            auto loaded_pred = loaded->predict(xq_ref);
            auto loaded_loo = loaded->loo_predict();
            ASSERT_TRUE(loaded_pred.has_value());
            ASSERT_TRUE(loaded_loo.has_value());
            for (std::size_t j = 0; j < 2U; ++j)
            {
                EXPECT_EQ(
                    std::bit_cast<std::uint64_t>(loaded_pred->mean[j]), std::bit_cast<std::uint64_t>(orig_pred->mean[j])
                );
                EXPECT_EQ(
                    std::bit_cast<std::uint64_t>(loaded_pred->variance[j]),
                    std::bit_cast<std::uint64_t>(orig_pred->variance[j])
                );
            }
            for (std::size_t i = 0; i < n; ++i)
            {
                EXPECT_EQ(
                    std::bit_cast<std::uint64_t>(loaded_loo->mean[i]), std::bit_cast<std::uint64_t>(orig_loo->mean[i])
                );
            }

            // Typed extraction and wrong-precision rejection.
            auto wrong_typed = loaded->into_fitted<cxxgpr::optimizer::Lbfgs, cxxgpr::SinglePrecision>();
            ASSERT_FALSE(wrong_typed.has_value());
            EXPECT_EQ(wrong_typed.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(wrong_typed.error().persist_error_kind(), cxxgpr::PersistErrorKind::WrongModel);

            auto typed = loaded->into_fitted<cxxgpr::optimizer::Lbfgs, cxxgpr::DoublePrecision>();
            ASSERT_TRUE(typed.has_value());
            EXPECT_NEAR(typed->neg_log_marginal_likelihood(), fitted->neg_log_marginal_likelihood(), 1e-12);
            auto typed_pred = typed->predict(xq_ref);
            ASSERT_TRUE(typed_pred.has_value());
            EXPECT_EQ(typed_pred->mean, orig_pred->mean);
        }

        // 2. save (has_factor = false) -> reconstructs L and alpha on load.
        const TempDirGuard dir_nofactor("nofactor_roundtrip");
        ASSERT_TRUE(cxxgpr::persist::save(*fitted, dir_nofactor.path).has_value());
        auto loaded_nf = cxxgpr::persist::LoadedGpr::load(dir_nofactor.path);
        ASSERT_TRUE(loaded_nf.has_value());
        EXPECT_FALSE(loaded_nf->has_factor());
        EXPECT_FALSE(loaded_nf->has_mmap_factor());
        auto pred_nf = loaded_nf->predict(xq_ref);
        ASSERT_TRUE(pred_nf.has_value());
        for (std::size_t j = 0; j < 2U; ++j)
        {
            EXPECT_NEAR(pred_nf->mean[j], orig_pred->mean[j], 1e-11);
            EXPECT_NEAR(pred_nf->variance[j], orig_pred->variance[j], 1e-11);
        }
        auto typed_nf = loaded_nf->into_fitted<cxxgpr::optimizer::Lbfgs, cxxgpr::DoublePrecision>();
        ASSERT_TRUE(typed_nf.has_value());
        EXPECT_NEAR(typed_nf->neg_log_marginal_likelihood(), fitted->neg_log_marginal_likelihood(), 1e-12);
    }

    /// @brief Verifies MixedPrecision<ReevaluateKernel> and CustomKernel PersistRegistry persistence.
    /// @details Confirms MixedReevaluate precision metadata and non-mmap load behavior, registry
    /// validation of empty/reserved/duplicate IDs, UnregisteredId rejection without registry, and
    /// predictive parity when loaded with a registered CustomKernel factory.
    TEST(PersistModelsTest, PrecisionVariantsAndRegistryRoundtrip)
    {
        // Spec: FMT-05, FMT-06
        const std::size_t n = 4;
        const std::size_t d = 1;
        const std::vector<double> x_train{0.0, 0.5, 1.0, 1.5};
        const std::vector<double> y_train{1.0, 1.2, 0.9, 0.5};
        const std::array<double, 1> xq{0.75};
        const auto xq_ref = cxxgpr::MatrixRef<double>(xq.data(), 1U, 1U, 1U);

        // MixedPrecision<ReevaluateKernel> roundtrip.
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const auto mp_trainer =
            cxxgpr::Gpr<cxxgpr::optimizer::Fixed, cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(spec);
        auto mp_fit = mp_trainer.factor(x_train, n, d, y_train);
        ASSERT_TRUE(mp_fit.has_value());

        const TempDirGuard dir_mp("mixed_reeval");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*mp_fit, dir_mp.path).has_value());
        auto mp_loaded = cxxgpr::persist::LoadedGpr::load(dir_mp.path);
        ASSERT_TRUE(mp_loaded.has_value());
        EXPECT_EQ(mp_loaded->precision_kind(), cxxgpr::PersistPrecisionKind::MixedReevaluate);
        EXPECT_FALSE(mp_loaded->has_mmap_factor());
        EXPECT_NEAR(mp_loaded->predict(xq_ref)->mean[0], mp_fit->predict(xq_ref)->mean[0], 1e-12);

        // CustomKernel + PersistRegistry validation & roundtrip.
        cxxgpr::persist::PersistRegistry reg;
        EXPECT_EQ(
            reg.register_kernel("", [](std::string_view) { return std::unexpected(cxxgpr::GprError::empty_input()); })
                .error()
                .persist_error_kind(),
            cxxgpr::PersistErrorKind::InvalidPersistId
        );
        EXPECT_EQ(
            reg.register_kernel(
                   "cxxgpr.reserved", [](std::string_view) { return std::unexpected(cxxgpr::GprError::empty_input()); }
            )
                .error()
                .persist_error_kind(),
            cxxgpr::PersistErrorKind::InvalidPersistId
        );

        auto make_custom = [](double scale)
        {
            return cxxgpr::CustomKernel::create(
                std::make_unique<CustomScaleKernel>(scale),
                std::make_unique<CustomScaleKernelFloat>(static_cast<float>(scale))
            )
                .value();
        };
        ASSERT_TRUE(
            reg.register_kernel(
                   "user.custom_scale",
                   [&](std::string_view state) -> std::expected<cxxgpr::CustomKernel, cxxgpr::GprError>
        {
            const double s = std::stod(std::string(state));
            return make_custom(s);
        }
            ).has_value()
        );
        // Duplicate registration must fail with InvalidPersistId.
        EXPECT_EQ(
            reg.register_kernel("user.custom_scale", [&](std::string_view) { return make_custom(1.0); })
                .error()
                .persist_error_kind(),
            cxxgpr::PersistErrorKind::InvalidPersistId
        );

        const cxxgpr::KernelSpec custom_spec(make_custom(2.5));
        const auto c_fit = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(custom_spec).factor(x_train, n, d, y_train);
        ASSERT_TRUE(c_fit.has_value());

        const TempDirGuard dir_custom("custom_kernel");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*c_fit, dir_custom.path, reg).has_value());

        // Loading without registration fails with UnregisteredId.
        auto unreg_load = cxxgpr::persist::LoadedGpr::load(dir_custom.path);
        ASSERT_FALSE(unreg_load.has_value());
        EXPECT_EQ(unreg_load.error().persist_error_kind(), cxxgpr::PersistErrorKind::UnregisteredId);

        // Loading with registry succeeds.
        auto reg_load = cxxgpr::persist::LoadedGpr::load(dir_custom.path, reg);
        ASSERT_TRUE(reg_load.has_value());
        EXPECT_NEAR(reg_load->predict(xq_ref)->mean[0], c_fit->predict(xq_ref)->mean[0], 1e-12);
    }

    /// @brief Negative controls verifying persistence load rejects missing, corrupt, and foreign artifacts.
    /// @details Checks Io errors on missing directories, Config errors on malformed or deeply nested
    /// JSON, UnsupportedPersistVersion, WrongModel rejections for sgpr/ldlt/distance configs, and
    /// Tensor errors on truncated or deeply nested safetensors headers.
    TEST(PersistModelsTest, NegativeControlsRejectCorruptAndUnsupportedFiles)
    {
        // Spec: FMT-01, FMT-02, FMT-08
        // Ensure loading from a non-existent directory fails with PersistErrorKind::Io.
        const TempDirGuard missing_dir("does_not_exist");
        std::error_code ec;
        std::filesystem::remove_all(missing_dir.path, ec);
        auto err_io = cxxgpr::persist::LoadedGpr::load(missing_dir.path);
        ASSERT_FALSE(err_io.has_value());
        EXPECT_EQ(err_io.error().persist_error_kind(), cxxgpr::PersistErrorKind::Io);

        const std::vector<double> x_train{0.0, 1.0};
        const std::vector<double> y_train{0.5, 1.5};
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const auto fit = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).factor(x_train, 2U, 1U, y_train);
        ASSERT_TRUE(fit.has_value());

        // Helper lambda to mutate config.json and expect specific error.
        auto check_bad_config = [&](const std::string& name, const std::string& json_text,
                                    cxxgpr::GprErrorCode expected_code,
                                    std::optional<cxxgpr::PersistErrorKind> expected_pk)
        {
            const TempDirGuard dir("bad_" + name);
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit, dir.path).has_value());
            {
                std::ofstream ofs(dir.path / "config.json", std::ios::binary | std::ios::trunc);
                ofs << json_text;
            }
            auto res = cxxgpr::persist::LoadedGpr::load(dir.path);
            ASSERT_FALSE(res.has_value());
            EXPECT_EQ(res.error().code(), expected_code);
            if (expected_pk.has_value())
            {
                EXPECT_EQ(res.error().persist_error_kind(), *expected_pk);
            }
        };

        check_bad_config(
            "syntax", "{not valid json", cxxgpr::GprErrorCode::PersistFailed, cxxgpr::PersistErrorKind::Config
        );
        {
            std::string deep_json;
            for (std::size_t i = 0; i < 130U; ++i)
            {
                deep_json += "{\"a\":";
            }
            deep_json += "1";
            for (std::size_t i = 0; i < 130U; ++i)
            {
                deep_json += "}";
            }
            check_bad_config(
                "deep_nesting", deep_json, cxxgpr::GprErrorCode::PersistFailed, cxxgpr::PersistErrorKind::Config
            );
        }
        check_bad_config(
            "version", R"({"format_version": 99, "factor_kind": "llt"})",
            cxxgpr::GprErrorCode::UnsupportedPersistVersion, std::nullopt
        );
        check_bad_config(
            "wrong_model", R"({"format_version": 1, "model": "sgpr", "factor_kind": "llt"})",
            cxxgpr::GprErrorCode::PersistFailed, cxxgpr::PersistErrorKind::WrongModel
        );
        check_bad_config(
            "ldlt", R"({"format_version": 1, "factor_kind": "ldlt"})", cxxgpr::GprErrorCode::PersistFailed,
            cxxgpr::PersistErrorKind::WrongModel
        );
        check_bad_config(
            "distance", R"({"format_version": 1, "distance": {}, "factor_kind": "llt"})",
            cxxgpr::GprErrorCode::PersistFailed, cxxgpr::PersistErrorKind::WrongModel
        );

        // Corrupt safetensors header -> PersistFailed(Tensor).
        const TempDirGuard dir_st("bad_safetensors");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit, dir_st.path).has_value());
        {
            std::ofstream ofs(dir_st.path / "model.safetensors", std::ios::binary | std::ios::trunc);
            const std::uint64_t bad_len = 999999ULL;
            ofs.write(reinterpret_cast<const char*>(&bad_len), sizeof(bad_len));
        }
        auto err_st = cxxgpr::persist::LoadedGpr::load(dir_st.path);
        ASSERT_FALSE(err_st.has_value());
        EXPECT_EQ(err_st.error().persist_error_kind(), cxxgpr::PersistErrorKind::Tensor);

        // Deeply nested JSON in safetensors header -> PersistFailed(Tensor).
        const TempDirGuard dir_st_deep("bad_safetensors_deep");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit, dir_st_deep.path).has_value());
        {
            std::string deep_hdr;
            for (std::size_t i = 0; i < 130U; ++i)
            {
                deep_hdr += "{\"a\":";
            }
            deep_hdr += "1";
            for (std::size_t i = 0; i < 130U; ++i)
            {
                deep_hdr += "}";
            }
            while ((deep_hdr.size() % 8U) != 0U)
            {
                deep_hdr.push_back(' ');
            }
            const std::uint64_t hdr_len = deep_hdr.size();
            std::ofstream ofs(dir_st_deep.path / "model.safetensors", std::ios::binary | std::ios::trunc);
            ofs.write(reinterpret_cast<const char*>(&hdr_len), sizeof(hdr_len));
            ofs.write(deep_hdr.data(), static_cast<std::streamsize>(deep_hdr.size()));
        }
        auto err_st_deep = cxxgpr::persist::LoadedGpr::load(dir_st_deep.path);
        ASSERT_FALSE(err_st_deep.has_value());
        EXPECT_EQ(err_st_deep.error().persist_error_kind(), cxxgpr::PersistErrorKind::Tensor);
    }

    /// @brief Checks that query batch sizes may shrink and grow in any order on one LoadedGpr.
    /// @details Fits a 2-feature model under precision policy P with fitted input and target transforms,
    /// saves it, and loads it several times. One loaded model sees an 8-point batch first; the following
    /// 3-point batch must match bitwise a loaded model that never saw the larger batch, for predict,
    /// predict_covariance, and sample, and returning to 8 points must match a model that never saw the
    /// smaller batch. Finally audits that predict_into() allocates nothing for a smaller batch after a larger one.
    /// @tparam P Precision policy of the saved model.
    /// @param[in] dir_name Unique temporary directory suffix for this policy.
    /// @param[in] with_factor Whether the factor is stored (save_with_factor) or rebuilt on load (save).
    template <cxxgpr::PrecisionPolicyConcept P>
    void verify_loaded_query_batch_may_shrink(const std::string& dir_name, bool with_factor)
    {
        const std::size_t n = 20;
        const std::size_t d = 2;
        std::vector<double> x(n * d);
        std::vector<double> y(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            x[i] = 0.3 * static_cast<double>(i);
            x[n + i] = 0.25 * static_cast<double>((i * 5U) % 7U);
            y[i] = 4.0 + std::sin(x[i]) + (0.5 * x[n + i]);
        }
        const std::vector<double> big{0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5, 2.9, 0.2, 1.4, 0.6, 1.0, 0.3, 1.2, 0.8, 0.4};
        const std::vector<double> small{0.2, 1.1, 2.3, 0.7, 0.1, 1.3};
        const auto big_ref = cxxgpr::MatrixRef<double>(big.data(), 8U, d, 8U);
        const auto small_ref = cxxgpr::MatrixRef<double>(small.data(), 3U, d, 3U);

        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array<double, 2>{1.0, 1.5}).value());
        const auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Fixed, P>(spec)
                                 .with_input_transform(cxxgpr::transform::StandardizeInput())
                                 .with_target_transform(cxxgpr::transform::StandardizeTarget())
                                 .with_noise_variance(1e-3);
        ASSERT_TRUE(trainer.has_value());
        const auto fitted = trainer->factor(x, n, d, y);
        ASSERT_TRUE(fitted.has_value());

        const TempDirGuard dir(dir_name);
        if (with_factor)
        {
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fitted, dir.path).has_value());
        }
        else
        {
            ASSERT_TRUE(cxxgpr::persist::save(*fitted, dir.path).has_value());
        }
        // Every load() yields an independent model with an empty query workspace.
        const auto load = [&]
        {
            return cxxgpr::persist::LoadedGpr::load(dir.path);
        };

        // predict: large -> small -> large on one loaded model.
        {
            auto used = load();
            auto never_big = load();
            auto never_small = load();
            ASSERT_TRUE(used.has_value());
            ASSERT_TRUE(never_big.has_value());
            ASSERT_TRUE(never_small.has_value());
            ASSERT_TRUE(used->predict(big_ref).has_value());
            auto shrunk = used->predict(small_ref);
            auto shrunk_expected = never_big->predict(small_ref);
            ASSERT_TRUE(shrunk.has_value()) << shrunk.error().message();
            ASSERT_TRUE(shrunk_expected.has_value());
            EXPECT_EQ(shrunk->mean, shrunk_expected->mean);
            EXPECT_EQ(shrunk->variance, shrunk_expected->variance);

            auto regrown = used->predict(big_ref);
            auto regrown_expected = never_small->predict(big_ref);
            ASSERT_TRUE(regrown.has_value()) << regrown.error().message();
            ASSERT_TRUE(regrown_expected.has_value());
            EXPECT_EQ(regrown->mean, regrown_expected->mean);
            EXPECT_EQ(regrown->variance, regrown_expected->variance);
        }

        // predict_covariance: large -> small -> large.
        {
            auto used = load();
            auto never_big = load();
            auto never_small = load();
            ASSERT_TRUE(used.has_value());
            ASSERT_TRUE(never_big.has_value());
            ASSERT_TRUE(never_small.has_value());
            ASSERT_TRUE(used->predict_covariance(big_ref).has_value());
            auto shrunk = used->predict_covariance(small_ref);
            auto shrunk_expected = never_big->predict_covariance(small_ref);
            ASSERT_TRUE(shrunk.has_value()) << shrunk.error().message();
            ASSERT_TRUE(shrunk_expected.has_value());
            EXPECT_EQ(shrunk->mean, shrunk_expected->mean);
            EXPECT_EQ(shrunk->covariance, shrunk_expected->covariance);

            auto regrown = used->predict_covariance(big_ref);
            auto regrown_expected = never_small->predict_covariance(big_ref);
            ASSERT_TRUE(regrown.has_value()) << regrown.error().message();
            ASSERT_TRUE(regrown_expected.has_value());
            EXPECT_EQ(regrown->mean, regrown_expected->mean);
            EXPECT_EQ(regrown->covariance, regrown_expected->covariance);
        }

        // sample with a fixed seed: large -> small -> large.
        {
            auto used = load();
            auto never_big = load();
            auto never_small = load();
            ASSERT_TRUE(used.has_value());
            ASSERT_TRUE(never_big.has_value());
            ASSERT_TRUE(never_small.has_value());
            ASSERT_TRUE(used->sample(big_ref, 2U, 7U).has_value());
            auto shrunk = used->sample(small_ref, 2U, 7U);
            auto shrunk_expected = never_big->sample(small_ref, 2U, 7U);
            ASSERT_TRUE(shrunk.has_value()) << shrunk.error().message();
            ASSERT_TRUE(shrunk_expected.has_value());
            EXPECT_EQ(*shrunk, *shrunk_expected);

            auto regrown = used->sample(big_ref, 2U, 7U);
            auto regrown_expected = never_small->sample(big_ref, 2U, 7U);
            ASSERT_TRUE(regrown.has_value()) << regrown.error().message();
            ASSERT_TRUE(regrown_expected.has_value());
            EXPECT_EQ(*regrown, *regrown_expected);
        }

        // predict_into: after the larger batch sized the workspace and the output, a smaller batch and a
        // return to the larger batch perform no heap allocation.
        {
            auto used = load();
            auto never_big = load();
            ASSERT_TRUE(used.has_value());
            ASSERT_TRUE(never_big.has_value());
            cxxgpr::Prediction<double> out;
            ASSERT_TRUE(used->predict_into(big_ref, out).has_value());
            {
                const cxxgpr::test_support::NoAllocationScope no_alloc;
                const auto res = used->predict_into(small_ref, out);
                const std::size_t allocations = no_alloc.delta_allocations();
                ASSERT_TRUE(res.has_value());
                EXPECT_EQ(allocations, 0U);
            }
            auto shrunk_expected = never_big->predict(small_ref);
            ASSERT_TRUE(shrunk_expected.has_value());
            EXPECT_EQ(out.mean, shrunk_expected->mean);
            EXPECT_EQ(out.variance, shrunk_expected->variance);
            {
                const cxxgpr::test_support::NoAllocationScope no_alloc;
                const auto res = used->predict_into(big_ref, out);
                const std::size_t allocations = no_alloc.delta_allocations();
                ASSERT_TRUE(res.has_value());
                EXPECT_EQ(allocations, 0U);
            }
        }
    }

    /// @brief Verifies a LoadedGpr accepts a smaller query batch after a larger one.
    /// @details Covers all four precision policies with a stored factor, plus the rebuild-on-load path,
    /// for predict, predict_covariance, sample, and allocation-free predict_into.
    TEST(PersistModelsTest, LoadedQueryBatchSizeMayShrinkAfterLargerBatch)
    {
        // Spec: REQ-07, REQ-10, REQ-11, REQ-12, DES-05, DES-16, DES-17, DES-18, FMT-05
        {
            SCOPED_TRACE("DoublePrecision, stored factor");
            verify_loaded_query_batch_may_shrink<cxxgpr::DoublePrecision>("shrink_f64", true);
        }
        {
            SCOPED_TRACE("DoublePrecision, factor rebuilt on load");
            verify_loaded_query_batch_may_shrink<cxxgpr::DoublePrecision>("shrink_f64_refactor", false);
        }
        {
            SCOPED_TRACE("SinglePrecision");
            verify_loaded_query_batch_may_shrink<cxxgpr::SinglePrecision>("shrink_f32", true);
        }
        {
            SCOPED_TRACE("MixedPrecision<PromoteStorage>");
            verify_loaded_query_batch_may_shrink<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(
                "shrink_mixed_promote", true
            );
        }
        {
            SCOPED_TRACE("MixedPrecision<ReevaluateKernel>");
            verify_loaded_query_batch_may_shrink<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(
                "shrink_mixed_reevaluate", true
            );
        }
    }

}  // namespace
