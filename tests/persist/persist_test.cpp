// SPDX-License-Identifier: Apache-2.0
// Spec: FMT-01, FMT-02, FMT-03, FMT-04, FMT-05, FMT-06, FMT-08

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
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace {

struct TempDirGuard {
    std::filesystem::path path;

    explicit TempDirGuard(const std::string& name) {
        path = std::filesystem::temp_directory_path() / ("cxxgpr_test_" + name);
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
        std::filesystem::create_directories(path, ec);
    }

    ~TempDirGuard() {
        std::error_code ec;
        try {
            std::filesystem::remove_all(path, ec);
        } catch (...) {
            ec.clear();
        }
    }
};

class CustomScaleKernel final : public cxxgpr::KernelTerm<double> {
public:
    explicit CustomScaleKernel(double scale) : scale_(scale) {}

    [[nodiscard]] std::unique_ptr<cxxgpr::KernelTerm<double>> clone() const override {
        return std::make_unique<CustomScaleKernel>(scale_);
    }

    [[nodiscard]] std::size_t num_params() const noexcept override {
        return 1U;
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError>
    get_params(std::span<double> out) const override {
        out[0] = std::log(scale_);
        return {};
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError>
    set_params(std::span<const double> params) override {
        scale_ = std::exp(params[0]);
        return {};
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError>
    bounds_into(std::span<cxxgpr::Interval> out) const override {
        out[0] = cxxgpr::Interval::default_positive();
        return {};
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError>
    apply(cxxgpr::MatrixRef<double> dist, cxxgpr::MatrixMut<double> out,
          cxxgpr::Triangle uplo) const override {
        const std::size_t n = dist.rows();
        for (std::size_t col = 0; col < n; ++col) {
            const std::size_t rb = (uplo == cxxgpr::Triangle::Lower) ? col : 0U;
            const std::size_t re = (uplo == cxxgpr::Triangle::Upper) ? (col + 1U) : n;
            for (std::size_t row = rb; row < re; ++row) {
                out(row, col) = scale_ * std::exp(-dist(row, col));
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError>
    apply_cross(cxxgpr::MatrixRef<double> dist, cxxgpr::MatrixMut<double> out) const override {
        for (std::size_t col = 0; col < dist.cols(); ++col) {
            for (std::size_t row = 0; row < dist.rows(); ++row) {
                out(row, col) = scale_ * std::exp(-dist(row, col));
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError>
    fill_diag(std::span<double> out) const override {
        std::ranges::fill(out, scale_);
        return {};
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError> grad(cxxgpr::MatrixRef<double> dist,
                                                             cxxgpr::MatrixMut<double> d_k,
                                                             [[maybe_unused]] std::size_t param_idx,
                                                             cxxgpr::Triangle uplo) const override {
        return apply(dist, d_k, uplo);
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError> hess(cxxgpr::MatrixRef<double> dist,
                                                             cxxgpr::MatrixMut<double> d2_k,
                                                             [[maybe_unused]] std::size_t i,
                                                             [[maybe_unused]] std::size_t j,
                                                             cxxgpr::Triangle uplo) const override {
        return apply(dist, d2_k, uplo);
    }

    [[nodiscard]] std::string_view persist_id() const noexcept override {
        return "user.custom_scale";
    }

    [[nodiscard]] std::expected<std::string, cxxgpr::GprError> persist_state_json() const override {
        return std::to_string(scale_);
    }

private:
    double scale_;
};

class CustomScaleKernelFloat final : public cxxgpr::KernelTerm<float> {
public:
    explicit CustomScaleKernelFloat(float scale) : scale_(scale) {}

    [[nodiscard]] std::unique_ptr<cxxgpr::KernelTerm<float>> clone() const override {
        return std::make_unique<CustomScaleKernelFloat>(scale_);
    }

    [[nodiscard]] std::size_t num_params() const noexcept override {
        return 1U;
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError>
    get_params(std::span<double> out) const override {
        out[0] = std::log(static_cast<double>(scale_));
        return {};
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError>
    set_params(std::span<const double> params) override {
        scale_ = static_cast<float>(std::exp(params[0]));
        return {};
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError>
    bounds_into(std::span<cxxgpr::Interval> out) const override {
        out[0] = cxxgpr::Interval::default_positive();
        return {};
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError>
    apply(cxxgpr::MatrixRef<float> dist, cxxgpr::MatrixMut<float> out,
          cxxgpr::Triangle uplo) const override {
        const std::size_t n = dist.rows();
        for (std::size_t col = 0; col < n; ++col) {
            const std::size_t rb = (uplo == cxxgpr::Triangle::Lower) ? col : 0U;
            const std::size_t re = (uplo == cxxgpr::Triangle::Upper) ? (col + 1U) : n;
            for (std::size_t row = rb; row < re; ++row) {
                out(row, col) = scale_ * std::exp(-dist(row, col));
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError>
    apply_cross(cxxgpr::MatrixRef<float> dist, cxxgpr::MatrixMut<float> out) const override {
        for (std::size_t col = 0; col < dist.cols(); ++col) {
            for (std::size_t row = 0; row < dist.rows(); ++row) {
                out(row, col) = scale_ * std::exp(-dist(row, col));
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError>
    fill_diag(std::span<float> out) const override {
        std::ranges::fill(out, scale_);
        return {};
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError> grad(cxxgpr::MatrixRef<float> dist,
                                                             cxxgpr::MatrixMut<float> d_k,
                                                             [[maybe_unused]] std::size_t param_idx,
                                                             cxxgpr::Triangle uplo) const override {
        return apply(dist, d_k, uplo);
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError> hess(cxxgpr::MatrixRef<float> dist,
                                                             cxxgpr::MatrixMut<float> d2_k,
                                                             [[maybe_unused]] std::size_t i,
                                                             [[maybe_unused]] std::size_t j,
                                                             cxxgpr::Triangle uplo) const override {
        return apply(dist, d2_k, uplo);
    }

    [[nodiscard]] std::string_view persist_id() const noexcept override {
        return "user.custom_scale";
    }

private:
    float scale_;
};

TEST(PersistModelsTest, SaveWithFactorMmapAndSaveRefactorRoundtrip) {
    // Spec: FMT-01, FMT-02, FMT-03, FMT-04, FMT-05
    const auto spec = (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.2).value()) *
                       cxxgpr::KernelSpec(
                           cxxgpr::RbfArdKernel::create(std::array<double, 2>{0.9, 1.3}).value())) +
                      cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.02).value());

    const std::size_t n = 6;
    const std::size_t d = 2;
    const std::vector<double> x_train{
        -1.0, -0.4, 0.1, 0.6,  1.2, 1.7, // col 0
        0.3,  -0.7, 0.5, -0.2, 0.9, -0.5 // col 1
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
        for (std::size_t j = 0; j < 2U; ++j) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(loaded_pred->mean[j]),
                      std::bit_cast<std::uint64_t>(orig_pred->mean[j]));
            EXPECT_EQ(std::bit_cast<std::uint64_t>(loaded_pred->variance[j]),
                      std::bit_cast<std::uint64_t>(orig_pred->variance[j]));
        }
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(loaded_loo->mean[i]),
                      std::bit_cast<std::uint64_t>(orig_loo->mean[i]));
        }

        // Typed extraction and wrong-precision rejection.
        auto wrong_typed = loaded->into_fitted<cxxgpr::optimizer::Lbfgs, cxxgpr::SinglePrecision>();
        ASSERT_FALSE(wrong_typed.has_value());
        EXPECT_EQ(wrong_typed.error().code(), cxxgpr::GprErrorCode::PersistFailed);
        EXPECT_EQ(wrong_typed.error().persist_error_kind(), cxxgpr::PersistErrorKind::WrongModel);

        auto typed = loaded->into_fitted<cxxgpr::optimizer::Lbfgs, cxxgpr::DoublePrecision>();
        ASSERT_TRUE(typed.has_value());
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
    for (std::size_t j = 0; j < 2U; ++j) {
        EXPECT_NEAR(pred_nf->mean[j], orig_pred->mean[j], 1e-11);
        EXPECT_NEAR(pred_nf->variance[j], orig_pred->variance[j], 1e-11);
    }
}

TEST(PersistModelsTest, PrecisionVariantsAndRegistryRoundtrip) {
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
        cxxgpr::Gpr<cxxgpr::optimizer::Fixed, cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(
            spec);
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
    EXPECT_EQ(reg.register_kernel("",
                                  [](std::string_view) {
                                      return std::unexpected(cxxgpr::GprError::empty_input());
                                  })
                  .error()
                  .persist_error_kind(),
              cxxgpr::PersistErrorKind::InvalidPersistId);
    EXPECT_EQ(reg.register_kernel("cxxgpr.reserved",
                                  [](std::string_view) {
                                      return std::unexpected(cxxgpr::GprError::empty_input());
                                  })
                  .error()
                  .persist_error_kind(),
              cxxgpr::PersistErrorKind::InvalidPersistId);

    auto make_custom = [](double scale) {
        return cxxgpr::CustomKernel::create(
                   std::make_unique<CustomScaleKernel>(scale),
                   std::make_unique<CustomScaleKernelFloat>(static_cast<float>(scale)))
            .value();
    };
    ASSERT_TRUE(reg.register_kernel("user.custom_scale",
                                    [&](std::string_view state)
                                        -> std::expected<cxxgpr::CustomKernel, cxxgpr::GprError> {
                                        const double s = std::stod(std::string(state));
                                        return make_custom(s);
                                    })
                    .has_value());
    // Duplicate registration must fail with InvalidPersistId.
    EXPECT_EQ(
        reg.register_kernel("user.custom_scale", [&](std::string_view) { return make_custom(1.0); })
            .error()
            .persist_error_kind(),
        cxxgpr::PersistErrorKind::InvalidPersistId);

    const cxxgpr::KernelSpec custom_spec(make_custom(2.5));
    const auto c_fit =
        cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(custom_spec).factor(x_train, n, d, y_train);
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

TEST(PersistModelsTest, NegativeControlsRejectCorruptAndUnsupportedFiles) {
    // Spec: FMT-01, FMT-02, FMT-08
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
                                std::optional<cxxgpr::PersistErrorKind> expected_pk) {
        const TempDirGuard dir("bad_" + name);
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit, dir.path).has_value());
        {
            std::ofstream ofs(dir.path / "config.json", std::ios::binary | std::ios::trunc);
            ofs << json_text;
        }
        auto res = cxxgpr::persist::LoadedGpr::load(dir.path);
        ASSERT_FALSE(res.has_value());
        EXPECT_EQ(res.error().code(), expected_code);
        if (expected_pk.has_value()) {
            EXPECT_EQ(res.error().persist_error_kind(), *expected_pk);
        }
    };

    check_bad_config("syntax", "{not valid json", cxxgpr::GprErrorCode::PersistFailed,
                     cxxgpr::PersistErrorKind::Config);
    check_bad_config("version", R"({"format_version": 99, "factor_kind": "llt"})",
                     cxxgpr::GprErrorCode::UnsupportedPersistVersion, std::nullopt);
    check_bad_config("wrong_model",
                     R"({"format_version": 1, "model": "sgpr", "factor_kind": "llt"})",
                     cxxgpr::GprErrorCode::PersistFailed, cxxgpr::PersistErrorKind::WrongModel);
    check_bad_config("ldlt", R"({"format_version": 1, "factor_kind": "ldlt"})",
                     cxxgpr::GprErrorCode::PersistFailed, cxxgpr::PersistErrorKind::WrongModel);
    check_bad_config("distance", R"({"format_version": 1, "distance": {}, "factor_kind": "llt"})",
                     cxxgpr::GprErrorCode::PersistFailed, cxxgpr::PersistErrorKind::WrongModel);

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
}

} // namespace
