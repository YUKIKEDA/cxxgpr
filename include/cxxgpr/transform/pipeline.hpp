// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/transform/input.hpp"
#include "cxxgpr/transform/target.hpp"

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <vector>

namespace cxxgpr::transform {

/// Fitted sequential composition of input feature transforms.
class FittedPipeline final : public FittedInputTransform {
public:
    /// Constructs an empty fitted input pipeline (equivalent to identity).
    FittedPipeline() noexcept = default;

    /// Constructs a fitted input pipeline from an ordered list of fitted steps.
    explicit FittedPipeline(std::vector<std::unique_ptr<FittedInputTransform>> steps) noexcept;

    /// Deep-copies the fitted input pipeline.
    FittedPipeline(const FittedPipeline& other);

    /// Deep-copy assigns the fitted input pipeline.
    FittedPipeline& operator=(const FittedPipeline& other);

    /// Moves the fitted input pipeline.
    FittedPipeline(FittedPipeline&&) noexcept = default;

    /// Move-assigns the fitted input pipeline.
    FittedPipeline& operator=(FittedPipeline&&) noexcept = default;

    ~FittedPipeline() override = default;

    /// Returns the number of transform steps.
    [[nodiscard]] std::size_t size() const noexcept {
        return steps_.size();
    }

    /// Returns true if the pipeline has no steps.
    [[nodiscard]] bool empty() const noexcept {
        return steps_.empty();
    }

    /// Returns a read-only view of the fitted steps in forward application order.
    [[nodiscard]] std::span<const std::unique_ptr<FittedInputTransform>> steps() const noexcept {
        return steps_;
    }

    /// Applies all steps in forward order to x in place.
    [[nodiscard]] std::expected<void, GprError> apply(std::span<double> x, std::size_t n_rows,
                                                      std::size_t n_cols) const override;

    /// Applies all steps in forward order from x into out without allocation.
    [[nodiscard]] std::expected<void, GprError> apply_into(std::span<const double> x,
                                                           std::size_t n_rows, std::size_t n_cols,
                                                           std::span<double> out) const override;

    /// Applies all steps in reverse order to invert the transform in place.
    [[nodiscard]] std::expected<void, GprError>
    inverse_apply(std::span<double> x, std::size_t n_rows, std::size_t n_cols) const override;

    /// Applies all steps in reverse order from x into out without allocation.
    [[nodiscard]] std::expected<void, GprError>
    inverse_apply_into(std::span<const double> x, std::size_t n_rows, std::size_t n_cols,
                       std::span<double> out) const override;

    /// Returns a polymorphic deep copy of this fitted input pipeline.
    [[nodiscard]] std::unique_ptr<FittedInputTransform> clone() const override;

private:
    std::vector<std::unique_ptr<FittedInputTransform>> steps_;
};

/// Unfitted sequential composition of input feature transforms.
class Pipeline final : public UnfittedInputTransform {
public:
    /// Constructs an empty input pipeline (acts as identity until steps are added).
    Pipeline() noexcept = default;

    /// Constructs an input pipeline from an ordered list of unfitted steps.
    explicit Pipeline(std::vector<std::unique_ptr<UnfittedInputTransform>> steps) noexcept;

    /// Deep-copies the unfitted input pipeline.
    Pipeline(const Pipeline& other);

    /// Deep-copy assigns the unfitted input pipeline.
    Pipeline& operator=(const Pipeline& other);

    /// Moves the unfitted input pipeline.
    Pipeline(Pipeline&&) noexcept = default;

    /// Move-assigns the unfitted input pipeline.
    Pipeline& operator=(Pipeline&&) noexcept = default;

    ~Pipeline() override = default;

    /// Appends a step by cloning it onto the end of this pipeline.
    Pipeline& then(const UnfittedInputTransform& step) &;

    /// Appends a step by cloning it onto the end of this rvalue pipeline.
    Pipeline&& then(const UnfittedInputTransform& step) &&;

    /// Appends an owned step onto the end of this pipeline.
    Pipeline& then(std::unique_ptr<UnfittedInputTransform> step) &;

    /// Appends an owned step onto the end of this rvalue pipeline.
    Pipeline&& then(std::unique_ptr<UnfittedInputTransform> step) &&;

    /// Returns the number of transform steps.
    [[nodiscard]] std::size_t size() const noexcept {
        return steps_.size();
    }

    /// Returns true if the pipeline has no steps.
    [[nodiscard]] bool empty() const noexcept {
        return steps_.empty();
    }

    /// Returns a read-only view of the unfitted steps in forward order.
    [[nodiscard]] std::span<const std::unique_ptr<UnfittedInputTransform>> steps() const noexcept {
        return steps_;
    }

    /// Fits each step sequentially on the output of the preceding step.
    [[nodiscard]] std::expected<FittedPipeline, GprError>
    fit_typed(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const;

    /// Fits each step sequentially and returns a boxed fitted pipeline.
    [[nodiscard]] std::expected<std::unique_ptr<FittedInputTransform>, GprError>
    fit(std::span<const double> x, std::size_t n_rows, std::size_t n_cols) const override;

    /// Returns a polymorphic deep copy of this unfitted input pipeline.
    [[nodiscard]] std::unique_ptr<UnfittedInputTransform> clone() const override;

private:
    std::vector<std::unique_ptr<UnfittedInputTransform>> steps_;
};

/// Fitted sequential composition of target transforms.
class FittedTargetPipeline final : public FittedTargetTransform {
public:
    /// Constructs an empty fitted target pipeline (equivalent to identity).
    FittedTargetPipeline() noexcept = default;

    /// Constructs a fitted target pipeline from an ordered list of fitted steps.
    explicit FittedTargetPipeline(
        std::vector<std::unique_ptr<FittedTargetTransform>> steps) noexcept;

    /// Deep-copies the fitted target pipeline.
    FittedTargetPipeline(const FittedTargetPipeline& other);

    /// Deep-copy assigns the fitted target pipeline.
    FittedTargetPipeline& operator=(const FittedTargetPipeline& other);

    /// Moves the fitted target pipeline.
    FittedTargetPipeline(FittedTargetPipeline&&) noexcept = default;

    /// Move-assigns the fitted target pipeline.
    FittedTargetPipeline& operator=(FittedTargetPipeline&&) noexcept = default;

    ~FittedTargetPipeline() override = default;

    /// Returns the number of target transform steps.
    [[nodiscard]] std::size_t size() const noexcept {
        return steps_.size();
    }

    /// Returns true if the target pipeline has no steps.
    [[nodiscard]] bool empty() const noexcept {
        return steps_.empty();
    }

    /// Returns a read-only view of the fitted steps in forward application order.
    [[nodiscard]] std::span<const std::unique_ptr<FittedTargetTransform>> steps() const noexcept {
        return steps_;
    }

    /// Applies all steps in forward order to y in place.
    [[nodiscard]] std::expected<void, GprError> transform(std::span<double> y) const override;

    /// Applies all steps in forward order from y into out without allocation.
    [[nodiscard]] std::expected<void, GprError>
    transform_into(std::span<const double> y, std::span<double> out) const override;

    /// Applies inverse mean transforms in reverse order in place.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_mean(std::span<double> mean) const override;

    /// Applies inverse mean transforms in reverse order from mean into out without allocation.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_mean_into(std::span<const double> mean, std::span<double> out) const override;

    /// Applies inverse variance transforms in reverse order in place.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_variance(std::span<double> var) const override;

    /// Applies inverse variance transforms in reverse order from var into out without allocation.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_variance_into(std::span<const double> var,
                                    std::span<double> out) const override;

    /// Applies inverse covariance transforms in reverse order in place.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_covariance(std::span<double> cov) const override;

    /// Applies inverse covariance transforms in reverse order from cov into out without allocation.
    [[nodiscard]] std::expected<void, GprError>
    inverse_transform_covariance_into(std::span<const double> cov,
                                      std::span<double> out) const override;

    /// Returns a polymorphic deep copy of this fitted target pipeline.
    [[nodiscard]] std::unique_ptr<FittedTargetTransform> clone() const override;

private:
    std::vector<std::unique_ptr<FittedTargetTransform>> steps_;
};

/// Unfitted sequential composition of target transforms.
class TargetPipeline final : public UnfittedTargetTransform {
public:
    /// Constructs an empty target pipeline (acts as identity until steps are added).
    TargetPipeline() noexcept = default;

    /// Constructs a target pipeline from an ordered list of unfitted steps.
    explicit TargetPipeline(std::vector<std::unique_ptr<UnfittedTargetTransform>> steps) noexcept;

    /// Deep-copies the unfitted target pipeline.
    TargetPipeline(const TargetPipeline& other);

    /// Deep-copy assigns the unfitted target pipeline.
    TargetPipeline& operator=(const TargetPipeline& other);

    /// Moves the unfitted target pipeline.
    TargetPipeline(TargetPipeline&&) noexcept = default;

    /// Move-assigns the unfitted target pipeline.
    TargetPipeline& operator=(TargetPipeline&&) noexcept = default;

    ~TargetPipeline() override = default;

    /// Appends a step by cloning it onto the end of this target pipeline.
    TargetPipeline& then(const UnfittedTargetTransform& step) &;

    /// Appends a step by cloning it onto the end of this rvalue target pipeline.
    TargetPipeline&& then(const UnfittedTargetTransform& step) &&;

    /// Appends an owned step onto the end of this target pipeline.
    TargetPipeline& then(std::unique_ptr<UnfittedTargetTransform> step) &;

    /// Appends an owned step onto the end of this rvalue target pipeline.
    TargetPipeline&& then(std::unique_ptr<UnfittedTargetTransform> step) &&;

    /// Returns the number of target transform steps.
    [[nodiscard]] std::size_t size() const noexcept {
        return steps_.size();
    }

    /// Returns true if the target pipeline has no steps.
    [[nodiscard]] bool empty() const noexcept {
        return steps_.empty();
    }

    /// Returns a read-only view of the unfitted steps in forward order.
    [[nodiscard]] std::span<const std::unique_ptr<UnfittedTargetTransform>> steps() const noexcept {
        return steps_;
    }

    /// Fits each step sequentially on the output of the preceding step.
    [[nodiscard]] std::expected<FittedTargetPipeline, GprError>
    fit_typed(std::span<const double> y) const;

    /// Fits each step sequentially and returns a boxed fitted target pipeline.
    [[nodiscard]] std::expected<std::unique_ptr<FittedTargetTransform>, GprError>
    fit(std::span<const double> y) const override;

    /// Returns a polymorphic deep copy of this unfitted target pipeline.
    [[nodiscard]] std::unique_ptr<UnfittedTargetTransform> clone() const override;

private:
    std::vector<std::unique_ptr<UnfittedTargetTransform>> steps_;
};

} // namespace cxxgpr::transform
