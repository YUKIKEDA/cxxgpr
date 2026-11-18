// SPDX-License-Identifier: Apache-2.0
/// @file pipeline.hpp
/// @brief Sequential composable input and target transform pipelines.
/// @details Defines Pipeline, FittedPipeline, TargetPipeline, and FittedTargetPipeline for sequential forward
/// application, reverse-order inverse transformation, and stagewise fitting.

#pragma once

#include "cxxgpr/transform/input.hpp"
#include "cxxgpr/transform/target.hpp"

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <vector>

/// @brief Feature and target preprocessing transforms and composable pipelines.
namespace cxxgpr::transform
{

    /// @brief Fitted sequential composition of input feature transforms.
    /// @details Executes fitted input transforms in forward order during apply/apply_into and in reverse order during
    /// inverse_apply/inverse_apply_into.
    class FittedPipeline final : public FittedInputTransform
    {
    public:
        /// @brief Constructs an empty fitted input pipeline (equivalent to identity).
        /// @details Behaves identically to FittedIdentityInput on apply and inverse_apply calls.
        FittedPipeline() noexcept = default;

        /// @brief Constructs a fitted input pipeline from an ordered list of fitted steps.
        /// @details Takes ownership of the provided step sequence in forward application order.
        /// @param steps Ordered vector of fitted input transform steps.
        explicit FittedPipeline(std::vector<std::unique_ptr<FittedInputTransform>> steps) noexcept;

        /// @brief Deep-copies the fitted input pipeline.
        /// @details Clones each non-null step preserving forward order.
        /// @param other Source FittedPipeline to copy.
        FittedPipeline(const FittedPipeline& other);

        /// @brief Deep-copy assigns the fitted input pipeline.
        /// @details Replaces owned steps with deep clones of other's steps, guarding self-assignment.
        /// @param other Source FittedPipeline to copy from.
        /// @return Reference to *this.
        FittedPipeline& operator=(const FittedPipeline& other);

        /// @brief Moves the fitted input pipeline.
        /// @details Transfers step ownership without cloning.
        FittedPipeline(FittedPipeline&&) noexcept = default;

        /// @brief Move-assigns the fitted input pipeline.
        /// @details Transfers step ownership without cloning.
        /// @return Reference to *this.
        FittedPipeline& operator=(FittedPipeline&&) noexcept = default;

        /// @brief Destroys the fitted input pipeline and its owned steps.
        /// @details Overrides FittedInputTransform virtual destructor.
        ~FittedPipeline() override = default;

        /// @brief Returns the number of transform steps.
        /// @details Zero indicates an identity pipeline.
        /// @return Count of pipeline steps.
        [[nodiscard]] std::size_t size() const noexcept { return steps_.size(); }

        /// @brief Returns true if the pipeline has no steps.
        /// @details Equivalent to size() == 0.
        /// @return True when the pipeline acts as identity.
        [[nodiscard]] bool empty() const noexcept { return steps_.empty(); }

        /// @brief Returns a read-only view of the fitted steps in forward application order.
        /// @details Inverse methods iterate this span in reverse.
        /// @return Read-only span of unique pointers to fitted input steps.
        [[nodiscard]] std::span<const std::unique_ptr<FittedInputTransform>> steps() const noexcept { return steps_; }

        /// @brief Applies all steps in forward order to x in place.
        /// @details Delegates to FittedIdentityInput when empty; otherwise runs step 0..k-1 sequentially in place.
        /// @param x Mutable column-major feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns.
        /// @return Empty expected on success, or GprError from validation, null step, or step execution.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When a step rejects n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<void, GprError> apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Applies all steps in forward order from x into out without allocation.
        /// @details Applies step 0 via apply_into(x, ..., out) and subsequent steps 1..k-1 in place on out.
        /// @param x Read-only column-major feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns.
        /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
        /// @return Empty expected on success, or GprError from validation, null step, or step execution.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When a step rejects n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<void, GprError> apply_into(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
        ) const override;

        /// @brief Applies all steps in reverse order to invert the transform in place.
        /// @details Delegates to FittedIdentityInput when empty; otherwise runs step k-1..0 sequentially in place.
        /// @param x Mutable column-major transformed feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns.
        /// @return Empty expected on success, or GprError from validation, null step, or step execution.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When a step rejects n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<void, GprError> inverse_apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Applies all steps in reverse order from x into out without allocation.
        /// @details Runs the final step out of place into out via inverse_apply_into and earlier steps in place on out.
        /// @param x Read-only column-major transformed feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns.
        /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
        /// @return Empty expected on success, or GprError from validation, null step, or step execution.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When a step rejects n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<void, GprError> inverse_apply_into(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
        ) const override;

        /// @brief Returns a polymorphic deep copy of this fitted input pipeline.
        /// @details Clones every step in forward order.
        /// @return Unique pointer owning a deep copy of *this.
        [[nodiscard]] std::unique_ptr<FittedInputTransform> clone() const override;

    private:
        /// @brief Ordered sequence of fitted input transforms applied forward during apply().
        std::vector<std::unique_ptr<FittedInputTransform>> steps_;
    };

    /// @brief Unfitted sequential composition of input feature transforms.
    /// @details Fits each step sequentially on the transformed output produced by preceding steps.
    class Pipeline final : public UnfittedInputTransform
    {
    public:
        /// @brief Constructs an empty input pipeline (acts as identity until steps are added).
        /// @details Fitting an empty Pipeline yields an empty FittedPipeline.
        Pipeline() noexcept = default;

        /// @brief Constructs an input pipeline from an ordered list of unfitted steps.
        /// @details Takes ownership of the provided step sequence in forward application order.
        /// @param steps Ordered vector of unfitted input transform steps.
        explicit Pipeline(std::vector<std::unique_ptr<UnfittedInputTransform>> steps) noexcept;

        /// @brief Deep-copies the unfitted input pipeline.
        /// @details Clones each non-null step preserving forward order.
        /// @param other Source Pipeline to copy.
        Pipeline(const Pipeline& other);

        /// @brief Deep-copy assigns the unfitted input pipeline.
        /// @details Replaces owned steps with deep clones of other's steps, guarding self-assignment.
        /// @param other Source Pipeline to copy from.
        /// @return Reference to *this.
        Pipeline& operator=(const Pipeline& other);

        /// @brief Moves the unfitted input pipeline.
        /// @details Transfers step ownership without cloning.
        Pipeline(Pipeline&&) noexcept = default;

        /// @brief Move-assigns the unfitted input pipeline.
        /// @details Transfers step ownership without cloning.
        /// @return Reference to *this.
        Pipeline& operator=(Pipeline&&) noexcept = default;

        /// @brief Destroys the unfitted input pipeline and its owned steps.
        /// @details Overrides UnfittedInputTransform virtual destructor.
        ~Pipeline() override = default;

        /// @brief Appends a step by cloning it onto the end of this pipeline.
        /// @details Lvalue overload supporting chained calls on a named Pipeline variable.
        /// @param step Unfitted input transform to clone and append.
        /// @return Reference to *this.
        Pipeline& then(const UnfittedInputTransform& step) &;

        /// @brief Appends a step by cloning it onto the end of this rvalue pipeline.
        /// @details Rvalue overload supporting fluent construction expressions.
        /// @param step Unfitted input transform to clone and append.
        /// @return Rvalue reference to *this.
        Pipeline&& then(const UnfittedInputTransform& step) &&;

        /// @brief Appends an owned step onto the end of this pipeline.
        /// @details Ignores null pointers rather than storing empty slots.
        /// @param step Owned unfitted input transform to append.
        /// @return Reference to *this.
        Pipeline& then(std::unique_ptr<UnfittedInputTransform> step) &;

        /// @brief Appends an owned step onto the end of this rvalue pipeline.
        /// @details Ignores null pointers rather than storing empty slots.
        /// @param step Owned unfitted input transform to append.
        /// @return Rvalue reference to *this.
        Pipeline&& then(std::unique_ptr<UnfittedInputTransform> step) &&;

        /// @brief Returns the number of transform steps.
        /// @details Zero indicates an identity pipeline.
        /// @return Count of configured pipeline steps.
        [[nodiscard]] std::size_t size() const noexcept { return steps_.size(); }

        /// @brief Returns true if the pipeline has no steps.
        /// @details Equivalent to size() == 0.
        /// @return True when zero steps are configured.
        [[nodiscard]] bool empty() const noexcept { return steps_.empty(); }

        /// @brief Returns a read-only view of the unfitted steps in forward order.
        /// @details Order matches sequential fit and forward application order.
        /// @return Read-only span of unique pointers to unfitted input steps.
        [[nodiscard]] std::span<const std::unique_ptr<UnfittedInputTransform>> steps() const noexcept { return steps_; }

        /// @brief Fits each step sequentially on the output of the preceding step.
        /// @details Copies x into a mutable working buffer and transforms it through each fitted step before fitting
        /// the next step.
        /// @param x Read-only column-major training feature matrix.
        /// @param n_rows Number of training sample rows.
        /// @param n_cols Number of feature columns.
        /// @return FittedPipeline on success, or GprError on validation failure, null step, or step fitting error.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When any step rejects n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x or an intermediate output is non-finite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<FittedPipeline, GprError> fit_typed(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const;

        /// @brief Fits each step sequentially and returns a boxed fitted pipeline.
        /// @details Delegates to fit_typed and boxes the result in std::unique_ptr<FittedInputTransform>.
        /// @param x Read-only column-major training feature matrix.
        /// @param n_rows Number of training sample rows.
        /// @param n_cols Number of feature columns.
        /// @return Boxed FittedPipeline on success, or GprError on validation failure, null step, or step fitting
        /// error.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When any step rejects n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x or an intermediate output is non-finite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<std::unique_ptr<FittedInputTransform>, GprError> fit(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Returns a polymorphic deep copy of this unfitted input pipeline.
        /// @details Clones every unfitted step in forward order.
        /// @return Unique pointer owning a deep copy of *this.
        [[nodiscard]] std::unique_ptr<UnfittedInputTransform> clone() const override;

    private:
        /// @brief Ordered sequence of unfitted input transforms fitted stagewise during fit().
        std::vector<std::unique_ptr<UnfittedInputTransform>> steps_;
    };

    /// @brief Fitted sequential composition of target transforms.
    /// @details Executes fitted target steps forward during transform() and in reverse order during inverse_transform_*
    /// calls so chained Jacobian scalings compose accurately.
    class FittedTargetPipeline final : public FittedTargetTransform
    {
    public:
        /// @brief Constructs an empty fitted target pipeline (equivalent to identity).
        /// @details Behaves identically to FittedIdentityTarget on transform and inverse calls.
        FittedTargetPipeline() noexcept = default;

        /// @brief Constructs a fitted target pipeline from an ordered list of fitted steps.
        /// @details Takes ownership of the provided step sequence in forward application order.
        /// @param steps Ordered vector of fitted target transform steps.
        explicit FittedTargetPipeline(std::vector<std::unique_ptr<FittedTargetTransform>> steps) noexcept;

        /// @brief Deep-copies the fitted target pipeline.
        /// @details Clones each non-null step preserving forward order.
        /// @param other Source FittedTargetPipeline to copy.
        FittedTargetPipeline(const FittedTargetPipeline& other);

        /// @brief Deep-copy assigns the fitted target pipeline.
        /// @details Replaces owned steps with deep clones of other's steps, guarding self-assignment.
        /// @param other Source FittedTargetPipeline to copy from.
        /// @return Reference to *this.
        FittedTargetPipeline& operator=(const FittedTargetPipeline& other);

        /// @brief Moves the fitted target pipeline.
        /// @details Transfers step ownership without cloning.
        FittedTargetPipeline(FittedTargetPipeline&&) noexcept = default;

        /// @brief Move-assigns the fitted target pipeline.
        /// @details Transfers step ownership without cloning.
        /// @return Reference to *this.
        FittedTargetPipeline& operator=(FittedTargetPipeline&&) noexcept = default;

        /// @brief Destroys the fitted target pipeline and its owned steps.
        /// @details Overrides FittedTargetTransform virtual destructor.
        ~FittedTargetPipeline() override = default;

        /// @brief Returns the number of target transform steps.
        /// @details Zero indicates an identity target pipeline.
        /// @return Count of target pipeline steps.
        [[nodiscard]] std::size_t size() const noexcept { return steps_.size(); }

        /// @brief Returns true if the target pipeline has no steps.
        /// @details Equivalent to size() == 0.
        /// @return True when the pipeline acts as identity.
        [[nodiscard]] bool empty() const noexcept { return steps_.empty(); }

        /// @brief Returns a read-only view of the fitted steps in forward application order.
        /// @details Inverse methods iterate this span in reverse.
        /// @return Read-only span of unique pointers to fitted target steps.
        [[nodiscard]] std::span<const std::unique_ptr<FittedTargetTransform>> steps() const noexcept { return steps_; }

        /// @brief Applies all steps in forward order to y in place.
        /// @details Delegates to FittedIdentityTarget when empty; otherwise runs step 0..k-1 sequentially in place.
        /// @param y Mutable span of target values updated in place.
        /// @return Empty expected on success, or GprError on empty/non-finite input or null step.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<void, GprError> transform(std::span<double> y) const override;

        /// @brief Applies all steps in forward order from y into out without allocation.
        /// @details Applies step 0 via transform_into(y, out) and subsequent steps 1..k-1 in place on out.
        /// @param y Read-only span of target values.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on invalid input, length mismatch, or null step.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != y.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<void, GprError> transform_into(
            std::span<const double> y, std::span<double> out
        ) const override;

        /// @brief Applies inverse mean transforms in reverse order in place.
        /// @details Delegates to FittedIdentityTarget when empty; otherwise runs step k-1..0 sequentially in place.
        /// @param mean Mutable span of transformed predictive means updated in place.
        /// @return Empty expected on success, or GprError on empty/non-finite input or null step.
        /// @retval GprErrorCode::EmptyInput When mean is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_mean(std::span<double> mean) const override;

        /// @brief Applies inverse mean transforms in reverse order from mean into out without allocation.
        /// @details Executes the last step out of place into out and earlier steps in reverse order in place on out.
        /// @param mean Read-only span of transformed predictive means.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on invalid input, length mismatch, or null step.
        /// @retval GprErrorCode::EmptyInput When mean is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != mean.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_mean_into(
            std::span<const double> mean, std::span<double> out
        ) const override;

        /// @brief Applies inverse variance transforms in reverse order in place.
        /// @details Composes squared derivative scalings across all steps from step k-1 down to 0.
        /// @param var Mutable span of transformed predictive variances updated in place.
        /// @return Empty expected on success, or GprError on empty/non-finite input or null step.
        /// @retval GprErrorCode::EmptyInput When var is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_variance(std::span<double> var) const override;

        /// @brief Applies inverse variance transforms in reverse order from var into out without allocation.
        /// @details Executes the last step out of place into out and earlier steps in reverse order in place on out.
        /// @param var Read-only span of transformed predictive variances.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on invalid input, length mismatch, or null step.
        /// @retval GprErrorCode::EmptyInput When var is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != var.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_variance_into(
            std::span<const double> var, std::span<double> out
        ) const override;

        /// @brief Applies inverse covariance transforms in reverse order in place.
        /// @details Composes covariance scalings from step k-1 down to 0 in place.
        /// @param cov Mutable flattened m x m covariance buffer updated in place.
        /// @return Empty expected on success, or GprError on empty/non-finite input or null step.
        /// @retval GprErrorCode::EmptyInput When cov is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of cov is NaN or infinite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_covariance(std::span<double> cov) const override;

        /// @brief Applies inverse covariance transforms in reverse order from cov into out without allocation.
        /// @details Executes the last step out of place into out and earlier steps in reverse order in place on out.
        /// @param cov Read-only flattened m x m transformed covariance buffer.
        /// @param out Preallocated mutable output buffer of matching length.
        /// @return Empty expected on success, or GprError on invalid input, length mismatch, or null step.
        /// @retval GprErrorCode::EmptyInput When cov is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != cov.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of cov is NaN or infinite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<void, GprError> inverse_transform_covariance_into(
            std::span<const double> cov, std::span<double> out
        ) const override;

        /// @brief Returns a polymorphic deep copy of this fitted target pipeline.
        /// @details Clones every fitted target step in forward order.
        /// @return Unique pointer owning a deep copy of *this.
        [[nodiscard]] std::unique_ptr<FittedTargetTransform> clone() const override;

    private:
        /// @brief Ordered sequence of fitted target transforms applied forward during transform().
        std::vector<std::unique_ptr<FittedTargetTransform>> steps_;
    };

    /// @brief Unfitted sequential composition of target transforms.
    /// @details Fits each target step sequentially on the transformed target values produced by preceding steps.
    class TargetPipeline final : public UnfittedTargetTransform
    {
    public:
        /// @brief Constructs an empty target pipeline (acts as identity until steps are added).
        /// @details Fitting an empty TargetPipeline yields an empty FittedTargetPipeline.
        TargetPipeline() noexcept = default;

        /// @brief Constructs a target pipeline from an ordered list of unfitted steps.
        /// @details Takes ownership of the provided step sequence in forward application order.
        /// @param steps Ordered vector of unfitted target transform steps.
        explicit TargetPipeline(std::vector<std::unique_ptr<UnfittedTargetTransform>> steps) noexcept;

        /// @brief Deep-copies the unfitted target pipeline.
        /// @details Clones each non-null step preserving forward order.
        /// @param other Source TargetPipeline to copy.
        TargetPipeline(const TargetPipeline& other);

        /// @brief Deep-copy assigns the unfitted target pipeline.
        /// @details Replaces owned steps with deep clones of other's steps, guarding self-assignment.
        /// @param other Source TargetPipeline to copy from.
        /// @return Reference to *this.
        TargetPipeline& operator=(const TargetPipeline& other);

        /// @brief Moves the unfitted target pipeline.
        /// @details Transfers step ownership without cloning.
        TargetPipeline(TargetPipeline&&) noexcept = default;

        /// @brief Move-assigns the unfitted target pipeline.
        /// @details Transfers step ownership without cloning.
        /// @return Reference to *this.
        TargetPipeline& operator=(TargetPipeline&&) noexcept = default;

        /// @brief Destroys the unfitted target pipeline and its owned steps.
        /// @details Overrides UnfittedTargetTransform virtual destructor.
        ~TargetPipeline() override = default;

        /// @brief Appends a step by cloning it onto the end of this target pipeline.
        /// @details Lvalue overload supporting chained calls on a named TargetPipeline variable.
        /// @param step Unfitted target transform to clone and append.
        /// @return Reference to *this.
        TargetPipeline& then(const UnfittedTargetTransform& step) &;

        /// @brief Appends a step by cloning it onto the end of this rvalue target pipeline.
        /// @details Rvalue overload supporting fluent construction expressions.
        /// @param step Unfitted target transform to clone and append.
        /// @return Rvalue reference to *this.
        TargetPipeline&& then(const UnfittedTargetTransform& step) &&;

        /// @brief Appends an owned step onto the end of this target pipeline.
        /// @details Ignores null pointers rather than storing empty slots.
        /// @param step Owned unfitted target transform to append.
        /// @return Reference to *this.
        TargetPipeline& then(std::unique_ptr<UnfittedTargetTransform> step) &;

        /// @brief Appends an owned step onto the end of this rvalue target pipeline.
        /// @details Ignores null pointers rather than storing empty slots.
        /// @param step Owned unfitted target transform to append.
        /// @return Rvalue reference to *this.
        TargetPipeline&& then(std::unique_ptr<UnfittedTargetTransform> step) &&;

        /// @brief Returns the number of target transform steps.
        /// @details Zero indicates an identity target pipeline.
        /// @return Count of configured target pipeline steps.
        [[nodiscard]] std::size_t size() const noexcept { return steps_.size(); }

        /// @brief Returns true if the target pipeline has no steps.
        /// @details Equivalent to size() == 0.
        /// @return True when zero steps are configured.
        [[nodiscard]] bool empty() const noexcept { return steps_.empty(); }

        /// @brief Returns a read-only view of the unfitted steps in forward order.
        /// @details Order matches sequential fit and forward application order.
        /// @return Read-only span of unique pointers to unfitted target steps.
        [[nodiscard]] std::span<const std::unique_ptr<UnfittedTargetTransform>> steps() const noexcept
        {
            return steps_;
        }

        /// @brief Fits each step sequentially on the output of the preceding step.
        /// @details Copies y into a mutable working buffer and transforms it through each fitted step before fitting
        /// the subsequent step.
        /// @param y Read-only span of training target values.
        /// @return FittedTargetPipeline on success, or GprError on invalid targets, null step, or step fitting failure.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of y or an intermediate output is non-finite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<FittedTargetPipeline, GprError> fit_typed(std::span<const double> y) const;

        /// @brief Fits each step sequentially and returns a boxed fitted target pipeline.
        /// @details Delegates to fit_typed and boxes the result in std::unique_ptr<FittedTargetTransform>.
        /// @param y Read-only span of training target values.
        /// @return Boxed FittedTargetPipeline on success, or GprError on invalid targets, null step, or step fitting
        /// failure.
        /// @retval GprErrorCode::EmptyInput When y is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of y or an intermediate output is non-finite.
        /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
        [[nodiscard]] std::expected<std::unique_ptr<FittedTargetTransform>, GprError> fit(
            std::span<const double> y
        ) const override;

        /// @brief Returns a polymorphic deep copy of this unfitted target pipeline.
        /// @details Clones every unfitted target step in forward order.
        /// @return Unique pointer owning a deep copy of *this.
        [[nodiscard]] std::unique_ptr<UnfittedTargetTransform> clone() const override;

    private:
        /// @brief Ordered sequence of unfitted target transforms fitted stagewise during fit().
        std::vector<std::unique_ptr<UnfittedTargetTransform>> steps_;
    };

}  // namespace cxxgpr::transform
