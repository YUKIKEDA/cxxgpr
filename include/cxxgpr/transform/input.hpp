// SPDX-License-Identifier: Apache-2.0
/// @file input.hpp
/// @brief Input feature transform interfaces and built-in preprocessing maps.
/// @details Defines FittedInputTransform and UnfittedInputTransform base classes alongside identity, per-column
/// standardization, and per-column min-max scaling transforms operating on tightly packed column-major feature
/// matrices.

#pragma once

#include "cxxgpr/core/error.hpp"

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// @brief Feature and target preprocessing transforms and composable pipelines.
namespace cxxgpr::transform
{

    /// @brief Base interface for a fitted input feature transform.
    /// @details Implementations map column-major feature matrices forward into model coordinates and back into user
    /// feature coordinates, supporting both in-place and zero-allocation out-of-place execution.
    class FittedInputTransform
    {
    public:
        /// @brief Virtual destructor enabling polymorphic deletion through base pointers.
        /// @details Ensures derived fitted input transforms release owned parameter buffers cleanly.
        virtual ~FittedInputTransform() = default;

        /// @brief Applies the fitted transform to a column-major feature matrix in place.
        /// @details Input matrix x is packed column-major as col * n_rows + row.
        /// @param x Mutable column-major feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows (must be positive).
        /// @param n_cols Number of feature columns (must match fitted dimensionality when applicable).
        /// @return Empty expected on success, or GprError on invalid dimensions, packing, or values.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols disagrees with the fitted column count.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] virtual std::expected<void, GprError> apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const = 0;

        /// @brief Applies the fitted transform out of place into a preallocated output buffer.
        /// @details Default implementation validates packing and finiteness, copies x into out when non-aliasing, and
        /// delegates to apply(out, n_rows, n_cols).
        /// @param x Read-only column-major feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows (must be positive).
        /// @param n_cols Number of feature columns.
        /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
        /// @return Empty expected on success, or GprError on invalid dimensions, buffer sizes, or values.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols disagrees with the fitted column count.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] virtual std::expected<void, GprError> apply_into(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
        ) const;

        /// @brief Maps transformed features back to original coordinates in place.
        /// @details Applies the analytical inverse of apply() over column-major storage.
        /// @param x Mutable column-major feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows (must be positive).
        /// @param n_cols Number of feature columns.
        /// @return Empty expected on success, or GprError on invalid dimensions, packing, or values.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols disagrees with the fitted column count.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] virtual std::expected<void, GprError> inverse_apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const = 0;

        /// @brief Maps transformed features back to original coordinates into a preallocated buffer.
        /// @details Default implementation validates inputs, copies x into out when non-aliasing, and delegates to
        /// inverse_apply(out, n_rows, n_cols).
        /// @param x Read-only column-major feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows (must be positive).
        /// @param n_cols Number of feature columns.
        /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
        /// @return Empty expected on success, or GprError on invalid dimensions, buffer sizes, or values.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols disagrees with the fitted column count.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] virtual std::expected<void, GprError> inverse_apply_into(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
        ) const;

        /// @brief Returns a polymorphic deep copy of this fitted transform.
        /// @details Used when copying pipelines, columnwise containers, or fitted models.
        /// @return Newly allocated unique pointer owning a deep copy of *this.
        [[nodiscard]] virtual std::unique_ptr<FittedInputTransform> clone() const = 0;

        /// @brief Returns the registry identifier for a caller-defined transform, or empty for built-ins.
        /// @details Built-in transforms serialize through structured configuration rather than custom registry IDs.
        /// @return Registry identifier view, or empty string view for built-in transforms.
        [[nodiscard]] virtual std::string_view persist_id() const { return ""; }

        /// @brief Returns the JSON state string paired with persist_id.
        /// @details Custom persistable transforms override this method; the default implementation returns
        /// PersistFailed(NotPersistable).
        /// @return Serialized JSON state string on success, or GprError if persistence is unsupported.
        /// @retval GprErrorCode::PersistFailed With PersistErrorKind::NotPersistable when not overridden.
        [[nodiscard]] virtual std::expected<std::string, GprError> persist_state_json() const;
    };

    /// @brief Base interface for an unfitted input feature transform.
    /// @details Fits transform parameters from column-major training features without mutating the unfitted
    /// specification.
    class UnfittedInputTransform
    {
    public:
        /// @brief Virtual destructor enabling polymorphic deletion through base pointers.
        /// @details Ensures derived unfitted transforms release owned configuration cleanly.
        virtual ~UnfittedInputTransform() = default;

        /// @brief Estimates transform parameters from column-major training features.
        /// @details Validates input packing and finiteness before producing a boxed FittedInputTransform.
        /// @param x Read-only column-major training feature matrix of size n_rows * n_cols.
        /// @param n_rows Number of training sample rows.
        /// @param n_cols Number of feature columns.
        /// @return Boxed fitted transform on success, or GprError on invalid training data or config.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] virtual std::expected<std::unique_ptr<FittedInputTransform>, GprError> fit(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const = 0;

        /// @brief Returns a polymorphic deep copy of this unfitted transform.
        /// @details Used when copying model specifications, pipelines, or columnwise containers.
        /// @return Newly allocated unique pointer owning a deep copy of *this.
        [[nodiscard]] virtual std::unique_ptr<UnfittedInputTransform> clone() const = 0;

        /// @brief Returns the registry identifier for a caller-defined transform, or empty for built-ins.
        /// @details Built-in transforms serialize through structured configuration rather than custom registry IDs.
        /// @return Registry identifier view, or empty string view for built-in transforms.
        [[nodiscard]] virtual std::string_view persist_id() const { return ""; }

        /// @brief Returns the JSON state string paired with persist_id.
        /// @details Custom persistable transforms override this method; the default implementation returns
        /// PersistFailed(NotPersistable).
        /// @return Serialized JSON state string on success, or GprError if persistence is unsupported.
        /// @retval GprErrorCode::PersistFailed With PersistErrorKind::NotPersistable when not overridden.
        [[nodiscard]] virtual std::expected<std::string, GprError> persist_state_json() const;
    };

    /// @brief Fitted identity input transform that leaves features unchanged.
    /// @details Still validates matrix packing, bounds, and finiteness on every call so error semantics match active
    /// transforms.
    class FittedIdentityInput final : public FittedInputTransform
    {
    public:
        /// @brief Validates packing and finiteness without modifying x.
        /// @details Performs canonical matrix validation without altering any values in x.
        /// @param x Mutable column-major feature buffer validated in place.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns.
        /// @return Empty expected on success, or GprError on invalid dimensions, packing, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Validates packing and finiteness and copies x into out.
        /// @details Avoids copying when out aliases x directly.
        /// @param x Read-only column-major feature buffer.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns.
        /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
        /// @return Empty expected on success, or GprError on invalid dimensions, buffer size, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> apply_into(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
        ) const override;

        /// @brief Validates packing and finiteness without modifying x.
        /// @details Delegates directly to apply() since identity is self-inverse.
        /// @param x Mutable column-major feature buffer validated in place.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns.
        /// @return Empty expected on success, or GprError on invalid dimensions, packing, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Validates packing and finiteness and copies x into out.
        /// @details Delegates directly to apply_into() since identity is self-inverse.
        /// @param x Read-only column-major feature buffer.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns.
        /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
        /// @return Empty expected on success, or GprError on invalid dimensions, buffer size, or non-finite values.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_apply_into(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
        ) const override;

        /// @brief Returns a polymorphic deep copy of this fitted identity transform.
        /// @details Allocates a new FittedIdentityInput instance.
        /// @return Unique pointer owning a copy of *this.
        [[nodiscard]] std::unique_ptr<FittedInputTransform> clone() const override;

        /// @brief Compares two FittedIdentityInput instances for equality.
        /// @details Always returns true because FittedIdentityInput carries no state.
        /// @param other Another FittedIdentityInput instance to compare against.
        /// @return True for any two FittedIdentityInput instances.
        [[nodiscard]] bool operator==(const FittedIdentityInput& other) const noexcept = default;
    };

    /// @brief Unfitted identity input transform (default).
    /// @details Produces FittedIdentityInput after validating the training matrix.
    class IdentityInput final : public UnfittedInputTransform
    {
    public:
        /// @brief Validates packing and finiteness and returns a typed fitted identity transform.
        /// @details Does not compute per-column statistics.
        /// @param x Read-only column-major training feature buffer.
        /// @param n_rows Number of training sample rows.
        /// @param n_cols Number of feature columns.
        /// @return FittedIdentityInput on success, or GprError on invalid training matrix.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<FittedIdentityInput, GprError> fit_typed(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const;

        /// @brief Validates packing and finiteness and returns a boxed fitted identity transform.
        /// @details Delegates to fit_typed and boxes the result into std::unique_ptr<FittedInputTransform>.
        /// @param x Read-only column-major training feature buffer.
        /// @param n_rows Number of training sample rows.
        /// @param n_cols Number of feature columns.
        /// @return Boxed FittedIdentityInput on success, or GprError on invalid training matrix.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<std::unique_ptr<FittedInputTransform>, GprError> fit(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Returns a polymorphic deep copy of this unfitted identity transform.
        /// @details Allocates a new IdentityInput instance.
        /// @return Unique pointer owning a copy of *this.
        [[nodiscard]] std::unique_ptr<UnfittedInputTransform> clone() const override;

        /// @brief Compares two IdentityInput instances for equality.
        /// @details Always returns true because IdentityInput carries no state.
        /// @param other Another IdentityInput instance to compare against.
        /// @return True for any two IdentityInput instances.
        [[nodiscard]] bool operator==(const IdentityInput& other) const noexcept = default;
    };

    /// @brief Fitted per-column standardization transform storing training means and population stds.
    /// @details Applies z = (x - mean) / std per feature column and inverts via x = z * std + mean.
    class FittedStandardizeInput final : public FittedInputTransform
    {
    public:
        /// @brief Constructs a FittedStandardizeInput from validated per-column means and positive stds.
        /// @details Enforces non-emptiness, matching vector lengths, finiteness, and strictly positive scales.
        /// @param mean Per-column mean offsets.
        /// @param std Per-column positive scale factors (typically population standard deviations, or 1.0 for constant
        /// columns).
        /// @return Validated FittedStandardizeInput on success, or GprError on invalid parameters.
        /// @retval GprErrorCode::EmptyInput When mean or std is empty.
        /// @retval GprErrorCode::LengthMismatch When std.size() != mean.size().
        /// @retval GprErrorCode::NonFiniteInput When any entry of mean or std is non-finite.
        /// @retval GprErrorCode::InvalidHyperparameter When any entry of std is <= 0.
        [[nodiscard]] static std::expected<FittedStandardizeInput, GprError> from_parts(
            std::vector<double> mean, std::vector<double> std
        );

        /// @brief Returns per-column training means.
        /// @details Length equals the fitted feature column count.
        /// @return Read-only span of per-column mean offsets.
        [[nodiscard]] std::span<const double> mean() const noexcept { return mean_; }

        /// @brief Returns per-column training scales (population std, or 1.0 for constant columns).
        /// @details Every entry is finite and strictly positive.
        /// @return Read-only span of per-column scale divisors.
        [[nodiscard]] std::span<const double> std() const noexcept { return std_; }

        /// @brief Standardizes features in place: (x - mean) / std.
        /// @details Operates independently on each column slice of length n_rows.
        /// @param x Mutable column-major feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns (must match mean().size()).
        /// @return Empty expected on success, or GprError on validation failure.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != mean().size().
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Standardizes features into out: (x - mean) / std.
        /// @details Writes directly to out without heap allocation.
        /// @param x Read-only column-major feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns (must match mean().size()).
        /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
        /// @return Empty expected on success, or GprError on validation failure.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != mean().size().
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> apply_into(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
        ) const override;

        /// @brief Restores original feature coordinates in place: x * std + mean.
        /// @details Inverts apply() column by column in place.
        /// @param x Mutable column-major standardized feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns (must match mean().size()).
        /// @return Empty expected on success, or GprError on validation failure.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != mean().size().
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Restores original feature coordinates into out: x * std + mean.
        /// @details Writes original coordinates directly into out without heap allocation.
        /// @param x Read-only column-major standardized feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns (must match mean().size()).
        /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
        /// @return Empty expected on success, or GprError on validation failure.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != mean().size().
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_apply_into(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
        ) const override;

        /// @brief Returns a polymorphic deep copy of this fitted standardize transform.
        /// @details Allocates a copy preserving per-column mean and scale vectors.
        /// @return Unique pointer owning a copy of *this.
        [[nodiscard]] std::unique_ptr<FittedInputTransform> clone() const override;

        /// @brief Compares two FittedStandardizeInput instances for value equality.
        /// @details Compares per-column mean and std vectors element-wise.
        /// @param other Another FittedStandardizeInput instance to compare against.
        /// @return True if both instances store identical mean and std vectors.
        [[nodiscard]] bool operator==(const FittedStandardizeInput& other) const noexcept = default;

    private:
        /// @brief Private constructor accepting pre-validated per-column means and scales.
        /// @details Called only by FittedStandardizeInput::from_parts after validation succeeds.
        /// @param mean Validated finite per-column mean vector.
        /// @param std Validated positive finite per-column scale vector.
        FittedStandardizeInput(std::vector<double> mean, std::vector<double> std) noexcept :
            mean_(std::move(mean)),
            std_(std::move(std))
        {
        }

        /// @brief Per-column training means subtracted during forward standardization.
        std::vector<double> mean_;
        /// @brief Per-column positive training scale factors (population std or 1.0 for constant columns).
        std::vector<double> std_;
    };

    /// @brief Unfitted per-column center-and-scale transform.
    /// @details Computes each column's empirical mean and population standard deviation (falling back to 1.0 when
    /// constant).
    class StandardizeInput final : public UnfittedInputTransform
    {
    public:
        /// @brief Estimates per-column means and population stds, using 1.0 for constant columns.
        /// @details Computes statistics using population normalization (dividing squared deviations by n_rows).
        /// @param x Read-only column-major training feature matrix.
        /// @param n_rows Number of training sample rows.
        /// @param n_cols Number of feature columns.
        /// @return FittedStandardizeInput on success, or GprError on invalid training input.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<FittedStandardizeInput, GprError> fit_typed(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const;

        /// @brief Estimates per-column means and population stds and returns a boxed fitted transform.
        /// @details Delegates to fit_typed and wraps the result in std::unique_ptr<FittedInputTransform>.
        /// @param x Read-only column-major training feature matrix.
        /// @param n_rows Number of training sample rows.
        /// @param n_cols Number of feature columns.
        /// @return Boxed FittedStandardizeInput on success, or GprError on invalid training input.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<std::unique_ptr<FittedInputTransform>, GprError> fit(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Returns a polymorphic deep copy of this unfitted standardize transform.
        /// @details Allocates a new StandardizeInput instance.
        /// @return Unique pointer owning a copy of *this.
        [[nodiscard]] std::unique_ptr<UnfittedInputTransform> clone() const override;

        /// @brief Compares two StandardizeInput instances for equality.
        /// @details Always returns true because StandardizeInput carries no configuration state.
        /// @param other Another StandardizeInput instance to compare against.
        /// @return True for any two StandardizeInput instances.
        [[nodiscard]] bool operator==(const StandardizeInput& other) const noexcept = default;
    };

    /// @brief Fitted per-column min-max transform onto a closed target interval.
    /// @details Maps [data_min, data_max] affinely onto [range_lo, range_hi], treating zero-span constant columns as
    /// unit span 1.0 to avoid division by zero.
    class FittedMinMaxInput final : public FittedInputTransform
    {
    public:
        /// @brief Constructs a FittedMinMaxInput from validated extrema and finite range [range_lo, range_hi].
        /// @details Enforces non-emptiness, matching vector lengths, finiteness, range_lo < range_hi, and data_min[c]
        /// <= data_max[c] for every column.
        /// @param data_min Per-column training minima.
        /// @param data_max Per-column training maxima.
        /// @param range_lo Finite lower endpoint of the target interval.
        /// @param range_hi Finite upper endpoint of the target interval (must exceed range_lo).
        /// @return Validated FittedMinMaxInput on success, or GprError on invalid extrema or range.
        /// @retval GprErrorCode::EmptyInput When data_min or data_max is empty.
        /// @retval GprErrorCode::LengthMismatch When data_max.size() != data_min.size().
        /// @retval GprErrorCode::NonFiniteInput When any entry of data_min or data_max is non-finite.
        /// @retval GprErrorCode::InvalidConfig When range_lo or range_hi is non-finite or range_hi <= range_lo.
        /// @retval GprErrorCode::InvalidHyperparameter When data_min[c] > data_max[c] for any column c.
        [[nodiscard]] static std::expected<FittedMinMaxInput, GprError> from_parts(
            std::vector<double> data_min, std::vector<double> data_max, double range_lo, double range_hi
        );

        /// @brief Returns the output interval [lo, hi].
        /// @details Configured at construction via MinMaxInput::create or default [0.0, 1.0].
        /// @return Pair {range_lo, range_hi}.
        [[nodiscard]] std::pair<double, double> feature_range() const noexcept { return {range_lo_, range_hi_}; }

        /// @brief Returns per-column training minima.
        /// @details Length equals the fitted feature column count.
        /// @return Read-only span of per-column training minima.
        [[nodiscard]] std::span<const double> min() const noexcept { return data_min_; }

        /// @brief Returns per-column training maxima.
        /// @details Length equals the fitted feature column count.
        /// @return Read-only span of per-column training maxima.
        [[nodiscard]] std::span<const double> max() const noexcept { return data_max_; }

        /// @brief Scales features in place onto [range_lo, range_hi].
        /// @details Applies range_lo + (range_hi - range_lo) * (x - data_min) / safe_span(data_max, data_min) per
        /// column.
        /// @param x Mutable column-major feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns (must match min().size()).
        /// @return Empty expected on success, or GprError on validation failure.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != min().size().
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Scales features into out onto [range_lo, range_hi].
        /// @details Writes scaled features directly into out without heap allocation.
        /// @param x Read-only column-major feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns (must match min().size()).
        /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
        /// @return Empty expected on success, or GprError on validation failure.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != min().size().
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> apply_into(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
        ) const override;

        /// @brief Restores original feature coordinates in place.
        /// @details Applies data_min + safe_span(data_max, data_min) * (x - range_lo) / (range_hi - range_lo) per
        /// column.
        /// @param x Mutable column-major scaled feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns (must match min().size()).
        /// @return Empty expected on success, or GprError on validation failure.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != min().size().
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Restores original feature coordinates into out.
        /// @details Writes unscaled feature values directly into out without heap allocation.
        /// @param x Read-only column-major scaled feature buffer of size n_rows * n_cols.
        /// @param n_rows Number of sample rows.
        /// @param n_cols Number of feature columns (must match min().size()).
        /// @param out Preallocated mutable output buffer of size n_rows * n_cols.
        /// @return Empty expected on success, or GprError on validation failure.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != min().size().
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<void, GprError> inverse_apply_into(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
        ) const override;

        /// @brief Returns a polymorphic deep copy of this fitted min-max transform.
        /// @details Allocates a copy preserving extrema and target output interval bounds.
        /// @return Unique pointer owning a copy of *this.
        [[nodiscard]] std::unique_ptr<FittedInputTransform> clone() const override;

        /// @brief Compares two FittedMinMaxInput instances for value equality.
        /// @details Compares extrema vectors and output interval bounds.
        /// @param other Another FittedMinMaxInput instance to compare against.
        /// @return True if both instances store identical extrema and output intervals.
        [[nodiscard]] bool operator==(const FittedMinMaxInput& other) const noexcept = default;

    private:
        /// @brief Private constructor accepting pre-validated extrema and target interval bounds.
        /// @details Called only by FittedMinMaxInput::from_parts after validation succeeds.
        /// @param data_min Validated per-column minima.
        /// @param data_max Validated per-column maxima.
        /// @param range_lo Validated lower bound of the target interval.
        /// @param range_hi Validated upper bound of the target interval.
        FittedMinMaxInput(
            std::vector<double> data_min, std::vector<double> data_max, double range_lo, double range_hi
        ) noexcept :
            data_min_(std::move(data_min)),
            data_max_(std::move(data_max)),
            range_lo_(range_lo),
            range_hi_(range_hi)
        {
        }

        /// @brief Per-column training minima observed during fitting.
        std::vector<double> data_min_;
        /// @brief Per-column training maxima observed during fitting.
        std::vector<double> data_max_;
        /// @brief Lower endpoint of the target output interval.
        double range_lo_ = 0.0;
        /// @brief Upper endpoint of the target output interval.
        double range_hi_ = 1.0;
    };

    /// @brief Unfitted per-column min-max transform onto [0, 1] or a validated [lo, hi] interval.
    /// @details Records per-column empirical extrema during fit() and stores the configured output range.
    class MinMaxInput final : public UnfittedInputTransform
    {
    public:
        /// @brief Constructs a MinMaxInput mapping onto the default interval [0, 1].
        /// @details Sets range_lo_ = 0.0 and range_hi_ = 1.0.
        MinMaxInput() noexcept = default;

        /// @brief Creates a MinMaxInput mapping onto [lo, hi], rejecting non-finite bounds or hi <= lo.
        /// @details Validates target interval bounds before constructing the transform configuration.
        /// @param lo Finite lower endpoint of the target output interval.
        /// @param hi Finite upper endpoint of the target output interval (must exceed lo).
        /// @return Validated MinMaxInput on success, or GprErrorCode::InvalidConfig on invalid bounds.
        /// @retval GprErrorCode::InvalidConfig When lo or hi is non-finite or hi <= lo.
        [[nodiscard]] static std::expected<MinMaxInput, GprError> create(double lo, double hi);

        /// @brief Returns the output interval [lo, hi].
        /// @details Defaults to {0.0, 1.0} when default-constructed.
        /// @return Pair {range_lo, range_hi}.
        [[nodiscard]] std::pair<double, double> feature_range() const noexcept { return {range_lo_, range_hi_}; }

        /// @brief Estimates per-column minima and maxima and returns a typed fitted transform.
        /// @details Validates x before computing columnwise extrema across all n_rows entries.
        /// @param x Read-only column-major training feature matrix.
        /// @param n_rows Number of training sample rows.
        /// @param n_cols Number of feature columns.
        /// @return FittedMinMaxInput on success, or GprError on invalid training input.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<FittedMinMaxInput, GprError> fit_typed(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const;

        /// @brief Estimates per-column minima and maxima and returns a boxed fitted transform.
        /// @details Delegates to fit_typed and boxes the result in std::unique_ptr<FittedInputTransform>.
        /// @param x Read-only column-major training feature matrix.
        /// @param n_rows Number of training sample rows.
        /// @param n_cols Number of feature columns.
        /// @return Boxed FittedMinMaxInput on success, or GprError on invalid training input.
        /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
        /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
        /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
        [[nodiscard]] std::expected<std::unique_ptr<FittedInputTransform>, GprError> fit(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const override;

        /// @brief Returns a polymorphic deep copy of this unfitted min-max transform.
        /// @details Allocates a new MinMaxInput preserving feature_range().
        /// @return Unique pointer owning a copy of *this.
        [[nodiscard]] std::unique_ptr<UnfittedInputTransform> clone() const override;

        /// @brief Compares two MinMaxInput instances for value equality.
        /// @details Compares configured range_lo and range_hi endpoints.
        /// @param other Another MinMaxInput instance to compare against.
        /// @return True if both instances share the same feature_range.
        [[nodiscard]] bool operator==(const MinMaxInput& other) const noexcept = default;

    private:
        /// @brief Private constructor storing validated interval endpoints [lo, hi].
        /// @details Called only by MinMaxInput::create after validate_feature_range succeeds.
        /// @param lo Validated lower bound.
        /// @param hi Validated upper bound.
        MinMaxInput(double lo, double hi) noexcept : range_lo_(lo), range_hi_(hi) {}

        /// @brief Lower endpoint of the target feature interval.
        double range_lo_ = 0.0;
        /// @brief Upper endpoint of the target feature interval.
        double range_hi_ = 1.0;
    };

}  // namespace cxxgpr::transform
