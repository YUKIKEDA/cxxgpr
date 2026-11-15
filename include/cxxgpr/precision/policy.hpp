// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <utility>

namespace cxxgpr {

/// Residual formula tag that builds mixed-precision residuals from the stored float system matrix.
struct PromoteStorage {
    /// Indicates that refinement reads and promotes the stored lower-precision matrix.
    static constexpr bool reads_storage = true;

    /// Returns the canonical persistence identifier for this residual formula.
    [[nodiscard]] static constexpr std::string_view name() noexcept {
        return "promote_storage";
    }

    /// Checks value equality between residual formula tags.
    [[nodiscard]] friend constexpr bool operator==(PromoteStorage,
                                                   PromoteStorage) noexcept = default;
};

/// Residual formula tag that recomputes kernel blocks in double precision during refinement.
struct ReevaluateKernel {
    /// Indicates that refinement recomputes the kernel matrix instead of reading stored float
    /// storage.
    static constexpr bool reads_storage = false;

    /// Returns the canonical persistence identifier for this residual formula.
    [[nodiscard]] static constexpr std::string_view name() noexcept {
        return "reevaluate_kernel";
    }

    /// Checks value equality between residual formula tags.
    [[nodiscard]] friend constexpr bool operator==(ReevaluateKernel,
                                                   ReevaluateKernel) noexcept = default;
};

/// Constrains types to valid mixed-precision residual formula tags.
template <typename R>
concept ResidualFormulaConcept =
    std::same_as<R, PromoteStorage> || std::same_as<R, ReevaluateKernel>;

/// Constrains floating-point scalar types supported for kernel storage and evaluation.
template <typename T>
concept KernelScalarConcept = std::same_as<T, float> || std::same_as<T, double>;

/// Enumerates serialized precision modes stored in model configuration files.
enum class PersistPrecisionKind : std::uint8_t {
    Double,
    Single,
    MixedPromote,
    MixedReevaluate,
};

/// Returns the canonical precision string ("double", "single", or "mixed") for persistence.
[[nodiscard]] constexpr std::string_view to_precision_string(PersistPrecisionKind kind) noexcept {
    switch (kind) {
    case PersistPrecisionKind::Double:
        return "double";
    case PersistPrecisionKind::Single:
        return "single";
    case PersistPrecisionKind::MixedPromote:
    case PersistPrecisionKind::MixedReevaluate:
        return "mixed";
    }
    return "double";
}

/// Returns the canonical residual formula string for persistence.
[[nodiscard]] constexpr std::string_view to_residual_string(PersistPrecisionKind kind) noexcept {
    switch (kind) {
    case PersistPrecisionKind::MixedReevaluate:
        return "reevaluate_kernel";
    case PersistPrecisionKind::Double:
    case PersistPrecisionKind::Single:
    case PersistPrecisionKind::MixedPromote:
        return "promote_storage";
    }
    return "promote_storage";
}

/// Precision policy storing factors and computing predictions in double precision.
struct DoublePrecision {
    /// Scalar type used for kernel matrices and Cholesky factors.
    using Storage = double;
    /// Scalar type used for refined prediction weights and prediction outputs.
    using Refine = double;

    /// False because factors are already computed in double precision.
    static constexpr bool refines_in_f64 = false;

    /// Returns the persistence tag corresponding to double precision.
    [[nodiscard]] static constexpr PersistPrecisionKind persist_kind() noexcept {
        return PersistPrecisionKind::Double;
    }

    /// Checks value equality between precision policy tags.
    [[nodiscard]] friend constexpr bool operator==(DoublePrecision,
                                                   DoublePrecision) noexcept = default;
};

/// Precision policy storing factors and computing predictions in single precision without
/// refinement.
struct SinglePrecision {
    /// Scalar type used for kernel matrices and Cholesky factors.
    using Storage = float;
    /// Scalar type used for prediction weights and prediction outputs.
    using Refine = float;

    /// False because single precision keeps the float solution without refinement.
    static constexpr bool refines_in_f64 = false;

    /// Returns the persistence tag corresponding to single precision.
    [[nodiscard]] static constexpr PersistPrecisionKind persist_kind() noexcept {
        return PersistPrecisionKind::Single;
    }

    /// Checks value equality between precision policy tags.
    [[nodiscard]] friend constexpr bool operator==(SinglePrecision,
                                                   SinglePrecision) noexcept = default;
};

/// Precision policy factoring in float and refining prediction weights in double precision.
template <ResidualFormulaConcept ResidualFormula = PromoteStorage> struct MixedPrecision {
    /// Scalar type used for kernel matrices and Cholesky factors during training.
    using Storage = float;
    /// Scalar type used for refined prediction weights and prediction outputs.
    using Refine = double;
    /// Residual evaluation strategy used by the iterative refinement loop.
    using Residual = ResidualFormula;

    /// True because prediction weights are iteratively refined in double precision.
    static constexpr bool refines_in_f64 = true;

    /// Returns the persistence tag corresponding to this mixed-precision configuration.
    [[nodiscard]] static constexpr PersistPrecisionKind persist_kind() noexcept {
        return ResidualFormula::reads_storage ? PersistPrecisionKind::MixedPromote
                                              : PersistPrecisionKind::MixedReevaluate;
    }

    /// Checks value equality between mixed-precision policy tags.
    [[nodiscard]] friend constexpr bool operator==(MixedPrecision,
                                                   MixedPrecision) noexcept = default;
};

/// Constrains types to the four closed precision policies supported by cxxgpr models.
template <typename P>
concept PrecisionPolicyConcept =
    std::same_as<P, DoublePrecision> || std::same_as<P, SinglePrecision> ||
    std::same_as<P, MixedPrecision<PromoteStorage>> ||
    std::same_as<P, MixedPrecision<ReevaluateKernel>>;

/// Alias concept matching model scalar policy template parameters.
template <typename P>
concept GpScalarConcept = PrecisionPolicyConcept<P>;

/// Copies storage-scalar weights into refine-scalar output storage without dynamic allocation.
template <PrecisionPolicyConcept P>
[[nodiscard]] inline std::expected<void, GprError>
copy_storage_to_refine(std::span<const typename P::Storage> src,
                       std::span<typename P::Refine> dst) {
    if (src.empty() || dst.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    if (dst.size() != src.size()) {
        return std::unexpected(
            detail::make_count_mismatch_error(dst.size(), src.size(), "refine_alpha"));
    }
    using Refine = typename P::Refine;
    const std::size_t n = std::min(src.size(), dst.size());
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<Refine>(static_cast<double>(src[i]));
    }
    return {};
}

/// Computes k_storage^T * alpha in double accumulation into preallocated refine-scalar storage.
template <PrecisionPolicyConcept P>
[[nodiscard]] inline std::expected<void, GprError>
compute_storage_means(MatrixRef<typename P::Storage> k_storage,
                      std::span<const typename P::Refine> alpha,
                      std::span<typename P::Refine> out) {
    if (k_storage.is_empty() || k_storage.data() == nullptr || alpha.empty() || out.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    if (alpha.size() != k_storage.rows()) {
        return std::unexpected(
            detail::make_count_mismatch_error(alpha.size(), k_storage.rows(), "alpha"));
    }
    if (out.size() != k_storage.cols()) {
        return std::unexpected(
            detail::make_count_mismatch_error(out.size(), k_storage.cols(), "out"));
    }

    using Refine = typename P::Refine;
    const std::size_t n_rows = std::min(k_storage.rows(), alpha.size());
    const std::size_t n_cols = std::min(k_storage.cols(), out.size());
    for (std::size_t col = 0; col < n_cols; ++col) {
        const auto column = k_storage.col(col);
        double sum = 0.0;
        for (std::size_t row = 0; row < n_rows; ++row) {
            sum += static_cast<double>(column[row]) * static_cast<double>(alpha[row]);
        }
        out[col] = static_cast<Refine>(sum);
    }
    return {};
}

} // namespace cxxgpr
