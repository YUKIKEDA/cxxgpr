// SPDX-License-Identifier: Apache-2.0
/// @file policy.hpp
/// @brief Precision policies, residual-formula tags, and storage/refine conversion helpers.
/// @details Defines DoublePrecision, SinglePrecision, and MixedPrecision policies parameterized by
/// PromoteStorage or ReevaluateKernel residual evaluation strategies, plus persistence mappings and
/// zero-allocation mixed-precision helper routines.

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

/// @brief Top-level namespace for the cxxgpr Gaussian process regression library.
namespace cxxgpr
{

    /// @brief Residual formula tag that builds mixed-precision residuals from the stored float system matrix.
    /// @details Promotes stored float Gram entries to double during iterative refinement without
    /// retaining or re-evaluating original feature coordinates for residual computation.
    struct PromoteStorage
    {
        /// @brief Indicates that refinement reads and promotes the stored lower-precision matrix.
        static constexpr bool reads_storage = true;

        /// @brief Returns the canonical persistence identifier for this residual formula.
        /// @details Serialized as "promote_storage" in model persistence metadata.
        /// @return String view "promote_storage".
        [[nodiscard]] static constexpr std::string_view name() noexcept { return "promote_storage"; }

        /// @brief Checks value equality between residual formula tags.
        /// @details Stateless tag structs always compare equal.
        /// @param[in] lhs Left-hand residual formula tag.
        /// @param[in] rhs Right-hand residual formula tag.
        /// @return Always true.
        [[nodiscard]] friend constexpr bool operator==(PromoteStorage lhs, PromoteStorage rhs) noexcept = default;
    };

    /// @brief Residual formula tag that recomputes kernel blocks in double precision during refinement.
    /// @details Re-evaluates kernel entries column-by-column in double precision for residuals and
    /// higher-precision acceptance verification.
    struct ReevaluateKernel
    {
        /// @brief Indicates that refinement recomputes the kernel matrix instead of reading stored float storage.
        static constexpr bool reads_storage = false;

        /// @brief Returns the canonical persistence identifier for this residual formula.
        /// @details Serialized as "reevaluate_kernel" in model persistence metadata.
        /// @return String view "reevaluate_kernel".
        [[nodiscard]] static constexpr std::string_view name() noexcept { return "reevaluate_kernel"; }

        /// @brief Checks value equality between residual formula tags.
        /// @details Stateless tag structs always compare equal.
        /// @param[in] lhs Left-hand residual formula tag.
        /// @param[in] rhs Right-hand residual formula tag.
        /// @return Always true.
        [[nodiscard]] friend constexpr bool operator==(ReevaluateKernel lhs, ReevaluateKernel rhs) noexcept = default;
    };

    /// @brief Constrains types to valid mixed-precision residual formula tags.
    /// @details Accepts only PromoteStorage or ReevaluateKernel.
    /// @tparam R Candidate residual formula tag type.
    template <typename R>
    concept ResidualFormulaConcept = std::same_as<R, PromoteStorage> || std::same_as<R, ReevaluateKernel>;

    /// @brief Constrains floating-point scalar types supported for kernel storage and evaluation.
    /// @details Accepts float or double.
    /// @tparam T Candidate scalar type.
    template <typename T>
    concept KernelScalarConcept = std::same_as<T, float> || std::same_as<T, double>;

    /// @brief Enumerates serialized precision modes stored in model configuration files.
    /// @details Uniquely encodes the four closed precision policies supported by cxxgpr models.
    enum class PersistPrecisionKind : std::uint8_t
    {
        /// @brief Pure double-precision storage, factorization, and prediction.
        Double,
        /// @brief Pure single-precision storage, factorization, and prediction without refinement.
        Single,
        /// @brief Float Cholesky factorization with double iterative refinement promoting stored float matrices.
        MixedPromote,
        /// @brief Float Cholesky factorization with double iterative refinement re-evaluating the kernel in double.
        MixedReevaluate,
    };

    /// @brief Returns the canonical precision string ("double", "single", or "mixed") for persistence.
    /// @details Collapses both mixed-precision residual variants to "mixed", leaving residual mode to
    /// to_residual_string().
    /// @param[in] kind Persisted precision enum value.
    /// @return Canonical precision identifier string view.
    [[nodiscard]] constexpr std::string_view to_precision_string(PersistPrecisionKind kind) noexcept
    {
        switch (kind)
        {
            case PersistPrecisionKind::Double:
                return "double";
            case PersistPrecisionKind::Single:
                return "single";
            case PersistPrecisionKind::MixedPromote:
            case PersistPrecisionKind::MixedReevaluate:
                return "mixed";
        }
        // Defensive default if an unenumerated value is cast to PersistPrecisionKind.
        return "double";
    }

    /// @brief Returns the canonical residual formula string for persistence.
    /// @details Emits "reevaluate_kernel" for MixedReevaluate and "promote_storage" otherwise.
    /// @param[in] kind Persisted precision enum value.
    /// @return Canonical residual formula identifier string view.
    [[nodiscard]] constexpr std::string_view to_residual_string(PersistPrecisionKind kind) noexcept
    {
        switch (kind)
        {
            case PersistPrecisionKind::MixedReevaluate:
                return "reevaluate_kernel";
            case PersistPrecisionKind::Double:
            case PersistPrecisionKind::Single:
            case PersistPrecisionKind::MixedPromote:
                return "promote_storage";
        }
        // Defensive default matching PersistPrecisionKind::MixedPromote.
        return "promote_storage";
    }

    /// @brief Precision policy storing factors and computing predictions in double precision.
    /// @details Uses double for both Storage and Refine types with iterative refinement disabled.
    struct DoublePrecision
    {
        /// @brief Scalar type used for kernel matrices and Cholesky factors.
        using Storage = double;
        /// @brief Scalar type used for refined prediction weights and prediction outputs.
        using Refine = double;

        /// @brief False because factors are already computed in double precision.
        static constexpr bool refines_in_f64 = false;

        /// @brief Returns the persistence tag corresponding to double precision.
        /// @details Always returns PersistPrecisionKind::Double.
        /// @return PersistPrecisionKind::Double.
        [[nodiscard]] static constexpr PersistPrecisionKind persist_kind() noexcept
        {
            return PersistPrecisionKind::Double;
        }

        /// @brief Checks value equality between precision policy tags.
        /// @details Stateless policy tags always compare equal.
        /// @param[in] lhs Left-hand double precision policy tag.
        /// @param[in] rhs Right-hand double precision policy tag.
        /// @return Always true.
        [[nodiscard]] friend constexpr bool operator==(DoublePrecision lhs, DoublePrecision rhs) noexcept = default;
    };

    /// @brief Precision policy storing factors and computing predictions in single precision without refinement.
    /// @details Uses float for both Storage and Refine types with iterative refinement disabled.
    struct SinglePrecision
    {
        /// @brief Scalar type used for kernel matrices and Cholesky factors.
        using Storage = float;
        /// @brief Scalar type used for prediction weights and prediction outputs.
        using Refine = float;

        /// @brief False because single precision keeps the float solution without refinement.
        static constexpr bool refines_in_f64 = false;

        /// @brief Returns the persistence tag corresponding to single precision.
        /// @details Always returns PersistPrecisionKind::Single.
        /// @return PersistPrecisionKind::Single.
        [[nodiscard]] static constexpr PersistPrecisionKind persist_kind() noexcept
        {
            return PersistPrecisionKind::Single;
        }

        /// @brief Checks value equality between precision policy tags.
        /// @details Stateless policy tags always compare equal.
        /// @param[in] lhs Left-hand single precision policy tag.
        /// @param[in] rhs Right-hand single precision policy tag.
        /// @return Always true.
        [[nodiscard]] friend constexpr bool operator==(SinglePrecision lhs, SinglePrecision rhs) noexcept = default;
    };

    /// @brief Precision policy factoring in float and refining prediction weights in double precision.
    /// @details Stores kernel matrices and Cholesky factors as float while iteratively refining alpha
    /// and emitting prediction means/variances as double.
    /// @tparam ResidualFormula Residual evaluation tag (PromoteStorage or ReevaluateKernel).
    template <ResidualFormulaConcept ResidualFormula = PromoteStorage> struct MixedPrecision
    {
        /// @brief Scalar type used for kernel matrices and Cholesky factors during training.
        using Storage = float;
        /// @brief Scalar type used for refined prediction weights and prediction outputs.
        using Refine = double;
        /// @brief Residual evaluation strategy used by the iterative refinement loop.
        using Residual = ResidualFormula;

        /// @brief True because prediction weights are iteratively refined in double precision.
        static constexpr bool refines_in_f64 = true;

        /// @brief Returns the persistence tag corresponding to this mixed-precision configuration.
        /// @details Selects MixedPromote when ResidualFormula::reads_storage is true, else MixedReevaluate.
        /// @return Matching PersistPrecisionKind enumerator.
        [[nodiscard]] static constexpr PersistPrecisionKind persist_kind() noexcept
        {
            return ResidualFormula::reads_storage ? PersistPrecisionKind::MixedPromote
                                                  : PersistPrecisionKind::MixedReevaluate;
        }

        /// @brief Checks value equality between mixed-precision policy tags.
        /// @details Stateless policy tags of the same instantiation always compare equal.
        /// @param[in] lhs Left-hand mixed-precision policy tag.
        /// @param[in] rhs Right-hand mixed-precision policy tag.
        /// @return Always true.
        [[nodiscard]] friend constexpr bool operator==(MixedPrecision lhs, MixedPrecision rhs) noexcept = default;
    };

    /// @brief Constrains types to the four closed precision policies supported by cxxgpr models.
    /// @details Accepts DoublePrecision, SinglePrecision, MixedPrecision<PromoteStorage>, and
    /// MixedPrecision<ReevaluateKernel>.
    /// @tparam P Candidate precision policy type.
    template <typename P>
    concept PrecisionPolicyConcept = std::same_as<P, DoublePrecision> || std::same_as<P, SinglePrecision>
        || std::same_as<P, MixedPrecision<PromoteStorage>> || std::same_as<P, MixedPrecision<ReevaluateKernel>>;

    /// @brief Alias concept matching model scalar policy template parameters.
    /// @details Equivalent to PrecisionPolicyConcept<P>.
    /// @tparam P Candidate model scalar policy type.
    template <typename P>
    concept GpScalarConcept = PrecisionPolicyConcept<P>;

    /// @brief Copies storage-scalar weights into refine-scalar output storage without dynamic allocation.
    /// @details Rejects empty spans or mismatched lengths and widens through double for consistent
    /// cross-precision conversion.
    /// @tparam P Precision policy defining Storage and Refine scalar types.
    /// @param[in] src Source span in P::Storage precision.
    /// @param[out] dst Destination span in P::Refine precision with matching length.
    /// @return Empty expected on success, or GprError on empty or mismatched spans.
    /// @retval GprErrorCode::EmptyInput When src or dst is empty.
    /// @retval GprErrorCode::LengthMismatch When dst.size() != src.size().
    template <PrecisionPolicyConcept P>
    [[nodiscard]] inline std::expected<void, GprError> copy_storage_to_refine(
        std::span<const typename P::Storage> src, std::span<typename P::Refine> dst
    )
    {
        // Reject empty inputs before comparing sizes so EmptyInput keeps priority.
        if (src.empty() || dst.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (dst.size() != src.size())
        {
            return std::unexpected(detail::make_count_mismatch_error(dst.size(), src.size(), "refine_alpha"));
        }
        using Refine = typename P::Refine;
        // Bound the loop explicitly with std::min so optimizers prove in-bounds indexing.
        const std::size_t n = std::min(src.size(), dst.size());
        for (std::size_t i = 0; i < n; ++i)
        {
            // Promote via double before casting to Refine for uniform float->double/double->double widening.
            dst[i] = static_cast<Refine>(static_cast<double>(src[i]));
        }
        return {};
    }

    /// @brief Computes k_storage^T * alpha in double accumulation into preallocated refine-scalar storage.
    /// @details Traverses k_storage column-by-column in ascending row order for deterministic dot
    /// products with zero dynamic heap allocation.
    /// @tparam P Precision policy defining Storage and Refine scalar types.
    /// @param[in] k_storage Column-major n_train x m cross-covariance matrix in P::Storage precision.
    /// @param[in] alpha Prediction weight vector of length n_train in P::Refine precision.
    /// @param[out] out Destination span of length m receiving k_storage^T * alpha in P::Refine precision.
    /// @return Empty expected on success, or GprError on invalid dimensions or empty inputs.
    /// @retval GprErrorCode::EmptyInput When k_storage, alpha, or out is empty or k_storage.data() is null.
    /// @retval GprErrorCode::LengthMismatch When alpha.size() != k_storage.rows() or out.size() != k_storage.cols().
    template <PrecisionPolicyConcept P>
    [[nodiscard]] inline std::expected<void, GprError> compute_storage_means(
        MatrixRef<typename P::Storage> k_storage,
        std::span<const typename P::Refine> alpha,
        std::span<typename P::Refine> out
    )
    {
        // Enforce EmptyInput prior to LengthMismatch checks on alpha and out spans.
        if (k_storage.is_empty() || k_storage.data() == nullptr || alpha.empty() || out.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (alpha.size() != k_storage.rows())
        {
            return std::unexpected(detail::make_count_mismatch_error(alpha.size(), k_storage.rows(), "alpha"));
        }
        if (out.size() != k_storage.cols())
        {
            return std::unexpected(detail::make_count_mismatch_error(out.size(), k_storage.cols(), "out"));
        }

        using Refine = typename P::Refine;
        // Clamp row/col bounds via std::min to make static bounds proofs straightforward.
        const std::size_t n_rows = std::min(k_storage.rows(), alpha.size());
        const std::size_t n_cols = std::min(k_storage.cols(), out.size());
        // Traverse column-major columns sequentially, accumulating each dot product in double.
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            const auto column = k_storage.col(col);
            double sum = 0.0;
            for (std::size_t row = 0; row < n_rows; ++row)
            {
                sum += static_cast<double>(column[row]) * static_cast<double>(alpha[row]);
            }
            out[col] = static_cast<Refine>(sum);
        }
        return {};
    }

}  // namespace cxxgpr
