// SPDX-License-Identifier: Apache-2.0
/// @file factor_validation.hpp
/// @brief Internal forward solve for borrowed or already verified Cholesky factors.
#pragma once

#include "cxxgpr/linalg/cholesky.hpp"

/// @brief Internal helpers for predicting with an owned Cholesky factor.
namespace cxxgpr::linalg::detail
{
    /// @brief Determines whether a factor needs its public input checks.
    enum class FactorValidation
    {
        /// @brief Check caller-supplied storage.
        Check,
        /// @brief The immutable model owns a finite factor with positive diagonal.
        AlreadyValidated
    };

    /// @brief Applies a forward solve, checking the right-hand side in both modes.
    /// @tparam T Storage scalar type.
    /// @param[in] l Lower Cholesky factor.
    /// @param[in,out] rhs Right-hand sides overwritten by solutions.
    /// @param[in] validation Provenance of the factor.
    /// @return Success or the public solve validation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> solve_lower(MatrixRef<T> l, MatrixMut<T> rhs, FactorValidation validation);
}  // namespace cxxgpr::linalg::detail
