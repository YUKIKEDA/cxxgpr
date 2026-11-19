// SPDX-License-Identifier: Apache-2.0
/// @file custom.cpp
/// @brief Implementation of CustomKernel value wrapper and dual-precision synchronization.
/// @details Validates paired double and float KernelTerm implementations, propagates parameter
/// updates atomically with clone-and-swap rollback, and compares leaves bitwise in log space.
#include "cxxgpr/kernel/custom.hpp"

#include "cxxgpr/core/checked.hpp"

#include <bit>
#include <vector>

namespace cxxgpr
{

    /// @brief Constructs a CustomKernel from paired double and float KernelTerm implementations.
    /// @details Rejects null implementations, parameter-count mismatches, and mismatched persist_id values.
    /// @param[in] term_double Heap-allocated double-precision leaf implementation.
    /// @param[in] term_float Heap-allocated float-precision leaf implementation.
    /// @return Constructed CustomKernel or GprError if the pair is inconsistent.
    /// @retval GprErrorCode::InvalidConfig When either pointer is null or persist_id values differ.
    /// @retval GprErrorCode::LengthMismatch When parameter counts of the two terms differ.
    std::expected<CustomKernel, GprError> CustomKernel::create(
        std::unique_ptr<KernelTerm<double>> term_double, std::unique_ptr<KernelTerm<float>> term_float
    )
    {
        // Enforce non-null paired terms with identical parameter count and persistence identity.
        if (term_double == nullptr || term_float == nullptr)
        {
            return std::unexpected(GprError::invalid_config("custom kernel requires both double and float terms"));
        }
        if (term_double->num_params() != term_float->num_params())
        {
            return std::unexpected(
                GprError::length_mismatch("custom kernel double and float terms must have equal parameter counts")
            );
        }
        if (term_double->persist_id() != term_float->persist_id())
        {
            return std::unexpected(
                GprError::invalid_config("custom kernel double and float terms must share the same persist_id")
            );
        }
        return CustomKernel(std::move(term_double), std::move(term_float));
    }

    /// @brief Copy-constructs a CustomKernel by cloning both underlying scalar implementations.
    /// @details Ensures value semantics across KernelSpec and CompiledKernel copies.
    /// @param[in] other Source CustomKernel to clone.
    CustomKernel::CustomKernel(const CustomKernel& other) :
        term_double_(other.term_double_->clone()),
        term_float_(other.term_float_->clone())
    {
    }

    /// @brief Copy-assigns a CustomKernel by cloning both underlying scalar implementations.
    /// @details Guards against self-assignment before replacing both owned leaf pointers.
    /// @param[in] other Source CustomKernel to clone.
    /// @return Reference to this CustomKernel.
    CustomKernel& CustomKernel::operator=(const CustomKernel& other)
    {
        if (this != &other)
        {
            term_double_ = other.term_double_->clone();
            term_float_ = other.term_float_->clone();
        }
        return *this;
    }

    /// @brief Returns the number of log-space hyperparameters.
    /// @details Queries the double-precision term, which matches the float term per create() invariant.
    /// @return Number of hyperparameters.
    std::size_t CustomKernel::num_params() const noexcept
    {
        return term_double_->num_params();
    }

    /// @brief Writes log-space hyperparameters into out.
    /// @details Delegates to the authoritative double-precision KernelTerm.
    /// @param[out] out Destination span receiving log-space parameters.
    /// @return Empty expected on success or GprError returned by the wrapped term.
    /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
    std::expected<void, GprError> CustomKernel::get_params(std::span<double> out) const
    {
        return term_double_->get_params(out);
    }

    /// @brief Atomically updates parameters from log-space params on both scalar implementations.
    /// @details Clones both double and float terms, applies set_params() to each clone, and commits
    /// both only if both succeed so neither term becomes desynchronized on partial failure.
    /// @param[in] params Input span of unconstrained log-space parameters.
    /// @return Empty expected on success or GprError if count checking or either term fails.
    /// @retval GprErrorCode::LengthMismatch When params.size() != num_params().
    /// @retval GprErrorCode::NonFiniteInput When any log-space parameter is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When any exponentiated value falls outside bounds.
    std::expected<void, GprError> CustomKernel::set_params(std::span<const double> params)
    {
        if (auto cnt = require_count(params.size(), num_params(), "params"); !cnt.has_value())
        {
            return cnt;
        }
        // Clone both terms so a failure in term_float_ never leaves term_double_ partially mutated.
        auto copy_d = term_double_->clone();
        if (auto res_d = copy_d->set_params(params); !res_d.has_value())
        {
            return res_d;
        }
        auto copy_f = term_float_->clone();
        if (auto res_f = copy_f->set_params(params); !res_f.has_value())
        {
            return res_f;
        }
        // Commit both updated clones atomically once both scalar validations pass.
        term_double_ = std::move(copy_d);
        term_float_ = std::move(copy_f);
        return {};
    }

    /// @brief Writes open parameter intervals into out.
    /// @details Delegates to the double-precision KernelTerm.
    /// @param[out] out Destination span receiving parameter intervals.
    /// @return Empty expected on success or GprError returned by the wrapped term.
    /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
    std::expected<void, GprError> CustomKernel::bounds_into(std::span<Interval> out) const
    {
        return term_double_->bounds_into(out);
    }

    /// @brief Returns the persistence identifier of the wrapped leaf.
    /// @details Delegates to the double-precision KernelTerm.
    /// @return Persistence identifier string view, or empty if non-persistable.
    std::string_view CustomKernel::persist_id() const noexcept
    {
        return term_double_->persist_id();
    }

    /// @brief Serializes custom leaf state as JSON via the wrapped double implementation.
    /// @details Delegates to term_double_->persist_state_json().
    /// @return JSON state string or GprError on unsupported or failed serialization.
    /// @retval GprErrorCode::PersistFailed When the leaf is non-persistable or serialization fails.
    std::expected<std::string, GprError> CustomKernel::persist_state_json() const
    {
        return term_double_->persist_state_json();
    }

    /// @brief Checks value equality between two CustomKernel wrappers.
    /// @details Compares parameter counts, persistence identifiers, and IEEE-754 bit representations of
    /// all log-space hyperparameters.
    /// @param[in] lhs Left-hand CustomKernel operand.
    /// @param[in] rhs Right-hand CustomKernel operand.
    /// @return True when metadata and every log-space parameter bit pattern match.
    bool operator==(const CustomKernel& lhs, const CustomKernel& rhs)
    {
        if (lhs.num_params() != rhs.num_params() || lhs.persist_id() != rhs.persist_id())
        {
            return false;
        }
        const std::size_t np = lhs.num_params();
        std::vector<double> pl(np);
        std::vector<double> pr(np);
        if (!lhs.get_params(pl).has_value() || !rhs.get_params(pr).has_value())
        {
            return false;
        }
        // Compare exact IEEE-754 bit patterns so signed zeros or special representations match strictly.
        for (std::size_t i = 0U; i < np; ++i)
        {
            if (std::bit_cast<std::uint64_t>(pl[i]) != std::bit_cast<std::uint64_t>(pr[i]))
            {
                return false;
            }
        }
        return true;
    }

}  // namespace cxxgpr
