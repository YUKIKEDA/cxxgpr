// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/interval.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <concepts>
#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace cxxgpr
{

    /// User-defined distance-based kernel leaf interface evaluated from squared-distance matrices.
    template <KernelScalarConcept T> class KernelTerm
    {
    public:
        /// Virtual destructor.
        virtual ~KernelTerm() = default;

        /// Clones this kernel leaf instance with independent state.
        [[nodiscard]] virtual std::unique_ptr<KernelTerm<T>> clone() const = 0;

        /// Returns the number of log-space hyperparameters.
        [[nodiscard]] virtual std::size_t num_params() const noexcept = 0;

        /// Writes log-space hyperparameters into out.
        [[nodiscard]] virtual std::expected<void, GprError> get_params(std::span<double> out) const = 0;

        /// Atomically updates parameters from log-space params.
        [[nodiscard]] virtual std::expected<void, GprError> set_params(std::span<const double> params) = 0;

        /// Writes open intervals in user units into out.
        [[nodiscard]] virtual std::expected<void, GprError> bounds_into(std::span<Interval> out) const = 0;

        /// Evaluates the training Gram matrix from squared distances into out for uplo.
        [[nodiscard]] virtual std::expected<void, GprError> apply(
            MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo
        ) const = 0;

        /// Evaluates rectangular cross-covariance from squared distances into out.
        [[nodiscard]] virtual std::expected<void, GprError> apply_cross(MatrixRef<T> dist, MatrixMut<T> out) const = 0;

        /// Fills the training diagonal k(x_i, x_i) into out.
        [[nodiscard]] virtual std::expected<void, GprError> fill_diag(std::span<T> out) const = 0;

        /// Evaluates the first derivative wrt log parameter param_idx from squared distances.
        [[nodiscard]] virtual std::expected<void, GprError> grad(
            MatrixRef<T> dist, MatrixMut<T> d_k, std::size_t param_idx, Triangle uplo
        ) const = 0;

        /// Evaluates the second derivative wrt log parameters (i, j) from squared distances.
        [[nodiscard]] virtual std::expected<void, GprError> hess(
            MatrixRef<T> dist, MatrixMut<T> d2_k, std::size_t i, std::size_t j, Triangle uplo
        ) const = 0;

        /// Returns the optional persistence identifier for this custom leaf (empty if non-persistable).
        [[nodiscard]] virtual std::string_view persist_id() const noexcept { return ""; }

        /// Serializes custom leaf state as a JSON string when persistable.
        [[nodiscard]] virtual std::expected<std::string, GprError> persist_state_json() const
        {
            return std::unexpected(
                GprError::persist_failed(
                    PersistErrorKind::NotPersistable, "custom kernel leaf does not support persistence"
                )
            );
        }
    };

    /// Value-semantic wrapper holding double and float implementations of a user-defined KernelTerm.
    class CustomKernel
    {
    public:
        /// Constructs a CustomKernel from paired double and float KernelTerm implementations.
        [[nodiscard]] static std::expected<CustomKernel, GprError> create(
            std::unique_ptr<KernelTerm<double>> term_double, std::unique_ptr<KernelTerm<float>> term_float
        );

        /// Copy-constructs a CustomKernel by cloning both underlying scalar implementations.
        CustomKernel(const CustomKernel& other);
        /// Copy-assigns a CustomKernel by cloning both underlying scalar implementations.
        CustomKernel& operator=(const CustomKernel& other);
        /// Move-constructs a CustomKernel.
        CustomKernel(CustomKernel&& other) noexcept = default;
        /// Move-assigns a CustomKernel.
        CustomKernel& operator=(CustomKernel&& other) noexcept = default;
        /// Destroys the CustomKernel.
        ~CustomKernel() = default;

        /// Returns the number of log-space hyperparameters.
        [[nodiscard]] std::size_t num_params() const noexcept;

        /// Writes log-space hyperparameters into out.
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// Atomically updates parameters from log-space params on both scalar implementations.
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// Writes open parameter intervals into out.
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// Returns the persistence identifier of the wrapped leaf.
        [[nodiscard]] std::string_view persist_id() const noexcept;

        /// Serializes custom leaf state as JSON via the wrapped double implementation.
        [[nodiscard]] std::expected<std::string, GprError> persist_state_json() const;

        /// Returns a read-only pointer to the typed KernelTerm<T> implementation.
        template <KernelScalarConcept T> [[nodiscard]] const KernelTerm<T>& term() const noexcept
        {
            if constexpr (std::same_as<T, double>)
            {
                return *term_double_;
            }
            else
            {
                return *term_float_;
            }
        }

        /// Checks value equality via persistence identifier, parameter count, and log parameters.
        friend bool operator==(const CustomKernel& lhs, const CustomKernel& rhs);

    private:
        CustomKernel(
            std::unique_ptr<KernelTerm<double>> term_double, std::unique_ptr<KernelTerm<float>> term_float
        ) noexcept :
            term_double_(std::move(term_double)),
            term_float_(std::move(term_float))
        {
        }

        std::unique_ptr<KernelTerm<double>> term_double_;
        std::unique_ptr<KernelTerm<float>> term_float_;
    };

    /// Helper constructing a CustomKernel from a templated Term<T> class with identical constructor
    /// args.
    template <template <typename> class TermTemplate, typename... Args>
    [[nodiscard]] std::expected<CustomKernel, GprError> make_custom_kernel(const Args&... args)
    {
        return CustomKernel::create(
            std::make_unique<TermTemplate<double>>(args...), std::make_unique<TermTemplate<float>>(args...)
        );
    }

}  // namespace cxxgpr
