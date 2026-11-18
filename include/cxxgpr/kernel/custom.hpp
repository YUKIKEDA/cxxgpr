// SPDX-License-Identifier: Apache-2.0
/// @file custom.hpp
/// @brief Public extension interface for user-defined distance-based kernel leaves.
/// @details Declares KernelTerm<T>, the dual-precision value wrapper CustomKernel, and the convenience
/// factory make_custom_kernel so user leaves integrate with composition, optimization, and optional
/// persistence without exposing internal Eigen or JSON headers.
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

/// @brief Root namespace of the cxxgpr Gaussian Process Regression library.
namespace cxxgpr
{

    /// @brief User-defined distance-based kernel leaf interface evaluated from squared-distance matrices.
    /// @details Implementations are evaluated in scalar T (double or float) using caller-provided
    /// squared Euclidean distances dist(i, j) = ||x_i - x_j||_2^2 and optimize parameters in unconstrained
    /// log space.
    /// @tparam T Kernel evaluation scalar type (double or float).
    template <KernelScalarConcept T> class KernelTerm
    {
    public:
        /// @brief Virtual destructor for polymorphic custom kernel leaves.
        /// @details Allows safe deletion via std::unique_ptr<KernelTerm<T>>.
        virtual ~KernelTerm() = default;

        /// @brief Clones this kernel leaf instance with independent state.
        /// @details Used for value-semantic copying of CustomKernel and atomic rollback during set_params().
        /// @return Deep copy of the derived KernelTerm<T> instance.
        [[nodiscard]] virtual std::unique_ptr<KernelTerm<T>> clone() const = 0;

        /// @brief Returns the number of log-space hyperparameters owned by this leaf.
        /// @details Must match between paired double and float KernelTerm instances wrapped in CustomKernel.
        /// @return Number of hyperparameters.
        [[nodiscard]] virtual std::size_t num_params() const noexcept = 0;

        /// @brief Writes log-space hyperparameters into out.
        /// @details Output vector length must match num_params().
        /// @param out Destination span receiving unconstrained log-space parameter values.
        /// @return Empty expected on success or GprError on length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] virtual std::expected<void, GprError> get_params(std::span<double> out) const = 0;

        /// @brief Atomically updates parameters from log-space params.
        /// @details Must validate finiteness, count, and bounds before mutating observable leaf state so
        /// failed updates leave parameters unchanged.
        /// @param params Input span of unconstrained log-space parameter values.
        /// @return Empty expected on success or GprError on invalid count, non-finite value, or out-of-range bound.
        /// @retval GprErrorCode::LengthMismatch When params.size() != num_params().
        /// @retval GprErrorCode::NonFiniteInput When any log-space value is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When exp(param) lies outside the parameter's open interval.
        [[nodiscard]] virtual std::expected<void, GprError> set_params(std::span<const double> params) = 0;

        /// @brief Writes open parameter intervals in user units into out.
        /// @details Interval order matches the log-space parameter order used by get_params() and set_params().
        /// @param out Destination span receiving open Interval descriptors.
        /// @return Empty expected on success or GprError on length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] virtual std::expected<void, GprError> bounds_into(std::span<Interval> out) const = 0;

        /// @brief Evaluates the training Gram matrix from squared distances into out for triangle uplo.
        /// @details Only entries in the requested triangle uplo need to be written when out is square.
        /// @param dist Column-major n x n squared-distance matrix.
        /// @param out Column-major n x n destination Gram matrix.
        /// @param uplo Triangular or full matrix region to populate.
        /// @return Empty expected on success or GprError if inputs or shapes are invalid.
        [[nodiscard]] virtual std::expected<void, GprError> apply(
            MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo
        ) const = 0;

        /// @brief Evaluates rectangular cross-covariance from squared distances into out.
        /// @details Computes k(x1_i, x2_j) for every pair in the n1 x n2 squared-distance matrix dist.
        /// @param dist Column-major n1 x n2 squared-distance matrix between query/training point sets.
        /// @param out Column-major n1 x n2 destination cross-covariance matrix.
        /// @return Empty expected on success or GprError if inputs or shapes are invalid.
        [[nodiscard]] virtual std::expected<void, GprError> apply_cross(MatrixRef<T> dist, MatrixMut<T> out) const = 0;

        /// @brief Fills the training diagonal k(x_i, x_i) into out.
        /// @details Implementations evaluate the zero-distance kernel diagonal without requiring full points.
        /// @param out Destination span of length n receiving diagonal kernel variances.
        /// @return Empty expected on success or GprError if output span is invalid.
        [[nodiscard]] virtual std::expected<void, GprError> fill_diag(std::span<T> out) const = 0;

        /// @brief Evaluates the first derivative with respect to log parameter param_idx from squared distances.
        /// @details Computes dK / d(log theta_{param_idx}) for the requested triangle uplo.
        /// @param dist Column-major n x n squared-distance matrix.
        /// @param d_k Column-major n x n destination matrix for the first derivative.
        /// @param param_idx Zero-based index of the log-space hyperparameter.
        /// @param uplo Triangular or full matrix region to populate.
        /// @return Empty expected on success or GprError on invalid index or matrix mismatch.
        [[nodiscard]] virtual std::expected<void, GprError> grad(
            MatrixRef<T> dist, MatrixMut<T> d_k, std::size_t param_idx, Triangle uplo
        ) const = 0;

        /// @brief Evaluates the second derivative with respect to log parameters (i, j) from squared distances.
        /// @details Computes d^2 K / (d(log theta_i) d(log theta_j)) for the requested triangle uplo.
        /// @param dist Column-major n x n squared-distance matrix.
        /// @param d2_k Column-major n x n destination matrix for the second derivative.
        /// @param i Zero-based index of the first log-space hyperparameter.
        /// @param j Zero-based index of the second log-space hyperparameter.
        /// @param uplo Triangular or full matrix region to populate.
        /// @return Empty expected on success or GprError on invalid indices or matrix mismatch.
        [[nodiscard]] virtual std::expected<void, GprError> hess(
            MatrixRef<T> dist, MatrixMut<T> d2_k, std::size_t i, std::size_t j, Triangle uplo
        ) const = 0;

        /// @brief Returns the optional persistence identifier for this custom leaf.
        /// @details Non-empty identifiers enable registry lookup during persistence load; an empty string
        /// marks the leaf as non-persistable.
        /// @return Non-owning string view identifying this custom kernel type, or empty if non-persistable.
        [[nodiscard]] virtual std::string_view persist_id() const noexcept { return ""; }

        /// @brief Serializes custom leaf state as a JSON string when persistable.
        /// @details Default implementation rejects persistence with PersistErrorKind::NotPersistable.
        /// @return JSON-encoded leaf state string or GprError when persistence is unsupported.
        /// @retval GprErrorCode::PersistFailed Always returned with NotPersistable by default.
        [[nodiscard]] virtual std::expected<std::string, GprError> persist_state_json() const
        {
            // Default leaves opt out of persistence unless overridden with a persist_id and state serializer.
            return std::unexpected(
                GprError::persist_failed(
                    PersistErrorKind::NotPersistable, "custom kernel leaf does not support persistence"
                )
            );
        }
    };

    /// @brief Value-semantic wrapper holding paired double and float implementations of a KernelTerm.
    /// @details Preserves mixed-precision and single-precision compatibility across KernelSpec and
    /// CompiledKernel by synchronizing cloned double and float implementations atomically.
    class CustomKernel
    {
    public:
        /// @brief Constructs a CustomKernel from paired double and float KernelTerm implementations.
        /// @details Verifies that both pointers are non-null and agree on num_params() and persist_id().
        /// @param term_double Heap-allocated double-precision leaf implementation.
        /// @param term_float Heap-allocated float-precision leaf implementation.
        /// @return Constructed CustomKernel or GprError on mismatched or null terms.
        /// @retval GprErrorCode::InvalidConfig When either term is null or persist_id values differ.
        /// @retval GprErrorCode::LengthMismatch When double and float terms report different parameter counts.
        [[nodiscard]] static std::expected<CustomKernel, GprError> create(
            std::unique_ptr<KernelTerm<double>> term_double, std::unique_ptr<KernelTerm<float>> term_float
        );

        /// @brief Copy-constructs a CustomKernel by cloning both underlying scalar implementations.
        /// @details Produces independent double and float KernelTerm instances via clone().
        /// @param other CustomKernel instance to copy.
        CustomKernel(const CustomKernel& other);

        /// @brief Copy-assigns a CustomKernel by cloning both underlying scalar implementations.
        /// @details Guards against self-assignment and clones both scalar terms.
        /// @param other CustomKernel instance to copy.
        /// @return Reference to this CustomKernel.
        CustomKernel& operator=(const CustomKernel& other);

        /// @brief Move-constructs a CustomKernel.
        /// @details Transfers ownership of the paired double and float leaf terms.
        /// @param other CustomKernel instance to move from.
        CustomKernel(CustomKernel&& other) noexcept = default;

        /// @brief Move-assigns a CustomKernel.
        /// @details Transfers ownership of the paired double and float leaf terms.
        /// @param other CustomKernel instance to move from.
        /// @return Reference to this CustomKernel.
        CustomKernel& operator=(CustomKernel&& other) noexcept = default;

        /// @brief Destroys the CustomKernel and its owned scalar leaf terms.
        /// @details Releases both std::unique_ptr<KernelTerm<T>> instances.
        ~CustomKernel() = default;

        /// @brief Returns the number of log-space hyperparameters.
        /// @details Reads the verified parameter count from the double leaf implementation.
        /// @return Number of hyperparameters.
        [[nodiscard]] std::size_t num_params() const noexcept;

        /// @brief Writes log-space hyperparameters into out.
        /// @details Delegates to the authoritative double-precision KernelTerm implementation.
        /// @param out Destination span of length num_params().
        /// @return Empty expected on success or GprError from the underlying leaf.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Atomically updates parameters from log-space params on both scalar implementations.
        /// @details Clones both terms first and commits only if both updates succeed, preserving atomic
        /// rollback when either double or float validation fails.
        /// @param params Input span of unconstrained log-space parameters of length num_params().
        /// @return Empty expected on success or GprError if either term rejects params.
        /// @retval GprErrorCode::LengthMismatch When params.size() != num_params().
        /// @retval GprErrorCode::NonFiniteInput When any parameter is non-finite.
        /// @retval GprErrorCode::InvalidInterval When any exponentiated value falls outside bounds.
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes open parameter intervals into out.
        /// @details Delegates to the double-precision KernelTerm implementation.
        /// @param out Destination span of length num_params().
        /// @return Empty expected on success or GprError on length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Returns the persistence identifier of the wrapped leaf.
        /// @details Both double and float terms are guaranteed by create() to share this identifier.
        /// @return Persistence identifier string view, or empty if non-persistable.
        [[nodiscard]] std::string_view persist_id() const noexcept;

        /// @brief Serializes custom leaf state as JSON via the wrapped double implementation.
        /// @details Delegates to term_double_->persist_state_json().
        /// @return JSON-encoded state string or GprError if persistence is unsupported or fails.
        /// @retval GprErrorCode::PersistFailed When the leaf does not support persistence or fails to serialize.
        [[nodiscard]] std::expected<std::string, GprError> persist_state_json() const;

        /// @brief Returns a read-only reference to the typed KernelTerm<T> implementation.
        /// @details Selects either the double or float implementation at compile time.
        /// @tparam T Kernel evaluation scalar type (double or float).
        /// @return Const reference to the wrapped KernelTerm<T>.
        template <KernelScalarConcept T> [[nodiscard]] const KernelTerm<T>& term() const noexcept
        {
            // Dispatch statically between the paired double and float custom leaf terms.
            if constexpr (std::same_as<T, double>)
            {
                return *term_double_;
            }
            else
            {
                return *term_float_;
            }
        }

        /// @brief Checks value equality via persistence identifier, parameter count, and log parameters.
        /// @details Compares log-space parameter vectors bitwise so NaN or signed-zero differences are exact.
        /// @param lhs Left-hand CustomKernel operand.
        /// @param rhs Right-hand CustomKernel operand.
        /// @return True if persist_id(), num_params(), and all bitwise log parameters match.
        friend bool operator==(const CustomKernel& lhs, const CustomKernel& rhs);

    private:
        /// @brief Private constructor storing validated double and float KernelTerm pairs.
        /// @details Called only by CustomKernel::create() after verifying invariants.
        /// @param term_double Validated double-precision leaf term.
        /// @param term_float Validated float-precision leaf term.
        CustomKernel(
            std::unique_ptr<KernelTerm<double>> term_double, std::unique_ptr<KernelTerm<float>> term_float
        ) noexcept :
            term_double_(std::move(term_double)),
            term_float_(std::move(term_float))
        {
        }

        /// @brief Authoritative double-precision implementation used for parameter queries and double eval.
        std::unique_ptr<KernelTerm<double>> term_double_;
        /// @brief Paired single-precision implementation kept in sync for float and mixed-precision eval.
        std::unique_ptr<KernelTerm<float>> term_float_;
    };

    /// @brief Constructs a CustomKernel from a templated TermTemplate<T> class with identical args.
    /// @details Instantiates TermTemplate<double>(args...) and TermTemplate<float>(args...) and
    /// validates the pair through CustomKernel::create().
    /// @tparam TermTemplate Class template accepting a scalar type parameter T and deriving from KernelTerm<T>.
    /// @tparam Args Constructor argument types forwarded by const reference to both instantiations.
    /// @param args Arguments forwarded to both TermTemplate<double> and TermTemplate<float> constructors.
    /// @return Validated CustomKernel or GprError from CustomKernel::create().
    /// @retval GprErrorCode::InvalidConfig When persist_id values differ or construction yields null.
    /// @retval GprErrorCode::LengthMismatch When double and float parameter counts differ.
    template <template <typename> class TermTemplate, typename... Args>
    [[nodiscard]] std::expected<CustomKernel, GprError> make_custom_kernel(const Args&... args)
    {
        // Construct both double and float leaf instances with identical constructor arguments.
        return CustomKernel::create(
            std::make_unique<TermTemplate<double>>(args...), std::make_unique<TermTemplate<float>>(args...)
        );
    }

}  // namespace cxxgpr
