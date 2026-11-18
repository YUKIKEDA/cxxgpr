// SPDX-License-Identifier: Apache-2.0
/// @file cholesky.cpp
/// @brief Implementations of in-place Cholesky factorization, jitter retries, solves, and log-determinants.
/// @details Wraps Eigen in-place LLT decomposition and triangular solvers over column-major MatrixRef/MatrixMut views,
/// with backup-restoring jitter escalation and Neumaier compensated log-determinant summation.

#include "cxxgpr/linalg/cholesky.hpp"

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/linalg/dense.hpp"

#include <Eigen/Cholesky>

#include <cmath>
#include <cstddef>
#include <vector>

#include "linalg/eigen_map.hpp"

namespace cxxgpr::linalg
{

    namespace
    {

        /// @brief Validates a square lower-triangular Cholesky factor L prior to solves or determinant queries.
        /// @details Enforces square view geometry, lower-triangle finiteness, and strictly positive diagonal entries.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param l Square read-only Cholesky factor view.
        /// @return Empty expected on success, or GprError if l is invalid or singular.
        /// @retval GprErrorCode::EmptyInput When l is empty or null.
        /// @retval GprErrorCode::SizeOverflow When col_stride * cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When col_stride < rows or l is non-square.
        /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of l is non-finite.
        /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> validate_lower_factor(MatrixRef<T> l)
        {
            // Enforce validation order: square view geometry -> lower finiteness -> positive diagonal.
            if (const auto check = detail::validate_matrix_view(l, true); !check.has_value())
            {
                return check;
            }
            if (const auto fin = detail::require_lower_finite(l); !fin.has_value())
            {
                return fin;
            }
            const std::size_t n = l.rows();
            for (std::size_t i = 0; i < n; ++i)
            {
                if (l(i, i) <= T{0})
                {
                    return std::unexpected(GprError::non_positive_definite_matrix());
                }
            }
            return {};
        }

        /// @brief Validates a Cholesky factor L and a mutable right-hand side matrix B before in-place solves.
        /// @details Checks L via validate_lower_factor, verifies rhs geometry and row count parity, and checks rhs
        /// finiteness.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param l Square read-only lower Cholesky factor view.
        /// @param rhs Mutable right-hand side matrix view.
        /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
        /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
        /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on l or rhs.
        /// @retval GprErrorCode::LengthMismatch When l is non-square, strides are invalid, or rhs.rows() != l.rows().
        /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
        /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> validate_solve_inputs(MatrixRef<T> l, MatrixMut<T> rhs)
        {
            // Validate factor first, then right-hand side geometry, row parity, and finiteness.
            if (const auto l_check = validate_lower_factor(l); !l_check.has_value())
            {
                return l_check;
            }
            if (const auto rhs_check = detail::validate_matrix_view(rhs, false); !rhs_check.has_value())
            {
                return rhs_check;
            }
            if (rhs.rows() != l.rows())
            {
                return std::unexpected(
                    GprError::length_mismatch("right-hand side row count must match factor dimension")
                );
            }
            if (const auto fin = detail::require_matrix_finite(detail::as_ref(rhs)); !fin.has_value())
            {
                return fin;
            }
            return {};
        }

        /// @brief Executes Eigen::LLT strictly in place on the lower triangle of a validated matrix view.
        /// @details Uses Eigen::Ref over the strided map so Eigen decomposes directly in caller storage without
        /// copying, then verifies Eigen::Success and positive finite diagonal pivots.
        /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
        /// @param a Square mutable matrix view containing A's lower triangle on entry.
        /// @param stage Factorization stage recorded in CholeskyFailed errors.
        /// @return Empty expected on success, or GprErrorCode::CholeskyFailed on breakdown.
        /// @retval GprErrorCode::CholeskyFailed When Eigen::LLT fails or produces a non-positive or non-finite pivot.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> run_eigen_llt_in_place(MatrixMut<T> a, CholeskyStage stage)
        {
            // Wrap caller storage via Eigen::Ref so Eigen::LLT overwrites the lower triangle in place.
            auto map = detail::to_eigen_map(a);
            using RefType = Eigen::Ref<detail::ColMajorDynamic<T>, 0, Eigen::OuterStride<>>;
            Eigen::LLT<RefType, Eigen::Lower> llt(map);
            const std::size_t n = a.rows();
            if (llt.info() != Eigen::Success)
            {
                return std::unexpected(GprError::cholesky_failed(0.0, n, stage));
            }
            // Guard against numerical underflow or silent NaNs on the diagonal after decomposition.
            for (std::size_t i = 0; i < n; ++i)
            {
                const T diag = a(i, i);
                if (!std::isfinite(diag) || diag <= T{0})
                {
                    return std::unexpected(GprError::cholesky_failed(0.0, n, stage));
                }
            }
            return {};
        }

    }  // namespace

    /// @brief Factors a symmetric positive-definite matrix A in place into its lower Cholesky factor L.
    /// @details Validates square matrix geometry and lower-triangle finiteness before invoking Eigen in-place LLT.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param a Square mutable matrix view containing the lower triangle of A on entry and L on success.
    /// @param stage Factorization call site recorded in CholeskyFailed diagnostics.
    /// @return Empty expected on success, or GprError on invalid input or non-positive-definite matrix.
    /// @retval GprErrorCode::EmptyInput When a is empty or null.
    /// @retval GprErrorCode::SizeOverflow When col_stride * cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When col_stride < rows or a is not square.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of a is non-finite.
    /// @retval GprErrorCode::CholeskyFailed When factorization fails or yields non-positive/non-finite pivots.
    template <KernelScalarConcept T>
    std::expected<void, GprError> cholesky_lower_in_place(MatrixMut<T> a, CholeskyStage stage)
    {
        if (const auto check = detail::validate_matrix_view(a, true); !check.has_value())
        {
            return check;
        }
        if (const auto fin = detail::require_lower_finite(detail::as_ref(a)); !fin.has_value())
        {
            return fin;
        }
        return run_eigen_llt_in_place(a, stage);
    }

    /// @brief Factors A in place trying j = 0 first and escalating via policy using a caller-provided backup buffer.
    /// @details Copies A's lower triangle to backup before j = 0; on each jittered retry, restores A's lower triangle
    /// from backup before adding j to the diagonal so failed partial factorizations never accumulate.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param a Square mutable matrix view factored in place into L.
    /// @param backup Preallocated n x n square mutable matrix view used to preserve the original lower triangle.
    /// @param policy Diagonal jitter retry policy evaluated if j = 0 fails.
    /// @param stage Factorization call site recorded in CholeskyFailed diagnostics.
    /// @return Accepted diagonal jitter (0.0 if unjittered factorization succeeded), or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When a or backup is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on a or backup.
    /// @retval GprErrorCode::LengthMismatch When stride < rows, either view is non-square, or dimensions differ.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of a is non-finite.
    /// @retval GprErrorCode::CholeskyFailed When j = 0 and all jittered retry attempts fail.
    template <KernelScalarConcept T>
    std::expected<double, GprError> cholesky_lower_with_jitter(
        MatrixMut<T> a, MatrixMut<T> backup, const JitterPolicy& policy, CholeskyStage stage
    )
    {
        // Validate target view, backup view, dimension match, and lower-triangle finiteness.
        if (const auto check = detail::validate_matrix_view(a, true); !check.has_value())
        {
            return std::unexpected(check.error());
        }
        if (const auto check = detail::validate_matrix_view(backup, true); !check.has_value())
        {
            return std::unexpected(check.error());
        }
        if (a.rows() != backup.rows())
        {
            return std::unexpected(GprError::length_mismatch("backup matrix dimension must match target matrix"));
        }
        if (const auto fin = detail::require_lower_finite(detail::as_ref(a)); !fin.has_value())
        {
            return std::unexpected(fin.error());
        }

        // Snapshot the pristine lower triangle before in-place LLT mutates `a`.
        const std::size_t n = a.rows();
        if (const auto cp = copy_lower_triangle(detail::as_ref(a), backup); !cp.has_value())
        {
            return std::unexpected(cp.error());
        }

        // Attempt unjittered factorization (j = 0) first; propagate non-Cholesky failures immediately.
        const auto first = run_eigen_llt_in_place(a, stage);
        if (first.has_value())
        {
            return 0.0;
        }
        if (first.error().code() != GprErrorCode::CholeskyFailed)
        {
            return std::unexpected(first.error());
        }

        // Retry with escalating jitter, restoring the uncorrupted lower triangle before each attempt.
        double last_j = 0.0;
        JitterSchedule schedule = policy.retry_schedule();
        while (const auto next_j = schedule.next())
        {
            const double j = *next_j;
            last_j = j;
            if (const auto restore = copy_lower_triangle(detail::as_ref(backup), a); !restore.has_value())
            {
                return std::unexpected(restore.error());
            }
            if (const auto add = add_to_diag(a, static_cast<T>(j)); !add.has_value())
            {
                return std::unexpected(add.error());
            }
            const auto attempt = run_eigen_llt_in_place(a, stage);
            if (attempt.has_value())
            {
                return j;
            }
            if (attempt.error().code() != GprErrorCode::CholeskyFailed)
            {
                return std::unexpected(attempt.error());
            }
        }

        return std::unexpected(GprError::cholesky_failed(last_j, n, stage));
    }

    /// @brief Factors A in place trying j = 0 first and escalating via policy with an internal backup buffer.
    /// @details Allocates an n x n temporary backup buffer after validating a and delegates to the backup overload.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param a Square mutable matrix view factored in place into L.
    /// @param policy Diagonal jitter retry policy evaluated if j = 0 fails.
    /// @param stage Factorization call site recorded in CholeskyFailed diagnostics.
    /// @return Accepted diagonal jitter (0.0 if unjittered factorization succeeded), or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When a is empty or null.
    /// @retval GprErrorCode::SizeOverflow When col_stride * cols or rows * cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When col_stride < rows or a is not square.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of a is non-finite.
    /// @retval GprErrorCode::CholeskyFailed When j = 0 and all jittered retry attempts fail.
    template <KernelScalarConcept T>
    std::expected<double, GprError> cholesky_lower_with_jitter(
        MatrixMut<T> a, const JitterPolicy& policy, CholeskyStage stage
    )
    {
        if (const auto check = detail::validate_matrix_view(a, true); !check.has_value())
        {
            return std::unexpected(check.error());
        }
        const auto len = column_major_len(a.rows(), a.cols());
        if (!len.has_value())
        {
            return std::unexpected(len.error());
        }
        // Allocate a packed n x n backup buffer once and reuse the zero-allocation retry loop.
        std::vector<T> backup_storage(*len);
        MatrixMut<T> backup(backup_storage.data(), a.rows(), a.cols());
        return cholesky_lower_with_jitter(a, backup, policy, stage);
    }

    /// @brief Solves L * X = B in place for a lower-triangular Cholesky factor L.
    /// @details Validates factor and right-hand side views, then performs lower-triangular forward substitution via
    /// Eigen.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param rhs Mutable matrix view holding B on entry and X on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on l or rhs.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, strides are invalid, or rhs.rows() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T>
    std::expected<void, GprError> solve_lower_in_place(MatrixRef<T> l, MatrixMut<T> rhs)
    {
        if (const auto check = validate_solve_inputs(l, rhs); !check.has_value())
        {
            return check;
        }
        const auto l_map = detail::to_eigen_map(l);
        auto rhs_map = detail::to_eigen_map(rhs);
        l_map.template triangularView<Eigen::Lower>().solveInPlace(rhs_map);
        return {};
    }

    /// @brief Solves L * x = b in place for a lower-triangular Cholesky factor L and vector b.
    /// @details Guards empty spans before forwarding rhs as an n x 1 MatrixMut view.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param rhs Mutable vector span holding b on entry and x on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, stride is invalid, or rhs.size() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T>
    std::expected<void, GprError> solve_lower_in_place(MatrixRef<T> l, std::span<T> rhs)
    {
        if (rhs.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        return solve_lower_in_place(l, MatrixMut<T>(rhs.data(), rhs.size(), 1));
    }

    /// @brief Solves L^T * X = B in place for a lower-triangular Cholesky factor L.
    /// @details Interprets l.transpose() as an upper-triangular factor so strictly upper entries of l are never read.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param rhs Mutable matrix view holding B on entry and X on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on l or rhs.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, strides are invalid, or rhs.rows() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T>
    std::expected<void, GprError> solve_lower_transpose_in_place(MatrixRef<T> l, MatrixMut<T> rhs)
    {
        if (const auto check = validate_solve_inputs(l, rhs); !check.has_value())
        {
            return check;
        }
        const auto l_map = detail::to_eigen_map(l);
        auto rhs_map = detail::to_eigen_map(rhs);
        l_map.transpose().template triangularView<Eigen::Upper>().solveInPlace(rhs_map);
        return {};
    }

    /// @brief Solves L^T * x = b in place for a lower-triangular Cholesky factor L and vector b.
    /// @details Guards empty spans before forwarding rhs as an n x 1 MatrixMut view.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param rhs Mutable vector span holding b on entry and x on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, stride is invalid, or rhs.size() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T>
    std::expected<void, GprError> solve_lower_transpose_in_place(MatrixRef<T> l, std::span<T> rhs)
    {
        if (rhs.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        return solve_lower_transpose_in_place(l, MatrixMut<T>(rhs.data(), rhs.size(), 1));
    }

    /// @brief Solves (L * L^T) * X = B in place for a lower-triangular Cholesky factor L.
    /// @details Performs forward substitution L * Y = B followed by backward substitution L^T * X = Y in place.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param rhs Mutable matrix view holding B on entry and X on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on l or rhs.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, strides are invalid, or rhs.rows() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T> std::expected<void, GprError> solve_llt_in_place(MatrixRef<T> l, MatrixMut<T> rhs)
    {
        if (const auto check = validate_solve_inputs(l, rhs); !check.has_value())
        {
            return check;
        }
        // Perform two-stage triangular solve: L * Y = B, then L^T * X = Y.
        const auto l_map = detail::to_eigen_map(l);
        auto rhs_map = detail::to_eigen_map(rhs);
        l_map.template triangularView<Eigen::Lower>().solveInPlace(rhs_map);
        l_map.transpose().template triangularView<Eigen::Upper>().solveInPlace(rhs_map);
        return {};
    }

    /// @brief Solves (L * L^T) * x = b in place for a lower-triangular Cholesky factor L and vector b.
    /// @details Guards empty spans before forwarding rhs as an n x 1 MatrixMut view.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param rhs Mutable vector span holding b on entry and x on success.
    /// @return Empty expected on success, or GprError on invalid factor or right-hand side.
    /// @retval GprErrorCode::EmptyInput When l or rhs is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square, stride is invalid, or rhs.size() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower entry of l or any entry of rhs is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T> std::expected<void, GprError> solve_llt_in_place(MatrixRef<T> l, std::span<T> rhs)
    {
        if (rhs.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        return solve_llt_in_place(l, MatrixMut<T>(rhs.data(), rhs.size(), 1));
    }

    /// @brief Computes diag((L * L^T)^-1) using a caller-provided n x n scratch matrix.
    /// @details Initializes scratch_n_by_n to I_n, solves L * X = I_n in place to obtain L^-1 in the lower triangle,
    /// and sums squared column norms in double precision.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @param out_diag Preallocated output span of length l.rows() receiving the inverse diagonal entries.
    /// @param scratch_n_by_n Caller-provided n x n mutable workspace overwritten during computation.
    /// @return Empty expected on success, or GprError on invalid factor, output length, or scratch buffer.
    /// @retval GprErrorCode::EmptyInput When l, out_diag, or scratch_n_by_n is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on l or scratch_n_by_n.
    /// @retval GprErrorCode::LengthMismatch When l or scratch_n_by_n is non-square, strides are invalid,
    /// out_diag.size() != l.rows(), or scratch_n_by_n.rows() != l.rows().
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of l is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T>
    std::expected<void, GprError> inv_diag_from_chol_l(
        MatrixRef<T> l, std::span<T> out_diag, MatrixMut<T> scratch_n_by_n
    )
    {
        // Validate factor, output span length, and square scratch matrix dimensions.
        if (const auto l_check = validate_lower_factor(l); !l_check.has_value())
        {
            return l_check;
        }
        if (out_diag.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (const auto cnt = require_count(out_diag.size(), l.rows(), "inv_diag output length"); !cnt.has_value())
        {
            return cnt;
        }
        if (const auto sc_check = detail::validate_matrix_view(scratch_n_by_n, true); !sc_check.has_value())
        {
            return sc_check;
        }
        if (scratch_n_by_n.rows() != l.rows())
        {
            return std::unexpected(GprError::length_mismatch("scratch matrix dimension must match factor dimension"));
        }

        // Seed scratch with I_n, then solve L * X = I_n in place so scratch holds L^-1 (lower triangular).
        const std::size_t n = l.rows();
        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = 0; row < n; ++row)
            {
                scratch_n_by_n(row, col) = (row == col) ? T{1} : T{0};
            }
        }

        const auto l_map = detail::to_eigen_map(l);
        auto scratch_map = detail::to_eigen_map(scratch_n_by_n);
        l_map.template triangularView<Eigen::Lower>().solveInPlace(scratch_map);

        // Since (L L^T)^-1 = (L^-1)^T L^-1, the col-th diagonal equals ||L^-1 e_col||_2^2.
        for (std::size_t col = 0; col < n; ++col)
        {
            double sum_sq = 0.0;
            for (std::size_t row = col; row < n; ++row)
            {
                const auto val = static_cast<double>(scratch_n_by_n(row, col));
                sum_sq += val * val;
            }
            out_diag[col] = static_cast<T>(sum_sq);
        }
        return {};
    }

    /// @brief Computes log|A| = sum_{i=0}^{n-1} 2 * log(L_ii) using Neumaier compensated summation.
    /// @details Validates L and sums 2 * log(L_ii) in double precision using Neumaier error compensation to mitigate
    /// rounding loss across varying magnitudes.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @return Log-determinant of A = L * L^T in double precision, or GprError on invalid factor.
    /// @retval GprErrorCode::EmptyInput When l is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square or has invalid stride.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of l is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T> std::expected<double, GprError> log_det_from_l(MatrixRef<T> l)
    {
        if (const auto check = validate_lower_factor(l); !check.has_value())
        {
            return std::unexpected(check.error());
        }
        // Accumulate 2 * log(L_ii) with Neumaier compensation, ordering subtraction by larger magnitude.
        const std::size_t n = l.rows();
        double sum = 0.0;
        double comp = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double term = 2.0 * std::log(static_cast<double>(l(i, i)));
            const double t = sum + term;
            if (std::abs(sum) >= std::abs(term))
            {
                comp += (sum - t) + term;
            }
            else
            {
                comp += (term - t) + sum;
            }
            sum = t;
        }
        return sum + comp;
    }

    /// @brief Computes 0.5 * log|A| from a lower Cholesky factor L using Neumaier compensated summation.
    /// @details Halves the result of log_det_from_l(l) after propagating any validation failure.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param l Square read-only lower Cholesky factor matrix view with strictly positive finite diagonal.
    /// @return Half log-determinant of A = L * L^T in double precision, or GprError on invalid factor.
    /// @retval GprErrorCode::EmptyInput When l is empty or null.
    /// @retval GprErrorCode::SizeOverflow When l.col_stride() * l.cols() overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When l is non-square or has invalid stride.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of l is non-finite.
    /// @retval GprErrorCode::NonPositiveDefiniteMatrix When any diagonal entry of l is <= 0.
    template <KernelScalarConcept T> std::expected<double, GprError> half_log_det_from_l(MatrixRef<T> l)
    {
        const auto ld = log_det_from_l(l);
        if (!ld.has_value())
        {
            return std::unexpected(ld.error());
        }
        return 0.5 * (*ld);
    }

    template std::expected<void, GprError> cholesky_lower_in_place<float>(MatrixMut<float>, CholeskyStage);
    template std::expected<void, GprError> cholesky_lower_in_place<double>(MatrixMut<double>, CholeskyStage);

    template std::expected<double, GprError> cholesky_lower_with_jitter<float>(
        MatrixMut<float>, MatrixMut<float>, const JitterPolicy&, CholeskyStage
    );
    template std::expected<double, GprError> cholesky_lower_with_jitter<double>(
        MatrixMut<double>, MatrixMut<double>, const JitterPolicy&, CholeskyStage
    );

    template std::expected<double, GprError> cholesky_lower_with_jitter<float>(
        MatrixMut<float>, const JitterPolicy&, CholeskyStage
    );
    template std::expected<double, GprError> cholesky_lower_with_jitter<double>(
        MatrixMut<double>, const JitterPolicy&, CholeskyStage
    );

    template std::expected<void, GprError> solve_lower_in_place<float>(MatrixRef<float>, MatrixMut<float>);
    template std::expected<void, GprError> solve_lower_in_place<double>(MatrixRef<double>, MatrixMut<double>);

    template std::expected<void, GprError> solve_lower_in_place<float>(MatrixRef<float>, std::span<float>);
    template std::expected<void, GprError> solve_lower_in_place<double>(MatrixRef<double>, std::span<double>);

    template std::expected<void, GprError> solve_lower_transpose_in_place<float>(MatrixRef<float>, MatrixMut<float>);
    template std::expected<void, GprError> solve_lower_transpose_in_place<double>(MatrixRef<double>, MatrixMut<double>);

    template std::expected<void, GprError> solve_lower_transpose_in_place<float>(MatrixRef<float>, std::span<float>);
    template std::expected<void, GprError> solve_lower_transpose_in_place<double>(MatrixRef<double>, std::span<double>);

    template std::expected<void, GprError> solve_llt_in_place<float>(MatrixRef<float>, MatrixMut<float>);
    template std::expected<void, GprError> solve_llt_in_place<double>(MatrixRef<double>, MatrixMut<double>);

    template std::expected<void, GprError> solve_llt_in_place<float>(MatrixRef<float>, std::span<float>);
    template std::expected<void, GprError> solve_llt_in_place<double>(MatrixRef<double>, std::span<double>);

    template std::expected<void, GprError> inv_diag_from_chol_l<float>(
        MatrixRef<float>, std::span<float>, MatrixMut<float>
    );
    template std::expected<void, GprError> inv_diag_from_chol_l<double>(
        MatrixRef<double>, std::span<double>, MatrixMut<double>
    );

    template std::expected<double, GprError> log_det_from_l<float>(MatrixRef<float>);
    template std::expected<double, GprError> log_det_from_l<double>(MatrixRef<double>);

    template std::expected<double, GprError> half_log_det_from_l<float>(MatrixRef<float>);
    template std::expected<double, GprError> half_log_det_from_l<double>(MatrixRef<double>);

}  // namespace cxxgpr::linalg
