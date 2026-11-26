// SPDX-License-Identifier: Apache-2.0
/// @file compiled.cpp
/// @brief Execution-layer compiled kernel tree evaluation, differentiation, and adjoint walks.
/// @details Implements `NestedBuffers`, associative sum/product flattening inside `CompiledKernel::Impl`, cached and
/// uncached kernel Gram evaluation, analytical first and second log-hyperparameter derivatives, `keep_plan` factor Gram
/// retention, and single-pass `weighted_grads` reverse contraction walks.

#include "cxxgpr/kernel/compiled.hpp"

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/linalg/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <variant>

#include "core/noinline.hpp"
#include "kernel/exp_simd.hpp"
#include "kernel/leaf_eval.hpp"

#ifdef _OPENMP
#    include <omp.h>
#endif

/// @brief Root namespace for the cxxgpr library.
namespace cxxgpr
{

    namespace
    {

        /// @brief Returns true if evaluating `leaf` can consume a shared scalar squared-distance cache.
        /// @details Isotropic distance leaves and `CustomKernel` read scalar squared distances; coordinate-only and
        /// constant/white leaves do not.
        /// @param[in] leaf Kernel leaf variant.
        /// @return True when `leaf` consumes scalar squared distances.
        [[nodiscard]] bool leaf_reads_distances(const KernelLeaf& leaf) noexcept
        {
            return std::holds_alternative<RbfKernel>(leaf) || std::holds_alternative<MaternKernel>(leaf)
                || std::holds_alternative<PeriodicKernel>(leaf) || std::holds_alternative<RationalQuadraticKernel>(leaf)
                || std::holds_alternative<CustomKernel>(leaf);
        }

        /// @brief Returns true if evaluating `leaf` can consume packed per-dimension ARD squared differences.
        /// @details Matches `RbfArdKernel`, `MaternArdKernel`, and `RationalQuadraticArdKernel`.
        /// @param[in] leaf Kernel leaf variant.
        /// @return True when `leaf` consumes packed ARD squared differences.
        [[nodiscard]] bool leaf_needs_ard(const KernelLeaf& leaf) noexcept
        {
            return std::holds_alternative<RbfArdKernel>(leaf) || std::holds_alternative<MaternArdKernel>(leaf)
                || std::holds_alternative<RationalQuadraticArdKernel>(leaf);
        }

        /// @brief Combines two leaf or subtree `CoordMode` values across composition operators.
        /// @details `Either` acts as an identity element, matching modes preserve themselves, and conflicting
        /// `Dist` and `Points` requirements promote to `CoordMode::Mixed`.
        /// @param[in] a First child coordinate mode.
        /// @param[in] b Second child coordinate mode.
        /// @return Combined `CoordMode` for the parent node.
        [[nodiscard]] CoordMode combine_coord_mode(CoordMode a, CoordMode b) noexcept
        {
            // Either is neutral under + and *; distinct Dist and Points modes combine into Mixed.
            if (a == b)
            {
                return a;
            }
            if (a == CoordMode::Either)
            {
                return b;
            }
            if (b == CoordMode::Either)
            {
                return a;
            }
            return CoordMode::Mixed;
        }

        /// @brief Populates a full symmetric squared Euclidean distance matrix from training points `x`.
        /// @details Sets diagonal entries to zero and computes each strictly lower-triangular pair once before
        /// mirroring to the upper triangle.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] x Input point matrix of shape `(n, d)`.
        /// @param[out] dist Target square squared-distance matrix of shape `(n, n)`.
        template <KernelScalarConcept T> void fill_square_sq_dist(MatrixRef<T> x, MatrixMut<T> dist) noexcept
        {
            const std::size_t n = x.rows();
            const std::size_t d = x.cols();
            if (n == 0U || d == 0U)
            {
                return;
            }
            auto run_lower = [&](std::size_t col_begin, std::size_t col_end)
            {
                for (std::size_t col = col_begin; col < col_end; ++col)
                {
                    const std::size_t len = n - col;
                    T* col_ptr = &dist(col, col);
                    detail::set_scaled_sq_diff_slice(col_ptr, x.col(0U).data() + col, x(col, 0U), T{1}, len);
                    for (std::size_t k = 1U; k < d; ++k)
                    {
                        detail::add_scaled_sq_diff_slice(col_ptr, x.col(k).data() + col, x(col, k), T{1}, len);
                    }
                    dist(col, col) = T{0};
                }
            };
            if (n >= 256U)
            {
                linalg::parallel_for_lower_blocks(n, 0U, run_lower);
            }
            else
            {
                run_lower(0U, n);
            }
            for (std::size_t col = 0U; col < n; ++col)
            {
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    const std::size_t r1 = row;
                    const std::size_t r2 = col;
                    dist(r2, r1) = dist(r1, r2);
                }
            }
        }

        /// @brief Populates a rectangular cross squared Euclidean distance matrix between `x1` and `x2`.
        /// @details Vectorizes each column slice along feature dimensions using `set_scaled_sq_diff_slice` and
        /// `add_scaled_sq_diff_slice`, parallelizing across columns when `(x1.rows() * x2.rows()) >= 65536U`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] x1 First point matrix of shape `(n1, d)`.
        /// @param[in] x2 Second point matrix of shape `(n2, d)`.
        /// @param[out] dist Target rectangular squared-distance matrix of shape `(n1, n2)`.
        template <KernelScalarConcept T>
        void fill_cross_sq_dist(MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> dist) noexcept
        {
            const std::size_t n1 = x1.rows();
            const std::size_t n2 = x2.rows();
            const std::size_t d = x1.cols();
            if (n1 == 0U || n2 == 0U || d == 0U)
            {
                return;
            }
            const auto n2_s = static_cast<std::ptrdiff_t>(n2);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n1 * n2) >= 65536U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n2_s; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                T* col_ptr = &dist(0U, col);
                detail::set_scaled_sq_diff_slice(col_ptr, x1.col(0U).data(), x2(col, 0U), T{1}, n1);
                for (std::size_t k = 1U; k < d; ++k)
                {
                    detail::add_scaled_sq_diff_slice(col_ptr, x1.col(k).data(), x2(col, k), T{1}, n1);
                }
            }
        }

        /// @brief Validates that every entry of a column-major matrix view is finite using branchless SSE2 scans.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] mat Matrix view whose entries are inspected in column-major order.
        /// @return Empty expected when all entries are finite, or `GprErrorCode::NonFiniteInput`.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> require_matrix_finite_simd(MatrixRef<T> mat)
        {
            if (mat.is_contiguous())
            {
                if (!detail::all_finite_slice(mat.data(), mat.rows() * mat.cols()))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
                return {};
            }
            for (std::size_t col = 0; col < mat.cols(); ++col)
            {
                if (!detail::all_finite_slice(mat.col(col).data(), mat.rows()))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
            return {};
        }

        /// @brief Validates that a scratch matrix matches `out` shape and does not alias `out`.
        /// @details Returns `WorkspaceTooSmall` if dimensions differ and `InvalidConfig` on pointer aliasing.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] out Reference output matrix view whose dimensions must be matched.
        /// @param[in] scratch Mutable scratch matrix view to check.
        /// @return Empty expected on success, or workspace/aliasing error.
        /// @retval GprErrorCode::WorkspaceTooSmall If `scratch` rows or columns differ from `out`.
        /// @retval GprErrorCode::InvalidConfig If `scratch.data() == out.data()`.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError>
            require_scratch(MatrixRef<T> out, MatrixMut<T> scratch, std::string_view /*label*/)
        {
            if (scratch.rows() != out.rows() || scratch.cols() != out.cols())
            {
                return std::unexpected(GprError::workspace_too_small());
            }
            if (scratch.data() == out.data())
            {
                return std::unexpected(GprError::invalid_config("scratch buffer must not alias output"));
            }
            return {};
        }

        /// @brief Applies an element-wise binary mutation `op(dst(row, col), src(row, col))` over `uplo`.
        /// @details Uses area-balanced `linalg::parallel_for_lower_blocks` (`n_cols >= 384U`) for square `Lower`
        /// matrices and column-parallel OpenMP (`n_cols >= 384U`) for `Upper` or `Full`.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam Op Callable mutating `T& dst_elem` using `T src_elem`.
        /// @param[in,out] dst Destination matrix updated in place.
        /// @param[in] src Source matrix read over the active region.
        /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        /// @param[in] op Element-wise update functor.
        template <KernelScalarConcept T, typename Op>
        void apply_binary_op(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo, Op&& op) noexcept
        {
            const std::size_t n_rows = dst.rows();
            const std::size_t n_cols = dst.cols();
            // Restrict updates to the active triangle on square outputs and parallelize disjoint columns.
            if (uplo == Triangle::Lower && n_rows == n_cols)
            {
                auto run_lower = [&](std::size_t col_begin, std::size_t col_end)
                {
                    for (std::size_t col = col_begin; col < col_end; ++col)
                    {
                        for (std::size_t row = col; row < n_rows; ++row)
                        {
                            op(dst(row, col), src(row, col));
                        }
                    }
                };
                if (n_cols >= 384U)
                {
                    linalg::parallel_for_lower_blocks(n_cols, 0U, run_lower);
                }
                else
                {
                    run_lower(0U, n_cols);
                }
                return;
            }
            const auto cols_s = static_cast<std::ptrdiff_t>(n_cols);
            if (uplo == Triangle::Upper && n_rows == n_cols)
            {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n_cols >= 384U)
#endif
                for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s)
                {
                    const auto col = static_cast<std::size_t>(col_s);
                    for (std::size_t row = 0; row <= col; ++row)
                    {
                        op(dst(row, col), src(row, col));
                    }
                }
                return;
            }
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n_cols >= 384U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                for (std::size_t row = 0; row < n_rows; ++row)
                {
                    op(dst(row, col), src(row, col));
                }
            }
        }

        /// @brief Copies `src` into `dst` over the specified matrix region `uplo`.
        /// @details Delegates column-parallel assignment to `apply_binary_op`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[out] dst Destination matrix.
        /// @param[in] src Source matrix.
        /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        template <KernelScalarConcept T> void copy_into(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo) noexcept
        {
            apply_binary_op(dst, src, uplo, [](T& d, T s) { d = s; });
        }

        /// @brief Accumulates `dst += src` over the specified matrix region `uplo`.
        /// @details Delegates column-parallel addition to `apply_binary_op`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in,out] dst Destination matrix accumulated in place.
        /// @param[in] src Summand matrix.
        /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        template <KernelScalarConcept T> void add_into(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo) noexcept
        {
            apply_binary_op(dst, src, uplo, [](T& d, T s) { d += s; });
        }

        /// @brief Accumulates Hadamard product `dst *= src` over the specified matrix region `uplo`.
        /// @details Delegates column-parallel element-wise multiplication to `apply_binary_op`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in,out] dst Destination matrix scaled in place.
        /// @param[in] src Multiplier matrix.
        /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        template <KernelScalarConcept T> void mul_into(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo) noexcept
        {
            apply_binary_op(dst, src, uplo, [](T& d, T s) { d *= s; });
        }

        /// @brief Multiplies the lower triangle (including diagonal) of `dst` by scalar `scale`.
        /// @details Uses vectorized `scale_set_slice` per contiguous column slice and area-balanced OpenMP (`n >=
        /// 384U`).
        /// @tparam T Storage floating-point scalar type.
        /// @param[in,out] dst Square matrix whose lower triangle is scaled in place.
        /// @param[in] scale Scalar multiplier.
        template <KernelScalarConcept T> void scale_lower(MatrixMut<T> dst, T scale) noexcept
        {
            const std::size_t n = dst.rows();
            auto run_range = [&](std::size_t col_begin, std::size_t col_end)
            {
                for (std::size_t col = col_begin; col < col_end; ++col)
                {
                    T* const col_ptr = &dst(col, col);
                    detail::scale_set_slice(col_ptr, col_ptr, scale, n - col);
                }
            };
            if (n >= 384U)
            {
                linalg::parallel_for_lower_blocks(n, 0U, run_range);
            }
            else
            {
                run_range(0U, n);
            }
        }

        /// @brief Scales all active entries in `dst` over `uplo` by scalar `scale` in place.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in,out] dst Target matrix view scaled in place.
        /// @param[in] scale Scalar multiplier (returns immediately when `scale == 1`).
        /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        template <KernelScalarConcept T> void scale_region(MatrixMut<T> dst, T scale, Triangle uplo) noexcept
        {
            if (scale == T{1})
            {
                return;
            }
            const std::size_t n_rows = dst.rows();
            const std::size_t n_cols = dst.cols();
            if (uplo == Triangle::Lower && n_rows == n_cols)
            {
                scale_lower(dst, scale);
                return;
            }
            const auto cols_s = static_cast<std::ptrdiff_t>(n_cols);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n_cols >= 384U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                const std::size_t len = (uplo == Triangle::Upper && n_rows == n_cols) ? (col + 1U) : n_rows;
                T* const col_ptr = &dst(0U, col);
                detail::scale_set_slice(col_ptr, col_ptr, scale, len);
            }
        }

        /// @brief Populates all active entries in `dst` over `uplo` with scalar constant `val`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[out] dst Destination matrix view populated in place.
        /// @param[in] val Constant fill scalar.
        /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        template <KernelScalarConcept T> void fill_region(MatrixMut<T> dst, T val, Triangle uplo) noexcept
        {
            const std::size_t n_rows = dst.rows();
            const std::size_t n_cols = dst.cols();
            if (uplo == Triangle::Lower && n_rows == n_cols)
            {
                auto run_range = [&](std::size_t col_begin, std::size_t col_end)
                {
                    for (std::size_t col = col_begin; col < col_end; ++col)
                    {
                        std::fill_n(&dst(col, col), n_rows - col, val);
                    }
                };
                if (n_cols >= 384U)
                {
                    linalg::parallel_for_lower_blocks(n_cols, 0U, run_range);
                }
                else
                {
                    run_range(0U, n_cols);
                }
                return;
            }
            const auto cols_s = static_cast<std::ptrdiff_t>(n_cols);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n_cols >= 384U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                const std::size_t len = (uplo == Triangle::Upper && n_rows == n_cols) ? (col + 1U) : n_rows;
                std::fill_n(&dst(0U, col), len, val);
            }
        }

        /// @brief Computes the symmetric Frobenius inner product $\langle A, B \rangle_F$ from lower triangles.
        /// @details Evaluates each column's `diag + 2 * strictly_lower` inner product in `double` across OpenMP
        /// workers (`n >= 384U`) into `col_partials`, then reduces `col_partials` serially in ascending column index
        /// order so the floating-point result is bit-for-bit independent of thread count.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] a First symmetric matrix (only lower triangle read).
        /// @param[in] b Second symmetric matrix (only lower triangle read).
        /// @param[in,out] col_partials Pre-allocated per-column partial sum buffer of length at least `a.rows()`.
        /// @return Full symmetric Frobenius inner product in `double`.
        template <KernelScalarConcept T>
        [[nodiscard]] double lower_frobenius(MatrixRef<T> a, MatrixRef<T> b, std::span<double> col_partials) noexcept
        {
            const std::size_t n = a.rows();
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
            // Compute column-wise symmetric contributions (diag + 2 * strictly lower) in parallel.
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 384U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                const double diag = static_cast<double>(a(col, col)) * static_cast<double>(b(col, col));
                double off = 0.0;
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    off += static_cast<double>(a(row, col)) * static_cast<double>(b(row, col));
                }
                col_partials[col] = diag + (2.0 * off);
            }
            // Reduce column partials serially in ascending column order for bit-identical OpenMP determinism.
            double total = 0.0;
            for (std::size_t col = 0; col < n; ++col)
            {
                total += col_partials[col];
            }
            return total;
        }

        /// @brief Computes the sum of all entries $\sum_{i,j} W_{ij}$ of a symmetric matrix from its lower triangle.
        /// @details Used to contract against constant matrices (`ConstantKernel`). Computes column-wise partials in
        /// parallel (`n >= 384U`) and sums them serially in ascending column order for deterministic OpenMP reductions.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] w Symmetric matrix whose lower triangle is summed.
        /// @param[in,out] col_partials Pre-allocated per-column partial sum buffer of length at least `w.rows()`.
        /// @return Sum of all $n^2$ symmetric entries in `double`.
        template <KernelScalarConcept T>
        [[nodiscard]] double lower_sum(MatrixRef<T> w, std::span<double> col_partials) noexcept
        {
            const std::size_t n = w.rows();
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
            // Accumulate diag + 2 * strictly lower per column in parallel.
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 384U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                const auto diag = static_cast<double>(w(col, col));
                double off = 0.0;
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    off += static_cast<double>(w(row, col));
                }
                col_partials[col] = diag + (2.0 * off);
            }
            // Reduce column partials serially in ascending column order for bit-identical results.
            double total = 0.0;
            for (std::size_t col = 0; col < n; ++col)
            {
                total += col_partials[col];
            }
            return total;
        }

        /// @brief Computes the trace $\sum_i W_{ii}$ from the diagonal of `w`.
        /// @details Used to contract against `WhiteKernel`, whose derivative is nonzero only on the diagonal.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] w Symmetric square matrix.
        /// @return Sum of diagonal entries in `double`.
        template <KernelScalarConcept T> [[nodiscard]] double lower_diag_sum(MatrixRef<T> w) noexcept
        {
            const std::size_t n = w.rows();
            double total = 0.0;
            for (std::size_t i = 0; i < n; ++i)
            {
                total += static_cast<double>(w(i, i));
            }
            return total;
        }

    }  // namespace

    // --- NestedBuffers ---

    /// @brief Constructs `depth` contiguous intermediate matrix buffers of shape `rows x cols`.
    /// @details Delegates capacity allocation to `resize`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] depth Number of nested buffer levels to allocate.
    /// @param[in] rows Row capacity per buffer level.
    /// @param[in] cols Column capacity per buffer level.
    template <KernelScalarConcept T>
    NestedBuffers<T>::NestedBuffers(std::size_t depth, std::size_t rows, std::size_t cols)
    {
        resize(depth, rows, cols);
    }

    /// @brief Reconfigures nested buffer dimensions and grows backing storage monotonically if needed.
    /// @details Never shrinks existing capacity so repeated resize calls for smaller subproblems avoid reallocations.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] depth Number of nested buffer levels.
    /// @param[in] rows Row dimension per level.
    /// @param[in] cols Column dimension per level.
    template <KernelScalarConcept T>
    void NestedBuffers<T>::resize(std::size_t depth, std::size_t rows, std::size_t cols)
    {
        depth_ = depth;
        rows_ = rows;
        cols_ = cols;
        // Grow storage monotonically so smaller sub-problems reuse already allocated capacity.
        const std::size_t needed = depth * rows * cols;
        if (storage_.size() < needed)
        {
            storage_.resize(needed);
        }
    }

    /// @brief Returns a mutable column-major view into buffer level `index`.
    /// @details Indexes into the contiguous slab at `index * rows_ * cols_` with shape `(rows, cols)` and stride
    /// `rows`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] index Zero-based buffer level index (`< depth()`).
    /// @param[in] rows Active view row count (`<= rows()`).
    /// @param[in] cols Active view column count (`<= cols()`).
    /// @return Mutable matrix view backed by level `index`.
    template <KernelScalarConcept T>
    MatrixMut<T> NestedBuffers<T>::level(std::size_t index, std::size_t rows, std::size_t cols)
    {
        T* ptr = storage_.data() + (index * rows_ * cols_);
        return {ptr, rows, cols, rows};
    }

    /// @brief Internal flattened tree representation and evaluation engine for `CompiledKernel<T>`.
    /// @details Compiles nested `KernelSpec` sum/product trees into associatively flattened `SumTerm` and
    /// `ProductTerm` nodes, binds hyperparameters in depth-first left-to-right order, caches ARD lengthscales for
    /// zero-allocation evaluation, and implements value, gradient, Hessian, `keep_plan`, and `weighted_grads` passes.
    /// @tparam T Storage floating-point scalar type.
    template <KernelScalarConcept T> struct CompiledKernel<T>::Impl
    {
        /// @brief Flattened reference to a single compiled kernel leaf and its parameter slice.
        /// @details Maps a leaf slot into the contiguous `leaves` vector and the global parameter vector range.
        struct LeafRef
        {
            /// @brief Zero-based index into `Impl::leaves`.
            std::size_t leaf_idx = 0;
            /// @brief Starting index of this leaf's hyperparameters in the global parameter vector.
            std::size_t param_begin = 0;
            /// @brief Number of hyperparameters owned by this leaf.
            std::size_t param_count = 0;
        };

        /// @brief Forward declaration of a flattened additive composite node.
        struct SumTerm;
        /// @brief Forward declaration of a flattened multiplicative composite node.
        struct ProductTerm;
        /// @brief Variant representing a compiled kernel AST node (`LeafRef`, `SumTerm`, or `ProductTerm`).
        using Term = std::variant<LeafRef, SumTerm, ProductTerm>;

        /// @brief Associatively flattened additive composite kernel term.
        /// @details Holds non-sum summands collected in left-to-right order along with the half-open global parameter
        /// range `[param_begin, param_end)` covered by all summands.
        struct SumTerm
        {
            /// @brief Child terms to be summed element-wise.
            std::vector<Term> terms;
            /// @brief Inclusive start index of hyperparameters owned by this subtree.
            std::size_t param_begin = 0;
            /// @brief Exclusive end index of hyperparameters owned by this subtree.
            std::size_t param_end = 0;
        };

        /// @brief Associatively flattened multiplicative composite kernel term.
        /// @details Holds non-product factors collected in left-to-right order along with the half-open global
        /// parameter range `[param_begin, param_end)` covered by all factors.
        struct ProductTerm
        {
            /// @brief Child terms to be multiplied element-wise.
            std::vector<Term> terms;
            /// @brief Inclusive start index of hyperparameters owned by this subtree.
            std::size_t param_begin = 0;
            /// @brief Exclusive end index of hyperparameters owned by this subtree.
            std::size_t param_end = 0;
        };

        /// @brief Sentinel slot index denoting a non-ARD kernel leaf.
        static constexpr std::size_t k_no_ard_slot = ~std::size_t{0};

        /// @brief Compiled leaf kernels stored in depth-first left-to-right evaluation order.
        std::vector<KernelLeaf> leaves;
        /// @brief Pre-extracted ARD lengthscale vectors per leaf to avoid heap allocations in hot loops.
        std::vector<std::vector<double>> cached_ls;
        /// @brief ARD leaf slot mapping in [0, ard_leaf_count) per leaf, or k_no_ard_slot if non-ARD.
        std::vector<std::size_t> leaf_ard_slots;
        /// @brief Feature dimension count per leaf for ARD leaves, or 0 for non-ARD leaves.
        std::vector<std::size_t> leaf_dims;
        /// @brief Number of ARD kernel leaves in `leaves`.
        std::size_t ard_leaf_count = 0;
        /// @brief Global-to-leaf hyperparameter bindings in depth-first left-to-right order.
        std::vector<ParameterBinding> bindings;
        /// @brief Root compiled AST node.
        Term root;
        /// @brief Input coordinate requirement inferred across all leaves.
        CoordMode coord_mode = CoordMode::Either;
        /// @brief Minimum number of nested `n x n` (or `n1 x n2`) buffers needed for tree evaluation.
        std::size_t nested_depth = 0;
        /// @brief Exponential evaluation backend used by built-in leaves.
        KernelExp exp_mode = KernelExp::Accurate;
        /// @brief True if at least one leaf can consume scalar squared-distance caches.
        bool reads_dist = false;
        /// @brief True if at least one leaf can consume packed per-dimension ARD squared differences.
        bool needs_ard = false;

        /// @brief Default-constructs an empty compiled implementation.
        Impl() = default;

        /// @brief Copy-constructs a compiled implementation iteratively so deep ASTs do not overflow the stack.
        /// @param[in] other Source compiled implementation.
        Impl(const Impl& other) :
            leaves(other.leaves),
            cached_ls(other.cached_ls),
            leaf_ard_slots(other.leaf_ard_slots),
            leaf_dims(other.leaf_dims),
            ard_leaf_count(other.ard_leaf_count),
            bindings(other.bindings),
            root(clone_term_iterative(other.root)),
            coord_mode(other.coord_mode),
            nested_depth(other.nested_depth),
            exp_mode(other.exp_mode),
            reads_dist(other.reads_dist),
            needs_ard(other.needs_ard)
        {
        }

        /// @brief Move-constructs a compiled implementation.
        Impl(Impl&&) noexcept = default;

        /// @brief Copy-assigns a compiled implementation iteratively.
        /// @param[in] other Source compiled implementation.
        /// @return Reference to *this.
        Impl& operator=(const Impl& other)
        {
            if (this == &other)
            {
                return *this;
            }
            Impl copy(other);
            *this = std::move(copy);
            return *this;
        }

        /// @brief Move-assigns a compiled implementation.
        /// @return Reference to *this.
        Impl& operator=(Impl&&) noexcept = default;

        /// @brief Destroys the compiled AST iteratively to keep stack depth O(1) on deep trees.
        ~Impl() noexcept { destroy_term_iterative(root); }

        /// @brief Iteratively deep-copies a compiled Term tree without recursive calls.
        /// @param[in] src Root compiled term to clone.
        /// @return Cloned compiled term.
        [[nodiscard]] static Term clone_term_iterative(const Term& src)
        {
            if (const auto* leaf = std::get_if<LeafRef>(&src))
            {
                return *leaf;
            }

            struct Frame
            {
                const std::vector<Term>* src_terms;
                std::size_t idx;
                bool is_sum;
                std::size_t param_begin;
                std::size_t param_end;
                std::vector<Term> dst_terms;
            };

            std::vector<Frame> stack;
            auto push_composite = [&stack](const Term& t)
            {
                if (const auto* s = std::get_if<SumTerm>(&t))
                {
                    stack.push_back(
                        Frame{
                            .src_terms = &s->terms,
                            .idx = 0U,
                            .is_sum = true,
                            .param_begin = s->param_begin,
                            .param_end = s->param_end,
                            .dst_terms = {},
                        }
                    );
                    stack.back().dst_terms.reserve(s->terms.size());
                }
                else
                {
                    const auto& p = std::get<ProductTerm>(t);
                    stack.push_back(
                        Frame{
                            .src_terms = &p.terms,
                            .idx = 0U,
                            .is_sum = false,
                            .param_begin = p.param_begin,
                            .param_end = p.param_end,
                            .dst_terms = {},
                        }
                    );
                    stack.back().dst_terms.reserve(p.terms.size());
                }
            };
            push_composite(src);
            while (true)
            {
                auto& top = stack.back();
                if (top.idx >= top.src_terms->size())
                {
                    Term finished = top.is_sum ? Term(
                                                     SumTerm{
                                                         .terms = std::move(top.dst_terms),
                                                         .param_begin = top.param_begin,
                                                         .param_end = top.param_end,
                                                     }
                                                 )
                                               : Term(
                                                     ProductTerm{
                                                         .terms = std::move(top.dst_terms),
                                                         .param_begin = top.param_begin,
                                                         .param_end = top.param_end,
                                                     }
                                                 );
                    stack.pop_back();
                    if (stack.empty())
                    {
                        return finished;
                    }
                    stack.back().dst_terms.push_back(std::move(finished));
                    continue;
                }
                const Term& child = (*top.src_terms)[top.idx++];
                if (const auto* leaf = std::get_if<LeafRef>(&child))
                {
                    top.dst_terms.push_back(*leaf);
                }
                else
                {
                    push_composite(child);
                }
            }
        }

        /// @brief Iteratively destroys nested SumTerm/ProductTerm subtrees without recursive calls.
        /// @param[in,out] target Root compiled term to drain.
        static void destroy_term_iterative(Term& target) noexcept
        {
            std::vector<Term> pending;
            try
            {
                pending.push_back(std::move(target));
                while (!pending.empty())
                {
                    Term cur = std::move(pending.back());
                    pending.pop_back();
                    if (auto* s = std::get_if<SumTerm>(&cur))
                    {
                        for (auto& child : s->terms)
                        {
                            pending.push_back(std::move(child));
                        }
                        s->terms.clear();
                    }
                    else if (auto* p = std::get_if<ProductTerm>(&cur))
                    {
                        for (auto& child : p->terms)
                        {
                            pending.push_back(std::move(child));
                        }
                        p->terms.clear();
                    }
                }
            }
            catch (...)
            {
                // Fallback to bounded cleanup if OOM occurs while draining; noexcept boundary must not throw.
                return;
            }
        }

        /// @brief Refreshes `cached_ls` snapshots from current ARD leaf hyperparameters.
        /// @details Populates per-leaf vectors for ARD leaves and clears non-ARD entries so hot-path evaluation reads
        /// contiguous spans without calling `lengthscales()` allocations.
        void refresh_cached_ls() { detail::refresh_cached_ls_entries(leaves, cached_ls); }

        /// @brief Validates that populated distance and ARD caches match the shape of point matrix `x`.
        /// @details Checks square distance cache dimensions `(n, n)`, ARD point count `n`, ARD feature count `d`, and
        /// packed triangular length `d * n * (n + 1) / 2`.
        /// @param[in] x Input point matrix of shape `(n, d)`.
        /// @param[in] cache Precomputed distance and ARD cache to validate.
        /// @return Empty expected on compatibility, or length/dimension mismatch error.
        /// @retval GprErrorCode::LengthMismatch If cached distance or ARD packed lengths disagree with `x.rows()`.
        /// @retval GprErrorCode::DimensionMismatch If `cache.num_dims() != x.cols()`.
        [[nodiscard]] std::expected<void, GprError> validate_cache_shape(
            MatrixRef<T> x, const DistanceCache<T>& cache
        ) const
        {
            if (cache.has_dist())
            {
                const auto dist = cache.dist_view();
                if (dist.rows() != x.rows() || dist.cols() != x.rows())
                {
                    return std::unexpected(GprError::length_mismatch("distance cache shape mismatch"));
                }
            }
            if (cache.has_ard())
            {
                if (cache.num_points() != x.rows())
                {
                    return std::unexpected(GprError::length_mismatch("ARD cache row count mismatch"));
                }
                if (cache.num_dims() != x.cols())
                {
                    return std::unexpected(GprError::dimension_mismatch(cache.num_dims(), x.cols()));
                }
                const std::size_t per_dim = (x.rows() * (x.rows() + 1U)) / 2U;
                if (cache.ard_packed().size() != cache.num_dims() * per_dim)
                {
                    return std::unexpected(GprError::length_mismatch("ARD cache length mismatch"));
                }
            }
            return {};
        }

        /// @brief Returns the half-open global parameter range `[begin, end)` owned by `term`.
        /// @details Inspects `LeafRef`, `SumTerm`, or `ProductTerm` variant fields.
        /// @param[in] term Compiled AST node.
        /// @return Pair `(begin, end)` of global parameter indices.
        static std::pair<std::size_t, std::size_t> param_range(const Term& term) noexcept
        {
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return {leaf->param_begin, leaf->param_begin + leaf->param_count};
            }
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                return {s->param_begin, s->param_end};
            }
            if (const auto* p = std::get_if<ProductTerm>(&term))
            {
                return {p->param_begin, p->param_end};
            }
            return {0, 0};
        }

        /// @brief Checks whether global hyperparameter index `param_idx` lies inside `term`'s subtree.
        /// @details Evaluates `param_idx >= begin && param_idx < end` using `param_range`.
        /// @param[in] term Compiled AST node.
        /// @param[in] param_idx Global hyperparameter index to query.
        /// @return True if `term` owns `param_idx`.
        static bool contains_param(const Term& term, std::size_t param_idx) noexcept
        {
            const auto [begin, end] = param_range(term);
            return param_idx >= begin && param_idx < end;
        }

        /// @brief Computes the nested intermediate buffer depth required to evaluate `term` without allocation.
        /// @details Leaf terms require depth 0 (they evaluate directly into caller-provided `out` or `scratch`); each
        /// non-leaf child adds one nested level above its own required depth.
        /// @param[in] term Compiled AST node.
        /// @return Maximum nested buffer depth required across all subtrees of `term`.
        static std::size_t compute_nested_depth(const Term& term) noexcept
        {
            struct DepthFrame
            {
                const Term* node;
                std::size_t depth;
            };

            std::vector<DepthFrame> work;
            std::size_t max_depth = 0U;
            try
            {
                work.push_back(DepthFrame{.node = &term, .depth = 0U});
                while (!work.empty())
                {
                    const DepthFrame cur = work.back();
                    work.pop_back();
                    const std::vector<Term>* children = nullptr;
                    if (const auto* s = std::get_if<SumTerm>(cur.node))
                    {
                        children = &s->terms;
                    }
                    else if (const auto* p = std::get_if<ProductTerm>(cur.node))
                    {
                        children = &p->terms;
                    }
                    else
                    {
                        continue;
                    }
                    for (const auto& child : *children)
                    {
                        if (!std::holds_alternative<LeafRef>(child))
                        {
                            const std::size_t child_depth = cur.depth + 1U;
                            max_depth = std::max(max_depth, child_depth);
                            work.push_back(DepthFrame{.node = &child, .depth = child_depth});
                        }
                    }
                }
            }
            catch (...)
            {
                return max_depth;
            }
            return max_depth;
        }

        /// @brief Iteratively compiles a `KernelSpec` AST into a flattened `Term` tree.
        /// @details Uses explicit work and flatten stacks so deep alternating sum/product trees do not overflow
        /// small thread stacks, while preserving strict depth-first left-to-right leaf and parameter ordering.
        /// @param[in] spec Root kernel specification node.
        /// @param[in,out] out_leaves Destination vector of compiled leaf objects.
        /// @param[in,out] out_bindings Destination vector of global hyperparameter bindings.
        /// @param[in,out] next_param_idx Running global parameter counter.
        /// @return Compiled `Term` tree.
        static Term compile_node(
            const KernelSpec& spec,
            std::vector<KernelLeaf>& out_leaves,
            std::vector<ParameterBinding>& out_bindings,
            std::size_t& next_param_idx
        )
        {
            enum class FrameKind : std::uint8_t
            {
                Sum,
                Product,
            };

            struct CompileFrame
            {
                FrameKind kind;
                std::vector<Term> terms;
                std::size_t param_begin;
            };

            struct WorkItem
            {
                const KernelSpec* node{nullptr};
                bool close_frame{false};
            };

            std::vector<CompileFrame> frames;
            std::vector<WorkItem> work;
            std::vector<const KernelSpec*> flatten;
            Term root;

            auto emit_term = [&frames, &root](Term&& t)
            {
                if (frames.empty())
                {
                    root = std::move(t);
                }
                else
                {
                    frames.back().terms.push_back(std::move(t));
                }
            };

            work.push_back(WorkItem{.node = &spec, .close_frame = false});
            while (!work.empty())
            {
                const WorkItem item = work.back();
                work.pop_back();

                if (item.close_frame)
                {
                    CompileFrame done = std::move(frames.back());
                    frames.pop_back();
                    if (done.kind == FrameKind::Sum)
                    {
                        emit_term(
                            SumTerm{
                                .terms = std::move(done.terms),
                                .param_begin = done.param_begin,
                                .param_end = next_param_idx,
                            }
                        );
                    }
                    else
                    {
                        emit_term(
                            ProductTerm{
                                .terms = std::move(done.terms),
                                .param_begin = done.param_begin,
                                .param_end = next_param_idx,
                            }
                        );
                    }
                    continue;
                }

                const KernelSpec& cur = *item.node;
                if (const auto* leaf = std::get_if<KernelLeaf>(&cur.node_))
                {
                    const std::size_t leaf_idx = out_leaves.size();
                    const std::size_t p_begin = next_param_idx;
                    const std::size_t p_count = detail::leaf_num_params(*leaf);
                    out_leaves.push_back(*leaf);
                    // Bind all hyperparameters of this leaf consecutively in depth-first left-to-right order.
                    for (std::size_t local = 0; local < p_count; ++local)
                    {
                        out_bindings.push_back(
                            ParameterBinding{
                                .index = next_param_idx++,
                                .leaf_id = leaf_idx,
                                .local_index = local,
                            }
                        );
                    }
                    emit_term(LeafRef{.leaf_idx = leaf_idx, .param_begin = p_begin, .param_count = p_count});
                    continue;
                }

                const bool is_sum = std::holds_alternative<KernelSpec::SumNode>(cur.node_);
                frames.push_back(
                    CompileFrame{
                        .kind = is_sum ? FrameKind::Sum : FrameKind::Product,
                        .terms = {},
                        .param_begin = next_param_idx,
                    }
                );
                work.push_back(WorkItem{.node = nullptr, .close_frame = true});

                // Flatten nested binary sums/products into work in reverse order so popping visits left-to-right.
                flatten.push_back(&cur);
                while (!flatten.empty())
                {
                    const KernelSpec* f = flatten.back();
                    flatten.pop_back();
                    if (is_sum)
                    {
                        if (const auto* s = std::get_if<KernelSpec::SumNode>(&f->node_))
                        {
                            flatten.push_back(s->lhs.get());
                            flatten.push_back(s->rhs.get());
                            continue;
                        }
                    }
                    else
                    {
                        if (const auto* p = std::get_if<KernelSpec::ProductNode>(&f->node_))
                        {
                            flatten.push_back(p->lhs.get());
                            flatten.push_back(p->rhs.get());
                            continue;
                        }
                    }
                    work.push_back(WorkItem{.node = f, .close_frame = false});
                }
            }
            return root;
        }

        /// @brief Verifies that every ARD leaf in `leaves` expects `x_dim` input features.
        /// @details Checks `RbfArdKernel`, `MaternArdKernel`, and `RationalQuadraticArdKernel` lengthscale counts.
        /// @param[in] x_dim Feature dimensionality of the point coordinate matrix.
        /// @return Empty expected on match, or dimension-mismatch error.
        /// @retval GprErrorCode::DimensionMismatch If any ARD leaf expects a different feature count.
        [[nodiscard]] std::expected<void, GprError> validate_ard_dims(std::size_t x_dim) const
        {
            for (const auto& leaf : leaves)
            {
                std::size_t expected_dim = 0;
                if (const auto* k = std::get_if<RbfArdKernel>(&leaf))
                {
                    expected_dim = k->num_params();
                }
                else if (const auto* k_m = std::get_if<MaternArdKernel>(&leaf))
                {
                    expected_dim = k_m->num_params();
                }
                else if (const auto* k_rq = std::get_if<RationalQuadraticArdKernel>(&leaf))
                {
                    // Subtract the scalar alpha parameter without allocating lengthscales() on the hot path.
                    expected_dim = k_rq->num_params() - 1U;
                }
                if (expected_dim > 0 && x_dim != expected_dim)
                {
                    return std::unexpected(GprError::dimension_mismatch(x_dim, expected_dim));
                }
            }
            return {};
        }

        /// @brief Validates kernel mode compatibility, dimensions, and finiteness for squared-distance calls.
        /// @details Rejects kernels with coordinate-only leaves (`Points` or `Mixed`), enforces non-empty and optional
        /// squareness constraints on `dist`, verifies matching `out` shape, and scans the active region of `dist` for
        /// finite values.
        /// @param[in] dist Input squared-distance matrix view.
        /// @param[in] out Reference output matrix view.
        /// @param[in] uplo Active matrix triangle when `square == true`.
        /// @param[in] square True for square self-covariance/derivative calls, false for cross-covariance calls.
        /// @return Empty expected on success, or mode/empty/shape/finiteness error.
        /// @retval GprErrorCode::UnsupportedKernelOperation If the compiled tree depends on raw coordinates.
        /// @retval GprErrorCode::EmptyInput If `dist` has zero rows or columns.
        /// @retval GprErrorCode::LengthMismatch If `square && dist.rows() != dist.cols()` or `out` shape mismatches.
        /// @retval GprErrorCode::NonFiniteInput If any accessed entry of `dist` is non-finite.
        [[nodiscard]] std::expected<void, GprError> validate_dist_call(
            MatrixRef<T> dist, MatrixRef<T> out, Triangle uplo, bool square
        ) const
        {
            // Reject distance-only evaluation when any leaf requires coordinate vectors.
            if (coord_mode == CoordMode::Points || coord_mode == CoordMode::Mixed)
            {
                return std::unexpected(
                    GprError::unsupported_kernel_operation(
                        "coordinate-dependent kernel cannot be evaluated from squared distances"
                    )
                );
            }
            if (dist.rows() == 0 || dist.cols() == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (square && dist.rows() != dist.cols())
            {
                return std::unexpected(
                    GprError::length_mismatch("distance matrix for square evaluation must be square")
                );
            }
            if (out.rows() != dist.rows() || out.cols() != dist.cols())
            {
                return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
            }
            const std::size_t n_rows = dist.rows();
            const std::size_t n_cols = dist.cols();
            if (!square && dist.is_contiguous())
            {
                if (!detail::all_finite_slice(dist.data(), n_rows * n_cols))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
                return {};
            }
            for (std::size_t col = 0; col < n_cols; ++col)
            {
                const std::size_t row_begin = (square && uplo == Triangle::Lower) ? col : 0U;
                const std::size_t row_end = (square && uplo == Triangle::Upper) ? (col + 1U) : n_rows;
                if (!detail::all_finite_slice(&dist(row_begin, col), row_end - row_begin))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
            return {};
        }

        /// @brief Validates point matrix emptiness, square output shape, and ARD feature dimensionality.
        /// @details Checks structural preconditions without scanning coordinate values; callers performing full-matrix
        /// evaluation follow up with `require_matrix_finite_simd(x)`.
        /// @param[in] x Input point coordinate matrix of shape `(n, d)`.
        /// @param[in] out Reference square output matrix of shape `(n, n)`.
        /// @return Empty expected on success, or shape/dimension validation error.
        /// @retval GprErrorCode::EmptyInput If `x` has zero rows or columns.
        /// @retval GprErrorCode::LengthMismatch If `out` is not `(x.rows(), x.rows())`.
        /// @retval GprErrorCode::DimensionMismatch If `x.cols()` mismatches an ARD leaf's dimensionality.
        [[nodiscard]] std::expected<void, GprError> validate_points_shape(MatrixRef<T> x, MatrixRef<T> out) const
        {
            if (x.rows() == 0 || x.cols() == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (out.rows() != x.rows() || out.cols() != x.rows())
            {
                return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
            }
            return validate_ard_dims(x.cols());
        }

        /// @brief Validates point matrix shape, square output shape, ARD dimensions, and coordinate finiteness.
        /// @details Enforces empty -> shape -> ARD dimension -> finite scan order before point-based tree evaluation.
        /// @param[in] x Input point coordinate matrix of shape `(n, d)`.
        /// @param[in] out Reference square output matrix of shape `(n, n)`.
        /// @return Empty expected on success, or validation error.
        /// @retval GprErrorCode::EmptyInput If `x` has zero rows or columns.
        /// @retval GprErrorCode::LengthMismatch If `out` is not `(x.rows(), x.rows())`.
        /// @retval GprErrorCode::DimensionMismatch If `x.cols()` mismatches an ARD leaf's dimensionality.
        /// @retval GprErrorCode::NonFiniteInput If any entry in `x` is non-finite.
        [[nodiscard]] std::expected<void, GprError> validate_points_call(MatrixRef<T> x, MatrixRef<T> out) const
        {
            if (auto shape = validate_points_shape(x, out); !shape.has_value())
            {
                return shape;
            }
            return require_matrix_finite_simd(x);
        }

        /// @brief Validates point matrices `x1` and `x2` and rectangular output `out` for cross-covariance evaluation.
        /// @details Enforces empty -> output shape -> matching feature dimension -> ARD dimension -> finiteness checks.
        /// @param[in] x1 First point matrix of shape `(n1, d)`.
        /// @param[in] x2 Second point matrix of shape `(n2, d)`.
        /// @param[in] out Reference output matrix of shape `(n1, n2)`.
        /// @return Empty expected on success, or validation error.
        /// @retval GprErrorCode::EmptyInput If `x1` or `x2` has zero rows or columns.
        /// @retval GprErrorCode::LengthMismatch If `out` is not `(x1.rows(), x2.rows())`.
        /// @retval GprErrorCode::DimensionMismatch If `x1.cols() != x2.cols()` or ARD leaf dimension mismatches.
        /// @retval GprErrorCode::NonFiniteInput If any entry in `x1` or `x2` is non-finite.
        [[nodiscard]] std::expected<void, GprError> validate_cross_points_call(
            MatrixRef<T> x1, MatrixRef<T> x2, MatrixRef<T> out
        ) const
        {
            if (x1.rows() == 0 || x1.cols() == 0 || x2.rows() == 0 || x2.cols() == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (out.rows() != x1.rows() || out.cols() != x2.rows())
            {
                return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
            }
            if (x1.cols() != x2.cols())
            {
                return std::unexpected(GprError::dimension_mismatch(x2.cols(), x1.cols()));
            }
            if (auto ard = validate_ard_dims(x1.cols()); !ard.has_value())
            {
                return ard;
            }
            if (auto f1 = require_matrix_finite_simd(x1); !f1.has_value())
            {
                return f1;
            }
            return require_matrix_finite_simd(x2);
        }

        // --- Evaluation helpers ---

        /// @brief Verifies that `nested` has enough buffer depth and element capacity for `level_idx`.
        /// @details Returns `GprErrorCode::WorkspaceTooSmall` without allocating if capacity is insufficient.
        /// @param[in,out] nested Caller-supplied nested buffer pool.
        /// @param[in] level_idx Required zero-based buffer level index.
        /// @param[in] rows Required row capacity.
        /// @param[in] cols Required column capacity.
        /// @return Empty expected when `nested` suffices, or workspace error.
        /// @retval GprErrorCode::WorkspaceTooSmall If depth or matrix dimensions are too small.
        [[nodiscard]] std::expected<void, GprError> ensure_nested(
            NestedBuffers<T>& nested, std::size_t level_idx, std::size_t rows, std::size_t cols
        ) const
        {
            if (nested.depth() <= level_idx || nested.rows() < rows || nested.cols() < cols)
            {
                return std::unexpected(GprError::workspace_too_small());
            }
            return {};
        }

        /// @brief Writes WorkspaceTooSmall into `err` out-of-line so recursive frames construct no `GprError`.
        /// @param[out] err Destination error slot.
        /// @return Always false.
        [[nodiscard]] static CXXGPR_NOINLINE bool fail_workspace_too_small(std::optional<GprError>& err) noexcept
        {
            err = GprError::workspace_too_small();
            return false;
        }

        /// @brief Writes UnsupportedKernelOperation for an empty composite into `err` out-of-line.
        /// @param[out] err Destination error slot.
        /// @return Always false.
        [[nodiscard]] static CXXGPR_NOINLINE bool fail_empty_composite(std::optional<GprError>& err)
        {
            err = GprError::unsupported_kernel_operation("composite kernel has no terms");
            return false;
        }

        /// @brief Writes IndexOutOfRange for a parameter index into `err` out-of-line.
        /// @param[out] err Destination error slot.
        /// @return Always false.
        [[nodiscard]] static CXXGPR_NOINLINE bool fail_param_out_of_range(std::optional<GprError>& err)
        {
            err = GprError::index_out_of_range("kernel parameter index out of range");
            return false;
        }

        /// @brief Verifies that `nested` has enough buffer depth and element capacity without constructing expected.
        /// @param[in,out] nested Caller-supplied nested buffer pool.
        /// @param[in] level_idx Required zero-based buffer level index.
        /// @param[in] rows Required row capacity.
        /// @param[in] cols Required column capacity.
        /// @param[out] err Populated with WorkspaceTooSmall on failure.
        /// @return True when `nested` suffices, false otherwise.
        [[nodiscard]] bool ensure_nested_ok(
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            std::size_t rows,
            std::size_t cols,
            std::optional<GprError>& err
        ) const noexcept
        {
            if (nested.depth() <= level_idx || nested.rows() < rows || nested.cols() < cols)
            {
                return fail_workspace_too_small(err);
            }
            return true;
        }

        /// @brief Assigns an error into `err` from a failed `std::expected` in an out-of-line helper.
        /// @param[in] res Failed expected result.
        /// @param[out] err Destination error slot.
        /// @return True if `res` succeeded, false after writing `err`.
        [[nodiscard]] static CXXGPR_NOINLINE bool capture_expected(
            std::expected<void, GprError>&& res, std::optional<GprError>& err
        ) noexcept
        {
            if (res.has_value())
            {
                return true;
            }
            err = std::move(res).error();
            return false;
        }

        /// @brief Invariant context passed by single reference across `eval_term_rec` calls.
        /// @tparam LeafEval Callable evaluating a single leaf term.
        template <typename LeafEval> struct TermEvalCtx
        {
            /// @brief Active matrix region (`Lower`, `Upper`, or `Full`).
            Triangle uplo;
            /// @brief Pre-allocated intermediate buffer pool for nested subtrees.
            NestedBuffers<T>& nested;
            /// @brief Out-of-line error slot populated on failure.
            std::optional<GprError>& err;
            /// @brief Leaf evaluator invoked out-of-line at leaves.
            const LeafEval& eval_leaf;
        };

        /// @brief Directly evaluates a compiled `Term` subtree with minimal per-level stack frame size.
        /// @details Recurses directly without intermediate lambda frames, passes matrices by reference and invariants
        /// in a single context pointer, and routes all errors and leaf evaluations through `CXXGPR_NOINLINE` helpers.
        /// @tparam LeafEval Callable invoked as `eval_leaf(std::size_t leaf_idx, const MatrixMut<T>& dst, std::size_t
        /// depth, std::optional<GprError>& err)`.
        /// @param[in] term Compiled AST node to evaluate.
        /// @param[out] out Destination matrix view receiving the evaluation.
        /// @param[in,out] scratch Reusable matrix view for sibling leaf evaluation.
        /// @param[in] level_idx Current nesting depth inside `ctx.nested`.
        /// @param[in] ctx Invariant evaluation context.
        /// @return True on success, false after writing `ctx.err`.
        template <typename LeafEval>
        [[nodiscard]] bool eval_term_rec(
            const Term& term,
            const MatrixMut<T>& out,
            const MatrixMut<T>& scratch,
            std::size_t level_idx,
            const TermEvalCtx<LeafEval>& ctx
        ) const
        {
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return ctx.eval_leaf(leaf->leaf_idx, out, level_idx, ctx.err);
            }
            const bool is_sum = std::holds_alternative<SumTerm>(term);
            const auto& terms = is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
            if (terms.empty())
            {
                return fail_empty_composite(ctx.err);
            }
            if (!is_sum)
            {
                // Fold all ConstantKernel scalars out of products and evaluate varying factors first.
                const T scale = constant_scale(terms);
                bool has_varying = false;
                for (const Term& child : terms)
                {
                    if (is_constant_term(child))
                    {
                        continue;
                    }
                    if (!has_varying)
                    {
                        if (!eval_term_rec(child, out, scratch, level_idx, ctx))
                        {
                            return false;
                        }
                        has_varying = true;
                        continue;
                    }
                    if (std::holds_alternative<LeafRef>(child))
                    {
                        if (!eval_term_rec(child, scratch, out, level_idx, ctx))
                        {
                            return false;
                        }
                        mul_into(out, scratch.as_ref(), ctx.uplo);
                    }
                    else
                    {
                        if (!ensure_nested_ok(ctx.nested, level_idx, out.rows(), out.cols(), ctx.err))
                        {
                            return false;
                        }
                        const auto buf = ctx.nested.level(level_idx, out.rows(), out.cols());
                        if (!eval_term_rec(child, buf, scratch, level_idx + 1U, ctx))
                        {
                            return false;
                        }
                        mul_into(out, buf.as_ref(), ctx.uplo);
                    }
                }
                if (!has_varying)
                {
                    fill_region(out, scale, ctx.uplo);
                    return true;
                }
                if (scale != T{1})
                {
                    scale_region(out, scale, ctx.uplo);
                }
                return true;
            }

            // SumTerm: evaluate first summand directly into `out`, then accumulate remaining summands.
            if (!eval_term_rec(terms[0], out, scratch, level_idx, ctx))
            {
                return false;
            }
            for (std::size_t k = 1U; k < terms.size(); ++k)
            {
                if (std::holds_alternative<LeafRef>(terms[k]))
                {
                    if (!eval_term_rec(terms[k], scratch, out, level_idx, ctx))
                    {
                        return false;
                    }
                    add_into(out, scratch.as_ref(), ctx.uplo);
                }
                else
                {
                    if (!ensure_nested_ok(ctx.nested, level_idx, out.rows(), out.cols(), ctx.err))
                    {
                        return false;
                    }
                    const auto buf = ctx.nested.level(level_idx, out.rows(), out.cols());
                    if (!eval_term_rec(terms[k], buf, scratch, level_idx + 1U, ctx))
                    {
                        return false;
                    }
                    add_into(out, buf.as_ref(), ctx.uplo);
                }
            }
            return true;
        }

        /// @brief Evaluates one distance leaf out-of-line so recursive frames stay free of `std::expected`.
        /// @param[in] leaf_idx Leaf index into `leaves`.
        /// @param[in] dist Square squared-distance matrix.
        /// @param[out] out Destination matrix view.
        /// @param[in] uplo Active matrix triangle.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing `err`.
        [[nodiscard]] CXXGPR_NOINLINE bool eval_leaf_dist_step(
            std::size_t leaf_idx, MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo, std::optional<GprError>& err
        ) const
        {
            return capture_expected(detail::leaf_apply_dist(leaves[leaf_idx], dist, out, uplo, exp_mode), err);
        }

        /// @brief Recursive body of `eval_term_dist` using `bool` + out-of-line error slot.
        /// @param[in] term Compiled AST node to evaluate.
        /// @param[in] dist Square squared-distance matrix.
        /// @param[out] out Destination Gram matrix view.
        /// @param[in] uplo Active matrix triangle.
        /// @param[in,out] scratch Reusable scratch matrix.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing `err`.
        [[nodiscard]] bool eval_term_dist_rec(
            const Term& term,
            const MatrixRef<T>& dist,
            const MatrixMut<T>& out,
            Triangle uplo,
            const MatrixMut<T>& scratch,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            std::optional<GprError>& err
        ) const
        {
            const auto leaf_eval =
                [this, &dist, uplo](
                    std::size_t leaf_idx, const MatrixMut<T>& dst, std::size_t /*depth*/, std::optional<GprError>& e
                )
            {
                return eval_leaf_dist_step(leaf_idx, dist, dst, uplo, e);
            };
            const TermEvalCtx<decltype(leaf_eval)> ctx{
                .uplo = uplo,
                .nested = nested,
                .err = err,
                .eval_leaf = leaf_eval,
            };
            return eval_term_rec(term, out, scratch, level_idx, ctx);
        }

        /// @brief Recursively evaluates a compiled `Term` into `out` from squared Euclidean distances.
        /// @param[in] term Compiled AST node to evaluate.
        /// @param[in] dist Square squared-distance matrix.
        /// @param[out] out Destination Gram matrix view.
        /// @param[in] uplo Active matrix triangle.
        /// @param[in,out] scratch Reusable scratch matrix.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @return Empty expected on success, or evaluation/workspace error.
        [[nodiscard]] std::expected<void, GprError> eval_term_dist(
            const Term& term,
            MatrixRef<T> dist,
            MatrixMut<T> out,
            Triangle uplo,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested,
            std::size_t level_idx
        ) const
        {
            std::optional<GprError> err;
            if (!eval_term_dist_rec(term, dist, out, uplo, scratch, nested, level_idx, err))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }

        /// @brief Evaluates a single compiled leaf at `leaf_idx` using caches when available.
        /// @details Prefers `DistanceCache::dist_view()` for distance leaves and `DistanceCache::ard_packed()` for ARD
        /// leaves; materializes square distances in `nested` for uncached `CustomKernel` leaves.
        /// @param[in] leaf_idx Zero-based index into `leaves`.
        /// @param[in] x Input point matrix of shape `(n, d)`.
        /// @param[out] out Destination square matrix of shape `(n, n)`.
        /// @param[in] uplo Active matrix triangle to populate.
        /// @param[in,out] nested Pre-allocated buffer pool used if `CustomKernel` needs uncached squared distances.
        /// @param[in] level_idx Nesting depth inside `nested` for `CustomKernel` distance buffer.
        /// @param[in] cache Optional precomputed distance and ARD cache.
        /// @param[out] ard_s2_buf Optional packed buffer capturing forward ARD $s^2$ slices.
        /// @param[in,out] ard_s2_ready Optional per-ARD-slot readiness flags set when `ard_s2_buf` is populated.
        /// @return Empty expected on success, or workspace/evaluation error.
        [[nodiscard]] std::expected<void, GprError> eval_leaf_at(
            std::size_t leaf_idx,
            MatrixRef<T> x,
            MatrixMut<T> out,
            Triangle uplo,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            const DistanceCache<T>* cache = nullptr,
            std::span<T> ard_s2_buf = {},
            std::span<std::uint8_t> ard_s2_ready = {}
        ) const
        {
            const KernelLeaf& kl = leaves[leaf_idx];
            // Route to cached distance or packed ARD tables whenever compatible caches are provided.
            if (cache != nullptr && cache->has_dist() && leaf_reads_distances(kl))
            {
                return detail::leaf_apply_dist(kl, cache->dist_view(), out, uplo, exp_mode);
            }
            if (leaf_needs_ard(kl))
            {
                std::span<T> out_ard_s2{};
                const std::size_t slot = leaf_ard_slots[leaf_idx];
                const std::size_t per_dim = (x.rows() * (x.rows() + 1U)) / 2U;
                if (slot < ard_s2_ready.size() && per_dim > 0U && (ard_s2_buf.size() / per_dim) > slot)
                {
                    out_ard_s2 = ard_s2_buf.subspan(slot * per_dim, per_dim);
                }
                std::expected<void, GprError> res;
                if (cache != nullptr && cache->has_ard())
                {
                    res = detail::leaf_apply_ard_packed(
                        kl, cached_ls[leaf_idx], cache->ard_packed(), x.rows(), out, uplo, exp_mode, out_ard_s2
                    );
                }
                else
                {
                    res = detail::leaf_apply_points(kl, x, out, uplo, exp_mode, cached_ls[leaf_idx], out_ard_s2);
                }
                if (res.has_value() && !out_ard_s2.empty())
                {
                    ard_s2_ready[slot] = 1U;
                }
                return res;
            }
            // Custom leaves expect a squared-distance matrix; build it in nested scratch when uncached.
            if (const auto* custom = std::get_if<CustomKernel>(&kl))
            {
                if (auto ok = ensure_nested(nested, level_idx, x.rows(), x.rows()); !ok.has_value())
                {
                    return ok;
                }
                auto dist_buf = nested.level(level_idx, x.rows(), x.rows());
                fill_square_sq_dist(x, dist_buf);
                return custom->template term<T>().apply(dist_buf.as_ref(), out, uplo);
            }
            return detail::leaf_apply_points(kl, x, out, uplo, exp_mode, cached_ls[leaf_idx]);
        }

        /// @brief Evaluates a single compiled leaf with incremental ARD $s^2$ state across Cached or Uncached modes.
        /// @details Routes ARD leaves to `leaf_apply_ard_incremental` (with `ard_packed` when `cache != nullptr &&
        /// cache->has_ard()` or uncached coordinates otherwise) and delegates non-ARD leaves cleanly to `eval_leaf_at`.
        /// @param[in] leaf_idx Zero-based index into `leaves`.
        /// @param[in] x Input point matrix of shape `(n, d)`.
        /// @param[in] cache Optional precomputed distance and ARD cache pointer.
        /// @param[out] out Destination square matrix `(n, n)`.
        /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        /// @param[in,out] nested Pre-allocated nested buffer pool.
        /// @param[in,out] ard_s2_cache Caller-owned packed $s^2$ buffer for this leaf.
        /// @param[in,out] prev_ls Caller-owned lengthscale snapshot buffer for this leaf.
        /// @param[in,out] s2_valid True while `ard_s2_cache` matches `prev_ls`.
        /// @param[in,out] inc_steps Consecutive incremental updates applied since last rebuild.
        /// @param[in] commit_s2 True to mutate `ard_s2_cache`/`prev_ls`; false for non-destructive trials.
        /// @return Empty expected on success, or workspace/evaluation error.
        [[nodiscard]] std::expected<void, GprError> eval_leaf_ard_incremental(
            std::size_t leaf_idx,
            MatrixRef<T> x,
            const DistanceCache<T>* cache,
            MatrixMut<T> out,
            Triangle uplo,
            NestedBuffers<T>& nested,
            std::span<T> ard_s2_cache,
            std::span<double> prev_ls,
            bool& s2_valid,
            std::size_t& inc_steps,
            bool commit_s2
        ) const
        {
            const KernelLeaf& kl = leaves[leaf_idx];
            if (leaf_needs_ard(kl))
            {
                const std::span<const T> ard_packed =
                    (cache != nullptr && cache->has_ard()) ? cache->ard_packed() : std::span<const T>{};
                return detail::leaf_apply_ard_incremental(
                    kl, cached_ls[leaf_idx], x, ard_packed, x.rows(), out, uplo, exp_mode, ard_s2_cache, prev_ls,
                    s2_valid, inc_steps, commit_s2
                );
            }
            if (!(cache != nullptr && cache->has_dist() && leaf_reads_distances(kl)))
            {
                if (auto fin = require_matrix_finite_simd(x); !fin.has_value())
                {
                    return fin;
                }
            }
            return eval_leaf_at(leaf_idx, x, out, uplo, nested, 0U, cache);
        }

        /// @brief Commits accepted lengthscales of `leaf_idx` to its `ard_s2_cache` baseline.
        /// @details No-op for non-ARD leaves.
        /// @param[in] leaf_idx Zero-based index into `leaves`.
        /// @param[in] x Input point matrix of shape `(n, d)`.
        /// @param[in] cache Optional precomputed distance and ARD cache pointer.
        /// @param[in,out] ard_s2_cache Caller-owned packed $s^2$ buffer for this leaf.
        /// @param[in,out] prev_ls Caller-owned lengthscale snapshot buffer for this leaf.
        /// @param[in,out] s2_valid True while `ard_s2_cache` matches `prev_ls`.
        /// @param[in,out] inc_steps Consecutive incremental updates applied since last rebuild.
        /// @return Empty expected on success, or error on invalid inputs.
        [[nodiscard]] std::expected<void, GprError> commit_leaf_ard_s2_impl(
            std::size_t leaf_idx,
            MatrixRef<T> x,
            const DistanceCache<T>* cache,
            std::span<T> ard_s2_cache,
            std::span<double> prev_ls,
            bool& s2_valid,
            std::size_t& inc_steps
        ) const
        {
            const KernelLeaf& kl = leaves[leaf_idx];
            if (!leaf_needs_ard(kl))
            {
                return {};
            }
            const std::span<const T> ard_packed =
                (cache != nullptr && cache->has_ard()) ? cache->ard_packed() : std::span<const T>{};
            return detail::leaf_commit_ard_s2(
                kl, cached_ls[leaf_idx], x, ard_packed, x.rows(), ard_s2_cache, prev_ls, s2_valid, inc_steps
            );
        }

        /// @brief Invariant context for `eval_combine_term_rec` to keep recursive stack frames minimal.
        struct CombineCtx
        {
            /// @brief Precomputed leaf Gram matrix views indexed by `leaf_idx`.
            std::span<const MatrixRef<T>> leaf_grams;
            /// @brief Active matrix triangle to combine.
            Triangle uplo;
            /// @brief Pre-allocated intermediate buffer pool for nested subtrees.
            NestedBuffers<T>& nested;
            /// @brief Out-of-line error slot.
            std::optional<GprError>& err;
        };

        /// @brief Recursive body of `eval_combine_term` with minimal per-level stack frame size.
        /// @param[in] term Compiled AST node.
        /// @param[out] out Destination matrix view.
        /// @param[in] level_idx Current nesting depth inside `ctx.nested`.
        /// @param[in] ctx Invariant combine context.
        /// @return True on success, false after writing `ctx.err`.
        [[nodiscard]] bool eval_combine_term_rec(
            const Term& term, const MatrixMut<T>& out, std::size_t level_idx, const CombineCtx& ctx
        ) const
        {
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                copy_into(out, ctx.leaf_grams[leaf->leaf_idx], ctx.uplo);
                return true;
            }
            const bool is_sum = std::holds_alternative<SumTerm>(term);
            const auto& terms = is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
            if (terms.empty())
            {
                return fail_empty_composite(ctx.err);
            }
            if (!eval_combine_term_rec(terms[0], out, level_idx, ctx))
            {
                return false;
            }
            for (std::size_t k = 1; k < terms.size(); ++k)
            {
                if (const auto* leaf = std::get_if<LeafRef>(&terms[k]))
                {
                    if (is_sum)
                    {
                        add_into(out, ctx.leaf_grams[leaf->leaf_idx], ctx.uplo);
                    }
                    else
                    {
                        mul_into(out, ctx.leaf_grams[leaf->leaf_idx], ctx.uplo);
                    }
                }
                else
                {
                    if (!ensure_nested_ok(ctx.nested, level_idx, out.rows(), out.cols(), ctx.err))
                    {
                        return false;
                    }
                    const auto buf = ctx.nested.level(level_idx, out.rows(), out.cols());
                    if (!eval_combine_term_rec(terms[k], buf, level_idx + 1U, ctx))
                    {
                        return false;
                    }
                    if (is_sum)
                    {
                        add_into(out, buf.as_ref(), ctx.uplo);
                    }
                    else
                    {
                        mul_into(out, buf.as_ref(), ctx.uplo);
                    }
                }
            }
            return true;
        }

        /// @brief Recursively combines precomputed leaf Gram matrices into `out` according to `term`.
        /// @param[in] term Compiled AST node.
        /// @param[in] leaf_grams Precomputed Gram matrix views indexed by `leaf_idx`.
        /// @param[out] out Destination matrix view.
        /// @param[in] uplo Active matrix triangle to combine.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool for nested subtrees.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @return Empty expected on success, or workspace/operation error.
        [[nodiscard]] std::expected<void, GprError> eval_combine_term(
            const Term& term,
            std::span<const MatrixRef<T>> leaf_grams,
            MatrixMut<T> out,
            Triangle uplo,
            NestedBuffers<T>& nested,
            std::size_t level_idx
        ) const
        {
            std::optional<GprError> err;
            const CombineCtx ctx{
                .leaf_grams = leaf_grams,
                .uplo = uplo,
                .nested = nested,
                .err = err,
            };
            if (!eval_combine_term_rec(term, out, level_idx, ctx))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }

        /// @brief Evaluates one point leaf out-of-line so recursive frames stay free of `std::expected`.
        /// @param[in] leaf_idx Leaf index into `leaves`.
        /// @param[in] x Input point matrix.
        /// @param[out] out Destination matrix view.
        /// @param[in] uplo Active matrix triangle.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @param[in] cache Optional precomputed distance and ARD cache.
        /// @param[out] ard_s2_buf Optional packed forward $s^2$ capture buffer.
        /// @param[in,out] ard_s2_ready Optional per-ARD-leaf readiness flags.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing `err`.
        [[nodiscard]] CXXGPR_NOINLINE bool eval_leaf_points_step(
            std::size_t leaf_idx,
            MatrixRef<T> x,
            MatrixMut<T> out,
            Triangle uplo,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            const DistanceCache<T>* cache,
            std::span<T> ard_s2_buf,
            std::span<std::uint8_t> ard_s2_ready,
            std::optional<GprError>& err
        ) const
        {
            return capture_expected(
                eval_leaf_at(leaf_idx, x, out, uplo, nested, level_idx, cache, ard_s2_buf, ard_s2_ready), err
            );
        }

        /// @brief Recursive body of `eval_term_points` using `bool` + out-of-line error slot.
        /// @param[in] term Compiled AST node to evaluate.
        /// @param[in] x Input point matrix.
        /// @param[out] out Destination matrix view.
        /// @param[in] uplo Active matrix triangle.
        /// @param[in,out] scratch Reusable scratch matrix.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @param[in] cache Optional precomputed distance and ARD cache.
        /// @param[out] err Populated with the error on failure.
        /// @param[out] ard_s2_buf Optional packed forward $s^2$ capture buffer.
        /// @param[in,out] ard_s2_ready Optional per-ARD-leaf readiness flags.
        /// @return True on success, false after writing `err`.
        [[nodiscard]] bool eval_term_points_rec(
            const Term& term,
            const MatrixRef<T>& x,
            const MatrixMut<T>& out,
            Triangle uplo,
            const MatrixMut<T>& scratch,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            const DistanceCache<T>* cache,
            std::optional<GprError>& err,
            std::span<T> ard_s2_buf = {},
            std::span<std::uint8_t> ard_s2_ready = {}
        ) const
        {
            const auto leaf_eval =
                [this, &x, uplo, &nested, cache, ard_s2_buf, ard_s2_ready](
                    std::size_t leaf_idx, const MatrixMut<T>& dst, std::size_t depth, std::optional<GprError>& e
                )
            {
                return eval_leaf_points_step(leaf_idx, x, dst, uplo, nested, depth, cache, ard_s2_buf, ard_s2_ready, e);
            };
            const TermEvalCtx<decltype(leaf_eval)> ctx{
                .uplo = uplo,
                .nested = nested,
                .err = err,
                .eval_leaf = leaf_eval,
            };
            return eval_term_rec(term, out, scratch, level_idx, ctx);
        }

        /// @brief Recursively evaluates a compiled `Term` from point coordinates (and optional cache).
        /// @param[in] term Compiled AST node to evaluate.
        /// @param[in] x Input point matrix.
        /// @param[out] out Destination matrix view.
        /// @param[in] uplo Active matrix triangle.
        /// @param[in,out] scratch Reusable scratch matrix.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @param[in] cache Optional precomputed distance and ARD cache.
        /// @param[out] ard_s2_buf Optional packed forward $s^2$ capture buffer.
        /// @param[in,out] ard_s2_ready Optional per-ARD-leaf readiness flags.
        /// @return Empty expected on success, or workspace/evaluation error.
        [[nodiscard]] std::expected<void, GprError> eval_term_points(
            const Term& term,
            MatrixRef<T> x,
            MatrixMut<T> out,
            Triangle uplo,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            const DistanceCache<T>* cache = nullptr,
            std::span<T> ard_s2_buf = {},
            std::span<std::uint8_t> ard_s2_ready = {}
        ) const
        {
            std::optional<GprError> err;
            if (!eval_term_points_rec(
                    term, x, out, uplo, scratch, nested, level_idx, cache, err, ard_s2_buf, ard_s2_ready
                ))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }

        /// @brief Evaluates one cross-distance leaf out-of-line so recursive frames stay free of `std::expected`.
        /// @param[in] leaf_idx Leaf index into `leaves`.
        /// @param[in] dist Rectangular squared-distance matrix.
        /// @param[out] out Destination cross-covariance matrix view.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing `err`.
        [[nodiscard]] CXXGPR_NOINLINE bool eval_leaf_cross_dist_step(
            std::size_t leaf_idx, MatrixRef<T> dist, MatrixMut<T> out, std::optional<GprError>& err
        ) const
        {
            return capture_expected(detail::leaf_apply_cross_dist(leaves[leaf_idx], dist, out, exp_mode), err);
        }

        /// @brief Recursive body of `eval_term_cross_dist` using `bool` + out-of-line error slot.
        /// @param[in] term Compiled AST node to evaluate.
        /// @param[in] dist Rectangular squared-distance matrix.
        /// @param[out] out Destination cross-covariance matrix view.
        /// @param[in,out] scratch Reusable scratch matrix.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing `err`.
        [[nodiscard]] bool eval_term_cross_dist_rec(
            const Term& term,
            const MatrixRef<T>& dist,
            const MatrixMut<T>& out,
            const MatrixMut<T>& scratch,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            std::optional<GprError>& err
        ) const
        {
            const auto leaf_eval =
                [this, &dist](
                    std::size_t leaf_idx, const MatrixMut<T>& dst, std::size_t /*depth*/, std::optional<GprError>& e
                )
            {
                return eval_leaf_cross_dist_step(leaf_idx, dist, dst, e);
            };
            const TermEvalCtx<decltype(leaf_eval)> ctx{
                .uplo = Triangle::Full,
                .nested = nested,
                .err = err,
                .eval_leaf = leaf_eval,
            };
            return eval_term_rec(term, out, scratch, level_idx, ctx);
        }

        /// @brief Recursively evaluates a compiled `Term` cross-covariance from rectangular squared distances.
        /// @param[in] term Compiled AST node to evaluate.
        /// @param[in] dist Rectangular squared-distance matrix.
        /// @param[out] out Destination cross-covariance matrix view.
        /// @param[in,out] scratch Reusable scratch matrix.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @return Empty expected on success, or workspace/evaluation error.
        [[nodiscard]] std::expected<void, GprError> eval_term_cross_dist(
            const Term& term,
            MatrixRef<T> dist,
            MatrixMut<T> out,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested,
            std::size_t level_idx
        ) const
        {
            std::optional<GprError> err;
            if (!eval_term_cross_dist_rec(term, dist, out, scratch, nested, level_idx, err))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }

        /// @brief Evaluates one cross-points leaf out-of-line so recursive frames stay free of `std::expected`.
        /// @param[in] leaf_idx Leaf index into `leaves`.
        /// @param[in] x1 First point matrix.
        /// @param[in] x2 Second point matrix.
        /// @param[out] out Destination cross-covariance matrix view.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing `err`.
        [[nodiscard]] CXXGPR_NOINLINE bool eval_leaf_cross_points_step(
            std::size_t leaf_idx,
            MatrixRef<T> x1,
            MatrixRef<T> x2,
            MatrixMut<T> out,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            std::optional<GprError>& err
        ) const
        {
            const KernelLeaf& kl = leaves[leaf_idx];
            if (const auto* custom = std::get_if<CustomKernel>(&kl))
            {
                if (!ensure_nested_ok(nested, level_idx, x1.rows(), x2.rows(), err))
                {
                    return false;
                }
                auto dist_buf = nested.level(level_idx, x1.rows(), x2.rows());
                fill_cross_sq_dist(x1, x2, dist_buf);
                return capture_expected(custom->template term<T>().apply_cross(dist_buf.as_ref(), out), err);
            }
            return capture_expected(
                detail::leaf_apply_cross_points(kl, x1, x2, out, exp_mode, cached_ls[leaf_idx], true), err
            );
        }

        /// @brief Recursive body of `eval_term_cross_points` using `bool` + out-of-line error slot.
        /// @param[in] term Compiled AST node to evaluate.
        /// @param[in] x1 First point matrix.
        /// @param[in] x2 Second point matrix.
        /// @param[out] out Destination cross-covariance matrix view.
        /// @param[in,out] scratch Reusable scratch matrix.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing `err`.
        [[nodiscard]] bool eval_term_cross_points_rec(
            const Term& term,
            const MatrixRef<T>& x1,
            const MatrixRef<T>& x2,
            const MatrixMut<T>& out,
            const MatrixMut<T>& scratch,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            std::optional<GprError>& err
        ) const
        {
            const auto leaf_eval =
                [this, &x1, &x2, &nested](
                    std::size_t leaf_idx, const MatrixMut<T>& dst, std::size_t depth, std::optional<GprError>& e
                )
            {
                return eval_leaf_cross_points_step(leaf_idx, x1, x2, dst, nested, depth, e);
            };
            const TermEvalCtx<decltype(leaf_eval)> ctx{
                .uplo = Triangle::Full,
                .nested = nested,
                .err = err,
                .eval_leaf = leaf_eval,
            };
            return eval_term_rec(term, out, scratch, level_idx, ctx);
        }

        /// @brief Recursively evaluates a compiled `Term` cross-covariance from two point coordinate matrices.
        /// @param[in] term Compiled AST node to evaluate.
        /// @param[in] x1 First point matrix.
        /// @param[in] x2 Second point matrix.
        /// @param[out] out Destination cross-covariance matrix view.
        /// @param[in,out] scratch Reusable scratch matrix.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @return Empty expected on success, or workspace/evaluation error.
        [[nodiscard]] std::expected<void, GprError> eval_term_cross_points(
            const Term& term,
            MatrixRef<T> x1,
            MatrixRef<T> x2,
            MatrixMut<T> out,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested,
            std::size_t level_idx
        ) const
        {
            std::optional<GprError> err;
            if (!eval_term_cross_points_rec(term, x1, x2, out, scratch, nested, level_idx, err))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }

        /// @brief Writes LengthMismatch for an undersized diagonal scratch buffer out-of-line.
        /// @param[out] err Destination error slot.
        /// @return Always false.
        [[nodiscard]] static CXXGPR_NOINLINE bool fail_diag_scratch_too_small(std::optional<GprError>& err)
        {
            err = GprError::length_mismatch("diagonal evaluation scratch buffer is too small");
            return false;
        }

        /// @brief Evaluates one diagonal leaf out-of-line so recursive frames stay free of `std::expected`.
        /// @param[in] leaf_idx Leaf index into `leaves`.
        /// @param[in] x Input point coordinate matrix.
        /// @param[out] out Output diagonal span.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing `err`.
        [[nodiscard]] CXXGPR_NOINLINE bool fill_diag_leaf_step(
            std::size_t leaf_idx, const MatrixRef<T>& x, std::span<T> out, std::optional<GprError>& err
        ) const
        {
            return capture_expected(detail::leaf_fill_diag(leaves[leaf_idx], x, out, cached_ls[leaf_idx]), err);
        }

        /// @brief Invariant context for `eval_fill_diag_rec` to keep recursive stack frames minimal.
        struct FillDiagCtx
        {
            /// @brief Input point coordinate matrix.
            MatrixRef<T> x;
            /// @brief Caller-supplied scratch vector sliced by recursion level.
            std::span<T> scratch_vec;
            /// @brief Out-of-line error slot.
            std::optional<GprError>& err;
        };

        /// @brief Recursive body of `eval_fill_diag` with minimal per-level stack frame size.
        /// @param[in] term Compiled AST node.
        /// @param[out] out Output diagonal span of length `n`.
        /// @param[in] level_idx Current nesting depth into `ctx.scratch_vec`.
        /// @param[in] ctx Invariant diagonal evaluation context.
        /// @return True on success, false after writing `ctx.err`.
        [[nodiscard]] bool eval_fill_diag_rec(
            const Term& term, std::span<T> out, std::size_t level_idx, const FillDiagCtx& ctx
        ) const
        {
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return fill_diag_leaf_step(leaf->leaf_idx, ctx.x, out, ctx.err);
            }
            const bool is_sum = std::holds_alternative<SumTerm>(term);
            const auto& terms = is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
            if (terms.empty())
            {
                return fail_empty_composite(ctx.err);
            }
            if (!eval_fill_diag_rec(terms[0], out, level_idx, ctx))
            {
                return false;
            }
            if (terms.size() == 1U)
            {
                return true;
            }
            // Slice scratch_vec by recursion depth so sibling terms never clobber outer levels.
            const std::size_t slice_offset = level_idx * out.size();
            if (ctx.scratch_vec.size() < slice_offset + out.size())
            {
                return fail_diag_scratch_too_small(ctx.err);
            }
            for (std::size_t k = 1; k < terms.size(); ++k)
            {
                const std::span<T> sub(ctx.scratch_vec.data() + slice_offset, out.size());
                if (!eval_fill_diag_rec(terms[k], sub, level_idx + 1U, ctx))
                {
                    return false;
                }
                for (std::size_t idx = 0; idx < out.size(); ++idx)
                {
                    if (is_sum)
                    {
                        out[idx] += sub[idx];
                    }
                    else
                    {
                        out[idx] *= sub[idx];
                    }
                }
            }
            return true;
        }

        /// @brief Recursively evaluates the self-covariance diagonal $k(x_i, x_i)$ of a compiled `Term`.
        /// @param[in] term Compiled AST node.
        /// @param[in] x Input point coordinate matrix of shape `(n, d)`.
        /// @param[out] out Output diagonal span of length `n`.
        /// @param[in,out] scratch_vec Caller-supplied scratch vector sliced by recursion level.
        /// @param[in] level_idx Current nesting depth into `scratch_vec`.
        /// @return Empty expected on success, or workspace/evaluation error.
        /// @retval GprErrorCode::LengthMismatch If `scratch_vec` is shorter than `(level_idx + 1) * out.size()`.
        [[nodiscard]] std::expected<void, GprError> eval_fill_diag(
            const Term& term, MatrixRef<T> x, std::span<T> out, std::span<T> scratch_vec, std::size_t level_idx
        ) const
        {
            std::optional<GprError> err;
            const FillDiagCtx ctx{
                .x = x,
                .scratch_vec = scratch_vec,
                .err = err,
            };
            if (!eval_fill_diag_rec(term, out, level_idx, ctx))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }

        /// @brief Evaluates one distance-leaf gradient out-of-line so recursive frames stay free of `std::expected`.
        /// @param[in] leaf Leaf AST reference.
        /// @param[in] dist Square squared-distance matrix.
        /// @param[out] d_k Destination derivative matrix.
        /// @param[in] param_idx Global hyperparameter index.
        /// @param[in] uplo Active matrix triangle.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing `err`.
        [[nodiscard]] CXXGPR_NOINLINE bool grad_leaf_dist_step(
            const LeafRef& leaf,
            MatrixRef<T> dist,
            MatrixMut<T> d_k,
            std::size_t param_idx,
            Triangle uplo,
            std::optional<GprError>& err
        ) const
        {
            return capture_expected(
                detail::leaf_grad_dist(leaves[leaf.leaf_idx], dist, d_k, param_idx - leaf.param_begin, uplo, exp_mode),
                err
            );
        }

        /// @brief Invariant context for `eval_grad_dist_rec` to keep recursive frames minimal.
        struct GradDistCtx
        {
            /// @brief Square squared-distance matrix.
            MatrixRef<T> dist;
            /// @brief Active matrix triangle.
            Triangle uplo;
            /// @brief First reusable square scratch buffer.
            MatrixMut<T> scratch_a;
            /// @brief Second reusable square scratch buffer.
            MatrixMut<T> scratch_b;
            /// @brief Pre-allocated intermediate buffer pool.
            NestedBuffers<T>& nested;
            /// @brief Out-of-line error slot.
            std::optional<GprError>& err;
        };

        /// @brief Recursive body of `eval_grad_dist` with minimal stack frame size.
        /// @param[in] term Compiled AST node.
        /// @param[out] d_k Output derivative matrix.
        /// @param[in] param_idx Global hyperparameter index.
        /// @param[in] level_idx Current nesting depth inside `ctx.nested`.
        /// @param[in] ctx Invariant gradient evaluation context.
        /// @return True on success, false after writing `ctx.err`.
        [[nodiscard]] bool eval_grad_dist_rec(
            const Term& term,
            const MatrixMut<T>& d_k,
            std::size_t param_idx,
            std::size_t level_idx,
            const GradDistCtx& ctx
        ) const
        {
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return grad_leaf_dist_step(*leaf, ctx.dist, d_k, param_idx, ctx.uplo, ctx.err);
            }
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                // In a sum, only the summand containing param_idx contributes to dK/dtheta.
                for (const auto& child : s->terms)
                {
                    if (contains_param(child, param_idx))
                    {
                        return eval_grad_dist_rec(child, d_k, param_idx, level_idx, ctx);
                    }
                }
                return fail_param_out_of_range(ctx.err);
            }
            const auto& p = std::get<ProductTerm>(term);
            std::size_t owner = p.terms.size();
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (contains_param(p.terms[m], param_idx))
                {
                    owner = m;
                    break;
                }
            }
            if (owner == p.terms.size())
            {
                return fail_param_out_of_range(ctx.err);
            }
            // Differentiate the owning factor into d_k, then multiply by all sibling factor Grams.
            if (!eval_grad_dist_rec(p.terms[owner], d_k, param_idx, level_idx, ctx))
            {
                return false;
            }
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (m == owner)
                {
                    continue;
                }
                if (!eval_term_dist_rec(
                        p.terms[m], ctx.dist, ctx.scratch_a, ctx.uplo, ctx.scratch_b, ctx.nested, level_idx, ctx.err
                    ))
                {
                    return false;
                }
                mul_into(d_k, ctx.scratch_a.as_ref(), ctx.uplo);
            }
            return true;
        }

        /// @brief Recursively evaluates $\partial K / \partial \theta_p$ from squared Euclidean distances.
        /// @param[in] term Compiled AST node.
        /// @param[in] dist Square squared-distance matrix.
        /// @param[out] d_k Output derivative matrix.
        /// @param[in] param_idx Global hyperparameter index.
        /// @param[in] uplo Active matrix triangle.
        /// @param[in,out] scratch_a First reusable square scratch buffer.
        /// @param[in,out] scratch_b Second reusable square scratch buffer.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @return Empty expected on success, or index/workspace/evaluation error.
        [[nodiscard]] std::expected<void, GprError> eval_grad_dist(
            const Term& term,
            MatrixRef<T> dist,
            MatrixMut<T> d_k,
            std::size_t param_idx,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            NestedBuffers<T>& nested,
            std::size_t level_idx
        ) const
        {
            std::optional<GprError> err;
            const GradDistCtx ctx{
                .dist = dist,
                .uplo = uplo,
                .scratch_a = scratch_a,
                .scratch_b = scratch_b,
                .nested = nested,
                .err = err,
            };
            if (!eval_grad_dist_rec(term, d_k, param_idx, level_idx, ctx))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }

        /// @brief Evaluates one point-leaf gradient out-of-line so recursive frames stay free of `std::expected`.
        /// @param[in] leaf Leaf AST reference.
        /// @param[in] x Input point matrix.
        /// @param[out] d_k Output derivative matrix.
        /// @param[in] param_idx Global hyperparameter index.
        /// @param[in] uplo Active matrix triangle.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @param[in] cache Optional precomputed distance and ARD cache.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing `err`.
        [[nodiscard]] CXXGPR_NOINLINE bool grad_leaf_points_step(
            const LeafRef& leaf,
            MatrixRef<T> x,
            MatrixMut<T> d_k,
            std::size_t param_idx,
            Triangle uplo,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            const DistanceCache<T>* cache,
            std::optional<GprError>& err
        ) const
        {
            const KernelLeaf& kl = leaves[leaf.leaf_idx];
            if (cache != nullptr)
            {
                if (cache->has_dist() && leaf_reads_distances(kl))
                {
                    return capture_expected(
                        detail::leaf_grad_dist(
                            kl, cache->dist_view(), d_k, param_idx - leaf.param_begin, uplo, exp_mode
                        ),
                        err
                    );
                }
                if (cache->has_ard() && leaf_needs_ard(kl))
                {
                    return capture_expected(
                        detail::leaf_grad_ard_packed(
                            kl, cached_ls[leaf.leaf_idx], cache->ard_packed(), x.rows(), d_k,
                            param_idx - leaf.param_begin, uplo, exp_mode
                        ),
                        err
                    );
                }
            }
            if (const auto* custom = std::get_if<CustomKernel>(&kl))
            {
                if (!ensure_nested_ok(nested, level_idx, x.rows(), x.rows(), err))
                {
                    return false;
                }
                auto dist_buf = nested.level(level_idx, x.rows(), x.rows());
                fill_square_sq_dist(x, dist_buf);
                return capture_expected(
                    custom->template term<T>().grad(dist_buf.as_ref(), d_k, param_idx - leaf.param_begin, uplo), err
                );
            }
            return capture_expected(
                detail::leaf_grad_points(
                    kl, x, d_k, param_idx - leaf.param_begin, uplo, exp_mode, cached_ls[leaf.leaf_idx]
                ),
                err
            );
        }

        /// @brief Invariant context for `eval_grad_points_rec` to keep recursive frames minimal.
        struct GradPointsCtx
        {
            /// @brief Input point matrix.
            MatrixRef<T> x;
            /// @brief Active matrix triangle.
            Triangle uplo;
            /// @brief First reusable square scratch buffer.
            MatrixMut<T> scratch_a;
            /// @brief Second reusable square scratch buffer.
            MatrixMut<T> scratch_b;
            /// @brief Pre-allocated intermediate buffer pool.
            NestedBuffers<T>& nested;
            /// @brief Optional precomputed distance and ARD cache.
            const DistanceCache<T>* cache;
            /// @brief Out-of-line error slot.
            std::optional<GprError>& err;
        };

        /// @brief Recursive body of `eval_grad_points` with minimal stack frame size.
        /// @param[in] term Compiled AST node.
        /// @param[out] d_k Output derivative matrix.
        /// @param[in] param_idx Global hyperparameter index.
        /// @param[in] level_idx Current nesting depth inside `ctx.nested`.
        /// @param[in] ctx Invariant gradient evaluation context.
        /// @return True on success, false after writing `ctx.err`.
        [[nodiscard]] bool eval_grad_points_rec(
            const Term& term,
            const MatrixMut<T>& d_k,
            std::size_t param_idx,
            std::size_t level_idx,
            const GradPointsCtx& ctx
        ) const
        {
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return grad_leaf_points_step(
                    *leaf, ctx.x, d_k, param_idx, ctx.uplo, ctx.nested, level_idx, ctx.cache, ctx.err
                );
            }
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                // Route sum gradient directly to the child owning param_idx.
                for (const auto& child : s->terms)
                {
                    if (contains_param(child, param_idx))
                    {
                        return eval_grad_points_rec(child, d_k, param_idx, level_idx, ctx);
                    }
                }
                return fail_param_out_of_range(ctx.err);
            }
            const auto& p = std::get<ProductTerm>(term);
            std::size_t owner = p.terms.size();
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (contains_param(p.terms[m], param_idx))
                {
                    owner = m;
                    break;
                }
            }
            if (owner == p.terms.size())
            {
                return fail_param_out_of_range(ctx.err);
            }
            // Product rule: evaluate the owning factor's derivative, then scale by sibling factor values.
            if (!eval_grad_points_rec(p.terms[owner], d_k, param_idx, level_idx, ctx))
            {
                return false;
            }
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (m == owner)
                {
                    continue;
                }
                if (!eval_term_points_rec(
                        p.terms[m], ctx.x, ctx.scratch_a, ctx.uplo, ctx.scratch_b, ctx.nested, level_idx, ctx.cache,
                        ctx.err
                    ))
                {
                    return false;
                }
                mul_into(d_k, ctx.scratch_a.as_ref(), ctx.uplo);
            }
            return true;
        }

        /// @brief Recursively evaluates $\partial K / \partial \theta_p$ from point coordinates (and optional cache).
        /// @param[in] term Compiled AST node.
        /// @param[in] x Input point matrix.
        /// @param[out] d_k Output derivative matrix.
        /// @param[in] param_idx Global hyperparameter index.
        /// @param[in] uplo Active matrix triangle.
        /// @param[in,out] scratch_a First reusable square scratch buffer.
        /// @param[in,out] scratch_b Second reusable square scratch buffer.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @param[in] cache Optional precomputed distance and ARD cache.
        /// @return Empty expected on success, or index/workspace/evaluation error.
        [[nodiscard]] std::expected<void, GprError> eval_grad_points(
            const Term& term,
            MatrixRef<T> x,
            MatrixMut<T> d_k,
            std::size_t param_idx,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            const DistanceCache<T>* cache = nullptr
        ) const
        {
            std::optional<GprError> err;
            const GradPointsCtx ctx{
                .x = x,
                .uplo = uplo,
                .scratch_a = scratch_a,
                .scratch_b = scratch_b,
                .nested = nested,
                .cache = cache,
                .err = err,
            };
            if (!eval_grad_points_rec(term, d_k, param_idx, level_idx, ctx))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }

        /// @brief Evaluates one distance-leaf Hessian out-of-line so recursive frames stay free of `std::expected`.
        /// @param[in] leaf Leaf AST reference.
        /// @param[in] dist Square squared-distance matrix.
        /// @param[out] d2_k Output second-derivative matrix.
        /// @param[in] i First global hyperparameter index.
        /// @param[in] j Second global hyperparameter index.
        /// @param[in] uplo Active matrix triangle.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing `err`.
        [[nodiscard]] CXXGPR_NOINLINE bool hess_leaf_dist_step(
            const LeafRef& leaf,
            MatrixRef<T> dist,
            MatrixMut<T> d2_k,
            std::size_t i,
            std::size_t j,
            Triangle uplo,
            std::optional<GprError>& err
        ) const
        {
            return capture_expected(
                detail::leaf_hess_dist(
                    leaves[leaf.leaf_idx], dist, d2_k, i - leaf.param_begin, j - leaf.param_begin, uplo, exp_mode
                ),
                err
            );
        }

        /// @brief Invariant context for `eval_hess_dist_rec` to keep recursive frames minimal.
        struct HessDistCtx
        {
            /// @brief Square squared-distance matrix.
            MatrixRef<T> dist;
            /// @brief First global hyperparameter index.
            std::size_t i;
            /// @brief Second global hyperparameter index.
            std::size_t j;
            /// @brief Active matrix triangle.
            Triangle uplo;
            /// @brief First reusable square scratch buffer.
            MatrixMut<T> scratch_a;
            /// @brief Second reusable square scratch buffer.
            MatrixMut<T> scratch_b;
            /// @brief Third reusable square scratch buffer.
            MatrixMut<T> scratch_c;
            /// @brief Pre-allocated intermediate buffer pool.
            NestedBuffers<T>& nested;
            /// @brief Out-of-line error slot.
            std::optional<GprError>& err;
        };

        /// @brief Recursive body of `eval_hess_dist` with minimal stack frame size.
        /// @param[in] term Compiled AST node.
        /// @param[out] d2_k Output second-derivative matrix.
        /// @param[in] level_idx Current nesting depth inside `ctx.nested`.
        /// @param[in] ctx Invariant Hessian evaluation context.
        /// @return True on success, false after writing `ctx.err`.
        [[nodiscard]] bool eval_hess_dist_rec(
            const Term& term, const MatrixMut<T>& d2_k, std::size_t level_idx, const HessDistCtx& ctx
        ) const
        {
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return hess_leaf_dist_step(*leaf, ctx.dist, d2_k, ctx.i, ctx.j, ctx.uplo, ctx.err);
            }
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                for (const auto& child : s->terms)
                {
                    const bool has_i = contains_param(child, ctx.i);
                    const bool has_j = contains_param(child, ctx.j);
                    if (has_i && has_j)
                    {
                        return eval_hess_dist_rec(child, d2_k, level_idx, ctx);
                    }
                    if (has_i || has_j)
                    {
                        // Cross-derivatives between separate summands vanish identically.
                        fill_region(d2_k, T{0}, ctx.uplo);
                        return true;
                    }
                }
                return fail_param_out_of_range(ctx.err);
            }
            const auto& p = std::get<ProductTerm>(term);
            std::size_t owner_i = p.terms.size();
            std::size_t owner_j = p.terms.size();
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (contains_param(p.terms[m], ctx.i))
                {
                    owner_i = m;
                }
                if (contains_param(p.terms[m], ctx.j))
                {
                    owner_j = m;
                }
            }
            if (owner_i == p.terms.size() || owner_j == p.terms.size())
            {
                return fail_param_out_of_range(ctx.err);
            }
            // Case 1: both i and j belong to the same factor -> second derivative of owner_i times remaining factors.
            if (owner_i == owner_j)
            {
                if (!eval_hess_dist_rec(p.terms[owner_i], d2_k, level_idx, ctx))
                {
                    return false;
                }
                for (std::size_t m = 0; m < p.terms.size(); ++m)
                {
                    if (m == owner_i)
                    {
                        continue;
                    }
                    if (!eval_term_dist_rec(
                            p.terms[m], ctx.dist, ctx.scratch_b, ctx.uplo, ctx.scratch_c, ctx.nested, level_idx, ctx.err
                        ))
                    {
                        return false;
                    }
                    mul_into(d2_k, ctx.scratch_b.as_ref(), ctx.uplo);
                }
                return true;
            }
            // Case 2: i and j belong to distinct factors -> product of first derivatives times remaining factors.
            const GradDistCtx gctx{
                .dist = ctx.dist,
                .uplo = ctx.uplo,
                .scratch_a = ctx.scratch_b,
                .scratch_b = ctx.scratch_c,
                .nested = ctx.nested,
                .err = ctx.err,
            };
            if (!eval_grad_dist_rec(p.terms[owner_i], d2_k, ctx.i, level_idx, gctx))
            {
                return false;
            }
            if (!eval_grad_dist_rec(p.terms[owner_j], ctx.scratch_a, ctx.j, level_idx, gctx))
            {
                return false;
            }
            mul_into(d2_k, ctx.scratch_a.as_ref(), ctx.uplo);
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (m == owner_i || m == owner_j)
                {
                    continue;
                }
                if (!eval_term_dist_rec(
                        p.terms[m], ctx.dist, ctx.scratch_b, ctx.uplo, ctx.scratch_c, ctx.nested, level_idx, ctx.err
                    ))
                {
                    return false;
                }
                mul_into(d2_k, ctx.scratch_b.as_ref(), ctx.uplo);
            }
            return true;
        }

        /// @brief Recursively evaluates the second log-hyperparameter derivative from squared Euclidean distances.
        /// @param[in] term Compiled AST node.
        /// @param[in] dist Square squared-distance matrix.
        /// @param[out] d2_k Output second-derivative matrix.
        /// @param[in] i First global hyperparameter index.
        /// @param[in] j Second global hyperparameter index.
        /// @param[in] uplo Active matrix triangle.
        /// @param[in,out] scratch_a First reusable square scratch buffer.
        /// @param[in,out] scratch_b Second reusable square scratch buffer.
        /// @param[in,out] scratch_c Third reusable square scratch buffer.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @return Empty expected on success, or index/workspace/evaluation error.
        [[nodiscard]] std::expected<void, GprError> eval_hess_dist(
            const Term& term,
            MatrixRef<T> dist,
            MatrixMut<T> d2_k,
            std::size_t i,
            std::size_t j,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            MatrixMut<T> scratch_c,
            NestedBuffers<T>& nested,
            std::size_t level_idx
        ) const
        {
            std::optional<GprError> err;
            const HessDistCtx ctx{
                .dist = dist,
                .i = i,
                .j = j,
                .uplo = uplo,
                .scratch_a = scratch_a,
                .scratch_b = scratch_b,
                .scratch_c = scratch_c,
                .nested = nested,
                .err = err,
            };
            if (!eval_hess_dist_rec(term, d2_k, level_idx, ctx))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }

        /// @brief Evaluates one point-leaf Hessian out-of-line so recursive frames stay free of `std::expected`.
        /// @param[in] leaf Leaf AST reference.
        /// @param[in] x Input point matrix.
        /// @param[out] d2_k Output second-derivative matrix.
        /// @param[in] i First global hyperparameter index.
        /// @param[in] j Second global hyperparameter index.
        /// @param[in] uplo Active matrix triangle.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @param[in] cache Optional precomputed distance and ARD cache.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing `err`.
        [[nodiscard]] CXXGPR_NOINLINE bool hess_leaf_points_step(
            const LeafRef& leaf,
            MatrixRef<T> x,
            MatrixMut<T> d2_k,
            std::size_t i,
            std::size_t j,
            Triangle uplo,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            const DistanceCache<T>* cache,
            std::optional<GprError>& err
        ) const
        {
            const KernelLeaf& kl = leaves[leaf.leaf_idx];
            if (cache != nullptr)
            {
                if (cache->has_dist() && leaf_reads_distances(kl))
                {
                    return capture_expected(
                        detail::leaf_hess_dist(
                            kl, cache->dist_view(), d2_k, i - leaf.param_begin, j - leaf.param_begin, uplo, exp_mode
                        ),
                        err
                    );
                }
                if (cache->has_ard() && leaf_needs_ard(kl))
                {
                    return capture_expected(
                        detail::leaf_hess_ard_packed(
                            kl, cached_ls[leaf.leaf_idx], cache->ard_packed(), x.rows(), d2_k, i - leaf.param_begin,
                            j - leaf.param_begin, uplo, exp_mode
                        ),
                        err
                    );
                }
            }
            if (const auto* custom = std::get_if<CustomKernel>(&kl))
            {
                if (!ensure_nested_ok(nested, level_idx, x.rows(), x.rows(), err))
                {
                    return false;
                }
                auto dist_buf = nested.level(level_idx, x.rows(), x.rows());
                fill_square_sq_dist(x, dist_buf);
                return capture_expected(
                    custom->template term<T>().hess(
                        dist_buf.as_ref(), d2_k, i - leaf.param_begin, j - leaf.param_begin, uplo
                    ),
                    err
                );
            }
            return capture_expected(
                detail::leaf_hess_points(
                    kl, x, d2_k, i - leaf.param_begin, j - leaf.param_begin, uplo, exp_mode, cached_ls[leaf.leaf_idx]
                ),
                err
            );
        }

        /// @brief Invariant context for `eval_hess_points_rec` to keep recursive frames minimal.
        struct HessPointsCtx
        {
            /// @brief Input point matrix.
            MatrixRef<T> x;
            /// @brief First global hyperparameter index.
            std::size_t i;
            /// @brief Second global hyperparameter index.
            std::size_t j;
            /// @brief Active matrix triangle.
            Triangle uplo;
            /// @brief First reusable square scratch buffer.
            MatrixMut<T> scratch_a;
            /// @brief Second reusable square scratch buffer.
            MatrixMut<T> scratch_b;
            /// @brief Third reusable square scratch buffer.
            MatrixMut<T> scratch_c;
            /// @brief Pre-allocated intermediate buffer pool.
            NestedBuffers<T>& nested;
            /// @brief Optional precomputed distance and ARD cache.
            const DistanceCache<T>* cache;
            /// @brief Out-of-line error slot.
            std::optional<GprError>& err;
        };

        /// @brief Recursive body of `eval_hess_points` with minimal stack frame size.
        /// @param[in] term Compiled AST node.
        /// @param[out] d2_k Output second-derivative matrix.
        /// @param[in] level_idx Current nesting depth inside `ctx.nested`.
        /// @param[in] ctx Invariant Hessian evaluation context.
        /// @return True on success, false after writing `ctx.err`.
        [[nodiscard]] bool eval_hess_points_rec(
            const Term& term, const MatrixMut<T>& d2_k, std::size_t level_idx, const HessPointsCtx& ctx
        ) const
        {
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return hess_leaf_points_step(
                    *leaf, ctx.x, d2_k, ctx.i, ctx.j, ctx.uplo, ctx.nested, level_idx, ctx.cache, ctx.err
                );
            }
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                for (const auto& child : s->terms)
                {
                    const bool has_i = contains_param(child, ctx.i);
                    const bool has_j = contains_param(child, ctx.j);
                    if (has_i && has_j)
                    {
                        return eval_hess_points_rec(child, d2_k, level_idx, ctx);
                    }
                    if (has_i || has_j)
                    {
                        // Mixed partial derivatives across separate summands vanish identically.
                        fill_region(d2_k, T{0}, ctx.uplo);
                        return true;
                    }
                }
                return fail_param_out_of_range(ctx.err);
            }
            const auto& p = std::get<ProductTerm>(term);
            std::size_t owner_i = p.terms.size();
            std::size_t owner_j = p.terms.size();
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (contains_param(p.terms[m], ctx.i))
                {
                    owner_i = m;
                }
                if (contains_param(p.terms[m], ctx.j))
                {
                    owner_j = m;
                }
            }
            if (owner_i == p.terms.size() || owner_j == p.terms.size())
            {
                return fail_param_out_of_range(ctx.err);
            }
            // Same-factor Hessian times remaining sibling factor Grams.
            if (owner_i == owner_j)
            {
                if (!eval_hess_points_rec(p.terms[owner_i], d2_k, level_idx, ctx))
                {
                    return false;
                }
                for (std::size_t m = 0; m < p.terms.size(); ++m)
                {
                    if (m == owner_i)
                    {
                        continue;
                    }
                    if (!eval_term_points_rec(
                            p.terms[m], ctx.x, ctx.scratch_b, ctx.uplo, ctx.scratch_c, ctx.nested, level_idx, ctx.cache,
                            ctx.err
                        ))
                    {
                        return false;
                    }
                    mul_into(d2_k, ctx.scratch_b.as_ref(), ctx.uplo);
                }
                return true;
            }
            // Cross-factor Hessian: multiply gradients of owner_i and owner_j by any remaining factors.
            const GradPointsCtx gctx{
                .x = ctx.x,
                .uplo = ctx.uplo,
                .scratch_a = ctx.scratch_b,
                .scratch_b = ctx.scratch_c,
                .nested = ctx.nested,
                .cache = ctx.cache,
                .err = ctx.err,
            };
            if (!eval_grad_points_rec(p.terms[owner_i], d2_k, ctx.i, level_idx, gctx))
            {
                return false;
            }
            if (!eval_grad_points_rec(p.terms[owner_j], ctx.scratch_a, ctx.j, level_idx, gctx))
            {
                return false;
            }
            mul_into(d2_k, ctx.scratch_a.as_ref(), ctx.uplo);
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (m == owner_i || m == owner_j)
                {
                    continue;
                }
                if (!eval_term_points_rec(
                        p.terms[m], ctx.x, ctx.scratch_b, ctx.uplo, ctx.scratch_c, ctx.nested, level_idx, ctx.cache,
                        ctx.err
                    ))
                {
                    return false;
                }
                mul_into(d2_k, ctx.scratch_b.as_ref(), ctx.uplo);
            }
            return true;
        }

        /// @brief Recursively evaluates the second log-hyperparameter derivative from point coordinates.
        /// @param[in] term Compiled AST node.
        /// @param[in] x Input point matrix.
        /// @param[out] d2_k Output second-derivative matrix.
        /// @param[in] i First global hyperparameter index.
        /// @param[in] j Second global hyperparameter index.
        /// @param[in] uplo Active matrix triangle.
        /// @param[in,out] scratch_a First reusable square scratch buffer.
        /// @param[in,out] scratch_b Second reusable square scratch buffer.
        /// @param[in,out] scratch_c Third reusable square scratch buffer.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @param[in] cache Optional precomputed distance and ARD cache.
        /// @return Empty expected on success, or index/workspace/evaluation error.
        [[nodiscard]] std::expected<void, GprError> eval_hess_points(
            const Term& term,
            MatrixRef<T> x,
            MatrixMut<T> d2_k,
            std::size_t i,
            std::size_t j,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            MatrixMut<T> scratch_c,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            const DistanceCache<T>* cache = nullptr
        ) const
        {
            std::optional<GprError> err;
            const HessPointsCtx ctx{
                .x = x,
                .i = i,
                .j = j,
                .uplo = uplo,
                .scratch_a = scratch_a,
                .scratch_b = scratch_b,
                .scratch_c = scratch_c,
                .nested = nested,
                .cache = cache,
                .err = err,
            };
            if (!eval_hess_points_rec(term, d2_k, level_idx, ctx))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }

        // --- keep_plan and weighted_grads ---

        /// @brief Returns true if `term` is a `LeafRef` wrapping a `ConstantKernel`.
        /// @details Constant factors fold into a single scalar multiplier in product retention and adjoint contraction.
        /// @param[in] term Compiled AST node.
        /// @return True when `term` is a `ConstantKernel` leaf.
        [[nodiscard]] bool is_constant_term(const Term& term) const noexcept
        {
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return std::holds_alternative<ConstantKernel>(leaves[leaf->leaf_idx]);
            }
            return false;
        }

        /// @brief Counts non-constant child factors in a product's term list.
        /// @details Excludes `ConstantKernel` leaves because they reduce to a scalar multiplier without needing Gram
        /// matrix buffers.
        /// @param[in] terms Child factors of a `ProductTerm`.
        /// @return Number of spatially varying factors.
        [[nodiscard]] std::size_t varying_factors(const std::vector<Term>& terms) const noexcept
        {
            std::size_t count = 0;
            for (const auto& t : terms)
            {
                if (!is_constant_term(t))
                {
                    ++count;
                }
            }
            return count;
        }

        /// @brief Computes the scalar product of all `ConstantKernel` factors in `terms`.
        /// @details Returns `T{1}` when no constant leaf appears directly in `terms`.
        /// @param[in] terms Child factors of a `ProductTerm`.
        /// @return Combined scalar constant multiplier cast to `T`.
        [[nodiscard]] T constant_scale(std::span<const Term> terms) const noexcept
        {
            T scale = T{1};
            for (const auto& t : terms)
            {
                if (const auto* leaf = std::get_if<LeafRef>(&t))
                {
                    if (const auto* ck = std::get_if<ConstantKernel>(&leaves[leaf->leaf_idx]))
                    {
                        scale *= static_cast<T>(ck->constant());
                    }
                }
            }
            return scale;
        }

        /// @brief Returns a span of top-level additive summands (`SumTerm::terms` or single-element `{&root, 1}`).
        /// @details Allows `keep_plan` and `weighted_grads` to iterate uniformly over sum-of-products roots and single
        /// non-sum roots.
        /// @return Contiguous span of top-level `Term` nodes.
        [[nodiscard]] std::span<const Term> root_terms() const noexcept
        {
            if (const auto* s = std::get_if<SumTerm>(&root))
            {
                return s->terms;
            }
            return std::span<const Term>(&root, 1U);
        }

        /// @brief Estimates peak `n x n` buffers needed for an unkept reverse contraction walk over `term`.
        /// @details Establishes the memory budget against which `compute_kept_products` caps retained factor Grams.
        /// @param[in] term Compiled AST node.
        /// @return Peak `n x n` buffer count without factor Gram retention.
        [[nodiscard]] static std::size_t unkept_buffers_of(const Term& term) noexcept
        {
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                std::size_t mx = 0;
                for (const auto& child : s->terms)
                {
                    mx = std::max(mx, unkept_buffers_of(child));
                }
                return mx;
            }
            if (const auto* p = std::get_if<ProductTerm>(&term))
            {
                std::size_t mx = 0;
                for (const auto& child : p->terms)
                {
                    mx = std::max(mx, unkept_buffers_of(child));
                }
                return p->terms.size() + 1U + mx;
            }
            return 1U;
        }

        /// @brief Computes `n x n` working buffers needed during `walk_weighted` for `term`.
        /// @details When `keeps == true`, a top-level product already has its varying factor Grams stored in
        /// `kept_bufs`, so the walk only needs 1 effective-weight buffer plus child walk buffers.
        /// @param[in] term Compiled AST node.
        /// @param[in] keeps True if this node's varying factor Grams were pre-retained by `apply_keeping`.
        /// @return Peak working buffer count required while walking `term`.
        [[nodiscard]] std::size_t node_buffers_of(const Term& term, bool keeps) const noexcept
        {
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                std::size_t mx = 0;
                for (const auto& child : s->terms)
                {
                    mx = std::max(mx, node_buffers_of(child, false));
                }
                return mx;
            }
            if (const auto* p = std::get_if<ProductTerm>(&term))
            {
                std::size_t mx = 0;
                for (const auto& child : p->terms)
                {
                    mx = std::max(mx, node_buffers_of(child, false));
                }
                if (keeps)
                {
                    // Retained factor Grams live in kept_bufs; only w_eff + recursive child work buffers are needed.
                    return 1U + mx;
                }
                return varying_factors(p->terms) + 1U + mx;
            }
            return 1U;
        }

        /// @brief Counts total `n x n` factor Gram slots retained for the first `products` multi-factor top-level
        /// products.
        /// @details Sums `varying_factors` over the first `products` top-level `ProductTerm` nodes with at least two
        /// varying factors.
        /// @param[in] products Number of eligible top-level `ProductTerm` nodes to retain.
        /// @return Total number of retained `n x n` factor Gram matrices.
        [[nodiscard]] std::size_t kept_grams_count(std::size_t products) const noexcept
        {
            std::size_t kept = 0;
            std::size_t grams = 0;
            for (const auto& t : root_terms())
            {
                if (const auto* p = std::get_if<ProductTerm>(&t))
                {
                    const std::size_t v = varying_factors(p->terms);
                    if (v >= 2U && kept < products)
                    {
                        ++kept;
                        grams += v;
                    }
                }
            }
            return grams;
        }

        /// @brief Computes peak working buffers required across all `root_terms()` when `products` top-level products
        /// retain factor Grams.
        /// @details Takes the maximum `node_buffers_of` across top-level terms, marking the first `products` eligible
        /// `ProductTerm` nodes as kept.
        /// @param[in] products Number of eligible top-level `ProductTerm` nodes whose factor Grams are kept.
        /// @return Peak scratch buffer depth needed by `walk_weighted` outside the kept Gram slots.
        [[nodiscard]] std::size_t walk_buffers_count(std::size_t products) const noexcept
        {
            std::size_t kept = 0;
            std::size_t mx = 0;
            for (const auto& t : root_terms())
            {
                bool keeps = false;
                if (const auto* p = std::get_if<ProductTerm>(&t))
                {
                    if (varying_factors(p->terms) >= 2U && kept < products)
                    {
                        ++kept;
                        keeps = true;
                    }
                }
                mx = std::max(mx, node_buffers_of(t, keeps));
            }
            return mx;
        }

        /// @brief Determines the maximum number of top-level multi-factor products to keep within the unkept buffer
        /// budget.
        /// @details Finds the largest `p <= candidates` satisfying `kept_grams_count(p) + walk_buffers_count(p) <=
        /// unkept_buffers_of(root)` so factor Gram retention never increases worst-case `n x n` buffer usage.
        /// @return Optimal count of top-level products whose factor Grams should be retained during forward evaluation.
        [[nodiscard]] std::size_t compute_kept_products() const noexcept
        {
            // Keep as many top-level multi-factor products as fit inside the unkept buffer budget.
            const std::size_t budget = unkept_buffers_of(root);
            std::size_t candidates = 0;
            for (const auto& t : root_terms())
            {
                if (const auto* p = std::get_if<ProductTerm>(&t))
                {
                    if (varying_factors(p->terms) >= 2U)
                    {
                        ++candidates;
                    }
                }
            }
            for (std::size_t p = candidates; p > 0U; --p)
            {
                if (kept_grams_count(p) + walk_buffers_count(p) <= budget)
                {
                    return p;
                }
            }
            return 0U;
        }

        /// @brief Evaluates the local derivative $\partial K_{\text{leaf}} / \partial \theta_{\text{local\_idx}}$
        /// using caches when available.
        /// @details Mirrors `eval_leaf_at` dispatch for gradients inside the `walk_weighted` adjoint pass.
        /// @param[in] leaf_idx Zero-based index into `leaves`.
        /// @param[in] x Input point matrix of shape `(n, d)`.
        /// @param[in] cache Optional precomputed distance and ARD cache.
        /// @param[out] d_k Output derivative matrix view.
        /// @param[in] local_idx Local hyperparameter index within `leaves[leaf_idx]`.
        /// @param[in] uplo Active matrix triangle to write.
        /// @param[in,out] nested Pre-allocated nested buffer pool for uncached `CustomKernel` distance construction.
        /// @param[in] level_idx Current nesting depth inside `nested`.
        /// @return Empty expected on success, or workspace/evaluation error.
        [[nodiscard]] std::expected<void, GprError> grad_leaf_at(
            std::size_t leaf_idx,
            MatrixRef<T> x,
            const DistanceCache<T>* cache,
            MatrixMut<T> d_k,
            std::size_t local_idx,
            Triangle uplo,
            NestedBuffers<T>& nested,
            std::size_t level_idx
        ) const
        {
            const KernelLeaf& kl = leaves[leaf_idx];
            if (cache != nullptr)
            {
                if (cache->has_dist() && leaf_reads_distances(kl))
                {
                    return detail::leaf_grad_dist(kl, cache->dist_view(), d_k, local_idx, uplo, exp_mode);
                }
                if (cache->has_ard() && leaf_needs_ard(kl))
                {
                    return detail::leaf_grad_ard_packed(
                        kl, cached_ls[leaf_idx], cache->ard_packed(), x.rows(), d_k, local_idx, uplo, exp_mode
                    );
                }
            }
            if (const auto* custom = std::get_if<CustomKernel>(&kl))
            {
                if (auto ok = ensure_nested(nested, level_idx, x.rows(), x.rows()); !ok.has_value())
                {
                    return ok;
                }
                auto dist_buf = nested.level(level_idx, x.rows(), x.rows());
                fill_square_sq_dist(x, dist_buf);
                return custom->template term<T>().grad(dist_buf.as_ref(), d_k, local_idx, uplo);
            }
            return detail::leaf_grad_points(kl, x, d_k, local_idx, uplo, exp_mode, cached_ls[leaf_idx]);
        }

        /// @brief Evaluates the training Gram matrix while saving kept factor Grams for `keep_plan` products.
        /// @details For the first `products` top-level multi-factor `ProductTerm` nodes, evaluates each non-constant
        /// factor once into `kept_bufs`, applies any folded `ConstantKernel` scale, and multiplies into the product
        /// accumulator before adding into `out`. Unkept terms evaluate normally via `eval_term_points`.
        /// @param[in] x Input point matrix of shape `(n, d)`.
        /// @param[in] cache Optional precomputed distance and ARD cache.
        /// @param[out] out Destination square Gram matrix of shape `(n, n)`.
        /// @param[in] uplo Active matrix triangle to populate.
        /// @param[in,out] scratch Reusable square scratch matrix.
        /// @param[in,out] nested Pre-allocated intermediate buffer pool for nested subtrees.
        /// @param[in,out] kept_bufs Pre-allocated retention buffer pool receiving kept factor Grams.
        /// @param[in] products Maximum number of top-level multi-factor products to retain.
        /// @param[out] ard_s2_buf Optional packed buffer capturing forward ARD $s^2$ slices.
        /// @param[in,out] ard_s2_ready Optional per-ARD-slot readiness flags set when `ard_s2_buf` is populated.
        /// @return Empty expected on success, or workspace/evaluation error.
        /// @retval GprErrorCode::WorkspaceTooSmall If `kept_bufs` or `nested` capacity is insufficient.
        [[nodiscard]] std::expected<void, GprError> eval_keeping_impl(
            MatrixRef<T> x,
            const DistanceCache<T>* cache,
            MatrixMut<T> out,
            Triangle uplo,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested,
            NestedBuffers<T>& kept_bufs,
            std::size_t products,
            std::span<T> ard_s2_buf = {},
            std::span<std::uint8_t> ard_s2_ready = {}
        ) const
        {
            const auto roots = root_terms();
            if (roots.empty())
            {
                return std::unexpected(GprError::unsupported_kernel_operation("composite kernel has no terms"));
            }
            if (!ard_s2_ready.empty())
            {
                std::ranges::fill(ard_s2_ready, std::uint8_t{0});
            }
            const std::size_t n = out.rows();
            std::size_t kept = 0;
            std::size_t offset = 0;
            for (std::size_t r_idx = 0; r_idx < roots.size(); ++r_idx)
            {
                const Term& t = roots[r_idx];
                MatrixMut<T> target = (r_idx == 0U) ? out : scratch;
                bool keeps = false;
                if (const auto* p = std::get_if<ProductTerm>(&t))
                {
                    const std::size_t v = varying_factors(p->terms);
                    if (v >= 2U && kept < products)
                    {
                        keeps = true;
                        ++kept;
                        // Fold ConstantKernel scalars into one multiplier and retain only varying factor Grams.
                        const T scale = constant_scale(p->terms);
                        std::size_t v_idx = 0;
                        for (const auto& f : p->terms)
                        {
                            if (is_constant_term(f))
                            {
                                continue;
                            }
                            if (kept_bufs.depth() <= (offset + v_idx) || kept_bufs.rows() < n || kept_bufs.cols() < n)
                            {
                                return std::unexpected(GprError::workspace_too_small());
                            }
                            auto kbuf = kept_bufs.level(offset + v_idx, n, n);
                            MatrixMut<T> tmp = scratch;
                            std::size_t sub_level = 0U;
                            // For r_idx > 0 and v_idx > 0, `target` (= scratch) already holds the running product,
                            // so composite factors must use nested level 0 instead of overwriting `scratch`.
                            if (r_idx > 0U && v_idx > 0U && !std::holds_alternative<LeafRef>(f))
                            {
                                if (auto ok = ensure_nested(nested, 0U, n, n); !ok.has_value())
                                {
                                    return ok;
                                }
                                tmp = nested.level(0U, n, n);
                                sub_level = 1U;
                            }
                            if (auto res = eval_term_points(
                                    f, x, kbuf, uplo, tmp, nested, sub_level, cache, ard_s2_buf, ard_s2_ready
                                );
                                !res.has_value())
                            {
                                return res;
                            }
                            if (v_idx == 0U)
                            {
                                copy_into(target, kbuf.as_ref(), uplo);
                                if (scale != T{1})
                                {
                                    scale_lower(target, scale);
                                }
                            }
                            else
                            {
                                mul_into(target, kbuf.as_ref(), uplo);
                            }
                            ++v_idx;
                        }
                        offset += v;
                    }
                }
                if (!keeps)
                {
                    if (r_idx == 0U || std::holds_alternative<LeafRef>(t))
                    {
                        if (auto res = eval_term_points(
                                t, x, target, uplo, scratch, nested, 0U, cache, ard_s2_buf, ard_s2_ready
                            );
                            !res.has_value())
                        {
                            return res;
                        }
                    }
                    else
                    {
                        if (auto ok = ensure_nested(nested, 0U, n, n); !ok.has_value())
                        {
                            return ok;
                        }
                        auto nbuf = nested.level(0U, n, n);
                        if (auto res = eval_term_points(
                                t, x, nbuf, uplo, scratch, nested, 1U, cache, ard_s2_buf, ard_s2_ready
                            );
                            !res.has_value())
                        {
                            return res;
                        }
                        add_into(out, nbuf.as_ref(), uplo);
                        continue;
                    }
                }
                if (r_idx > 0U)
                {
                    add_into(out, target.as_ref(), uplo);
                }
            }
            return {};
        }

        /// @brief Captures a `std::expected<double, GprError>` into `out_val` or `err` out-of-line.
        /// @param[in] res Expected scalar result.
        /// @param[out] out_val Destination value written on success.
        /// @param[out] err Destination error slot written on failure.
        /// @return True on success, false after writing `err`.
        [[nodiscard]] static CXXGPR_NOINLINE bool capture_expected_double(
            std::expected<double, GprError>&& res, double& out_val, std::optional<GprError>& err
        ) noexcept
        {
            if (res.has_value())
            {
                out_val = *res;
                return true;
            }
            err = std::move(res).error();
            return false;
        }

        /// @brief Invariant context for `walk_weighted_rec` to keep recursive frames minimal.
        struct WalkWeightedCtx
        {
            /// @brief Input point matrix.
            MatrixRef<T> x;
            /// @brief Optional precomputed distance/ARD cache.
            const DistanceCache<T>* cache;
            /// @brief Optional forward-captured packed ARD $s^2$ storage.
            std::span<const T> ard_s2_buf;
            /// @brief Optional readiness flags for `ard_s2_buf`.
            std::span<const std::uint8_t> ard_s2_ready;
            /// @brief Full output gradient span indexed by `grad_offset`.
            std::span<double> out_grads;
            /// @brief Reusable `n x n` scratch matrix for leaf evaluation.
            MatrixMut<T> kernel_scratch;
            /// @brief Nested buffer pool for `eval_term_points_rec`.
            NestedBuffers<T>& nested;
            /// @brief Working buffer pool for adjoint effective weights and intermediate factor Grams.
            NestedBuffers<T>& weighted_bufs;
            /// @brief Scratch span for deterministic OpenMP column reductions.
            std::span<double> col_partials;
            /// @brief Root product pointer when pre-kept factor Grams are active, otherwise nullptr.
            const Term* kept_root;
            /// @brief Starting buffer slot for `kept_root`.
            std::size_t kept_offset;
            /// @brief Count of pre-retained factor Grams for `kept_root`.
            std::size_t kept_count;
            /// @brief Out-of-line error slot.
            std::optional<GprError>& err;
        };

        /// @brief Evaluates one weighted leaf adjoint contraction out-of-line so recursive frames stay minimal.
        /// @param[in] leaf Compiled leaf reference.
        /// @param[in] weight Lower-triangular adjoint weight matrix `W`.
        /// @param[out] out_grads Output gradient slice for `leaf`.
        /// @param[in] need_value True when caller requires $\langle W, K_{\text{leaf}} \rangle_F$.
        /// @param[in] ctx Invariant walk context.
        /// @param[out] out_val Contracted scalar value on success.
        /// @return True on success, false after writing `ctx.err`.
        [[nodiscard]] CXXGPR_NOINLINE bool walk_weighted_leaf_step(
            const LeafRef& leaf,
            const MatrixRef<T>& weight,
            std::span<double> out_grads,
            bool need_value,
            const WalkWeightedCtx& ctx,
            double& out_val
        ) const
        {
            const KernelLeaf& kl = leaves[leaf.leaf_idx];
            if (const auto* ck = std::get_if<ConstantKernel>(&kl))
            {
                const double val = ck->constant() * lower_sum(weight, ctx.col_partials);
                out_grads[0] = val;
                out_val = val;
                return true;
            }
            if (const auto* wk = std::get_if<WhiteKernel>(&kl))
            {
                const double val = wk->variance() * lower_diag_sum(weight);
                out_grads[0] = val;
                out_val = need_value ? val : 0.0;
                return true;
            }
            if (leaf_needs_ard(kl))
            {
                const std::span<const T> ard_packed =
                    (ctx.cache != nullptr && ctx.cache->has_ard()) ? ctx.cache->ard_packed() : std::span<const T>{};
                std::span<const T> precomputed_s2{};
                const std::size_t slot = leaf_ard_slots[leaf.leaf_idx];
                const std::size_t per_dim = (ctx.x.rows() * (ctx.x.rows() + 1U)) / 2U;
                if (slot < ctx.ard_s2_ready.size() && ctx.ard_s2_ready[slot] != 0U && per_dim > 0U
                    && (ctx.ard_s2_buf.size() / per_dim) > slot)
                {
                    precomputed_s2 = ctx.ard_s2_buf.subspan(slot * per_dim, per_dim);
                }
                return capture_expected_double(
                    detail::leaf_ard_weighted_grads(
                        kl, cached_ls[leaf.leaf_idx], ctx.x, ard_packed, weight, ctx.kernel_scratch, ctx.col_partials,
                        out_grads, need_value, exp_mode, precomputed_s2
                    ),
                    out_val, ctx.err
                );
            }
            if (std::holds_alternative<RbfKernel>(kl) || std::holds_alternative<MaternKernel>(kl)
                || std::holds_alternative<PeriodicKernel>(kl) || std::holds_alternative<RationalQuadraticKernel>(kl)
                || std::holds_alternative<LinearKernel>(kl))
            {
                const MatrixRef<T> dist_view =
                    (ctx.cache != nullptr && ctx.cache->has_dist()) ? ctx.cache->dist_view() : MatrixRef<T>{};
                return capture_expected_double(
                    detail::leaf_iso_weighted_grads(
                        kl, ctx.x, dist_view, weight, ctx.kernel_scratch, ctx.col_partials, out_grads, need_value,
                        exp_mode
                    ),
                    out_val, ctx.err
                );
            }
            for (std::size_t p = 0; p < leaf.param_count; ++p)
            {
                if (!capture_expected(
                        grad_leaf_at(
                            leaf.leaf_idx, ctx.x, ctx.cache, ctx.kernel_scratch, p, Triangle::Lower, ctx.nested, 0
                        ),
                        ctx.err
                    ))
                {
                    return false;
                }
                out_grads[p] = lower_frobenius(weight, ctx.kernel_scratch.as_ref(), ctx.col_partials);
            }
            if (need_value)
            {
                if (!capture_expected(
                        eval_leaf_at(
                            leaf.leaf_idx, ctx.x, ctx.kernel_scratch, Triangle::Lower, ctx.nested, 0, ctx.cache
                        ),
                        ctx.err
                    ))
                {
                    return false;
                }
                out_val = lower_frobenius(weight, ctx.kernel_scratch.as_ref(), ctx.col_partials);
                return true;
            }
            out_val = 0.0;
            return true;
        }

        /// @brief Evaluates all-constant product adjoint contractions out-of-line.
        /// @param[in] p Product term containing only constant factors.
        /// @param[in] weight Lower-triangular adjoint weight matrix.
        /// @param[in] scale Product of all constant factor values.
        /// @param[in] grad_offset Base offset into `ctx.out_grads`.
        /// @param[in] ctx Invariant walk context.
        /// @param[out] out_val Contracted scalar value.
        CXXGPR_NOINLINE void walk_weighted_const_only_product(
            const ProductTerm& p,
            const MatrixRef<T>& weight,
            T scale,
            std::size_t grad_offset,
            const WalkWeightedCtx& ctx,
            double& out_val
        ) const
        {
            const double val = static_cast<double>(scale) * lower_sum(weight, ctx.col_partials);
            std::size_t p_off = 0U;
            for (const auto& f : p.terms)
            {
                const auto [fb, fe] = param_range(f);
                ctx.out_grads[grad_offset + p_off] = val;
                p_off += (fe - fb);
            }
            out_val = val;
        }

        /// @brief Applies constant scaling and fills constant factor gradients for a single-varying product
        /// out-of-line.
        /// @param[in] p Product term with one varying factor.
        /// @param[in] scale Product of constant factor values.
        /// @param[in] grad_offset Base offset into `ctx.out_grads`.
        /// @param[in] varying_off Offset of the single varying factor within `p`.
        /// @param[in] varying_count Parameter count of the single varying factor.
        /// @param[in] has_const True if `p` has at least one constant factor.
        /// @param[in] ctx Invariant walk context.
        /// @param[in,out] val Contracted scalar value scaled in place.
        CXXGPR_NOINLINE void walk_weighted_scale_and_fill_consts(
            const ProductTerm& p,
            T scale,
            std::size_t grad_offset,
            std::size_t varying_off,
            std::size_t varying_count,
            bool has_const,
            const WalkWeightedCtx& ctx,
            double& val
        ) const
        {
            if (has_const && scale != T{1})
            {
                const auto scale_d = static_cast<double>(scale);
                val *= scale_d;
                for (double& g : ctx.out_grads.subspan(grad_offset + varying_off, varying_count))
                {
                    g *= scale_d;
                }
            }
            if (has_const)
            {
                std::size_t c_off = 0U;
                for (const auto& f : p.terms)
                {
                    const auto [fb, fe] = param_range(f);
                    if (is_constant_term(f))
                    {
                        ctx.out_grads[grad_offset + c_off] = val;
                    }
                    c_off += (fe - fb);
                }
            }
        }

        /// @brief Evaluates sibling factor Grams and validates workspace slots for a multi-factor product out-of-line.
        /// @param[in] p Product term with two or more varying factors.
        /// @param[in] g_base Base buffer slot in `ctx.weighted_bufs`.
        /// @param[in] next_cursor Buffer slot reserved for `w_eff`.
        /// @param[in] varying Number of varying factors in `p`.
        /// @param[in] use_kept True when factor Grams were pre-retained in `ctx.weighted_bufs`.
        /// @param[in] ctx Invariant walk context.
        /// @return True on success, false after writing `ctx.err`.
        [[nodiscard]] CXXGPR_NOINLINE bool walk_weighted_prepare_factors(
            const ProductTerm& p,
            std::size_t g_base,
            std::size_t next_cursor,
            std::size_t varying,
            bool use_kept,
            const WalkWeightedCtx& ctx
        ) const
        {
            const std::size_t n = ctx.x.rows();
            if (!use_kept)
            {
                std::size_t v_idx = 0U;
                for (const auto& f : p.terms)
                {
                    if (is_constant_term(f))
                    {
                        continue;
                    }
                    if (!ensure_nested_ok(ctx.weighted_bufs, g_base + v_idx, n, n, ctx.err))
                    {
                        return false;
                    }
                    const auto g_buf = ctx.weighted_bufs.level(g_base + v_idx, n, n);
                    if (!eval_term_points_rec(
                            f, ctx.x, g_buf, Triangle::Lower, ctx.kernel_scratch, ctx.nested, 0, ctx.cache, ctx.err
                        ))
                    {
                        return false;
                    }
                    ++v_idx;
                }
            }
            else if (ctx.kept_count < varying)
            {
                return fail_workspace_too_small(ctx.err);
            }
            return ensure_nested_ok(ctx.weighted_bufs, next_cursor, n, n, ctx.err);
        }

        /// @brief Builds the effective adjoint weight matrix `w_eff` in slot `next_cursor` out-of-line.
        /// @param[in] weight Parent adjoint weight matrix.
        /// @param[in] scale Product of constant factor values.
        /// @param[in] g_base Base buffer slot holding varying factor Grams.
        /// @param[in] next_cursor Buffer slot receiving `w_eff`.
        /// @param[in] varying Number of varying factors.
        /// @param[in] cur_v Index of the excluded factor being differentiated.
        /// @param[in] ctx Invariant walk context.
        /// @return Read-only view of the populated `w_eff` buffer.
        [[nodiscard]] CXXGPR_NOINLINE MatrixRef<T> walk_weighted_build_weff(
            const MatrixRef<T>& weight,
            T scale,
            std::size_t g_base,
            std::size_t next_cursor,
            std::size_t varying,
            std::size_t cur_v,
            const WalkWeightedCtx& ctx
        ) const
        {
            const std::size_t n = ctx.x.rows();
            const auto w_eff = ctx.weighted_bufs.level(next_cursor, n, n);
            copy_into(w_eff, weight, Triangle::Lower);
            if (scale != T{1})
            {
                scale_lower(w_eff, scale);
            }
            for (std::size_t r = 0U; r < varying; ++r)
            {
                if (r == cur_v)
                {
                    continue;
                }
                mul_into(w_eff, ctx.weighted_bufs.level(g_base + r, n, n).as_ref(), Triangle::Lower);
            }
            return w_eff.as_ref();
        }

        /// @brief Contracts the multi-factor product value and fills constant factor gradients out-of-line.
        /// @param[in] p Product term with two or more varying factors.
        /// @param[in] weight Parent adjoint weight matrix.
        /// @param[in] scale Product of constant factor values.
        /// @param[in] g_base Base buffer slot holding varying factor Grams.
        /// @param[in] next_cursor Buffer slot reused as temporary `w_eff`.
        /// @param[in] varying Number of varying factors.
        /// @param[in] grad_offset Base offset into `ctx.out_grads`.
        /// @param[in] has_const True if `p` contains constant factors.
        /// @param[in] ctx Invariant walk context.
        /// @return Contracted scalar value $\langle W, K_{\text{prod}} \rangle_F$.
        [[nodiscard]] CXXGPR_NOINLINE double walk_weighted_finish_product(
            const ProductTerm& p,
            const MatrixRef<T>& weight,
            T scale,
            std::size_t g_base,
            std::size_t next_cursor,
            std::size_t varying,
            std::size_t grad_offset,
            bool has_const,
            const WalkWeightedCtx& ctx
        ) const
        {
            const std::size_t n = ctx.x.rows();
            const auto w_eff = ctx.weighted_bufs.level(next_cursor, n, n);
            copy_into(w_eff, weight, Triangle::Lower);
            if (scale != T{1})
            {
                scale_lower(w_eff, scale);
            }
            for (std::size_t r = 1U; r < varying; ++r)
            {
                mul_into(w_eff, ctx.weighted_bufs.level(g_base + r, n, n).as_ref(), Triangle::Lower);
            }
            const double val =
                lower_frobenius(w_eff.as_ref(), ctx.weighted_bufs.level(g_base, n, n).as_ref(), ctx.col_partials);
            if (has_const)
            {
                std::size_t c_off = 0U;
                for (const auto& f : p.terms)
                {
                    const auto [fb, fe] = param_range(f);
                    if (is_constant_term(f))
                    {
                        ctx.out_grads[grad_offset + c_off] = val;
                    }
                    c_off += (fe - fb);
                }
            }
            return val;
        }

        /// @brief Recursive body of `walk_weighted` with minimal per-level stack frame size.
        /// @param[in] term Compiled AST node.
        /// @param[in] weight Lower-triangular adjoint weight matrix `W`.
        /// @param[in] grad_offset Starting offset inside `ctx.out_grads` for `term`.
        /// @param[in] walk_cursor Next free buffer slot in `ctx.weighted_bufs`.
        /// @param[in] need_value True when caller requires $\langle W, K_{\text{term}} \rangle_F$.
        /// @param[in] ctx Invariant walk context.
        /// @param[out] out_val Contracted scalar value on success.
        /// @return True on success, false after writing `ctx.err`.
        [[nodiscard]] bool walk_weighted_rec(
            const Term& term,
            const MatrixRef<T>& weight,
            std::size_t grad_offset,
            std::size_t walk_cursor,
            bool need_value,
            const WalkWeightedCtx& ctx,
            double& out_val
        ) const
        {
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                std::size_t p_off = 0U;
                double value = 0.0;
                for (const auto& child : s->terms)
                {
                    const auto [cb, ce] = param_range(child);
                    double child_val = 0.0;
                    if (!walk_weighted_rec(child, weight, grad_offset + p_off, walk_cursor, need_value, ctx, child_val))
                    {
                        return false;
                    }
                    value += child_val;
                    p_off += (ce - cb);
                }
                out_val = value;
                return true;
            }
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return walk_weighted_leaf_step(
                    *leaf, weight, ctx.out_grads.subspan(grad_offset, leaf->param_count), need_value, ctx, out_val
                );
            }

            const auto& p = std::get<ProductTerm>(term);
            const std::size_t varying = varying_factors(p.terms);
            const T scale = constant_scale(p.terms);
            const bool has_const = (varying < p.terms.size());

            if (varying == 0U)
            {
                walk_weighted_const_only_product(p, weight, scale, grad_offset, ctx, out_val);
                return true;
            }

            if (varying == 1U)
            {
                double val = 0.0;
                std::size_t p_off = 0U;
                std::size_t varying_off = 0U;
                std::size_t varying_count = 0U;
                for (const auto& f : p.terms)
                {
                    const auto [fb, fe] = param_range(f);
                    const std::size_t count = fe - fb;
                    if (!is_constant_term(f))
                    {
                        varying_off = p_off;
                        varying_count = count;
                        if (!walk_weighted_rec(
                                f, weight, grad_offset + p_off, walk_cursor, need_value || has_const, ctx, val
                            ))
                        {
                            return false;
                        }
                    }
                    p_off += count;
                }
                walk_weighted_scale_and_fill_consts(
                    p, scale, grad_offset, varying_off, varying_count, has_const, ctx, val
                );
                out_val = val;
                return true;
            }

            const bool use_kept = (ctx.kept_root == &term);
            const std::size_t g_base = use_kept ? ctx.kept_offset : walk_cursor;
            const std::size_t next_cursor = use_kept ? walk_cursor : (walk_cursor + varying);
            if (!walk_weighted_prepare_factors(p, g_base, next_cursor, varying, use_kept, ctx))
            {
                return false;
            }

            std::size_t p_off = 0U;
            std::size_t cur_v = 0U;
            for (const auto& f : p.terms)
            {
                const auto [fb, fe] = param_range(f);
                if (!is_constant_term(f))
                {
                    const MatrixRef<T> w_eff_ref =
                        walk_weighted_build_weff(weight, scale, g_base, next_cursor, varying, cur_v, ctx);
                    double ignored_val = 0.0;
                    if (!walk_weighted_rec(
                            f, w_eff_ref, grad_offset + p_off, next_cursor + 1U, false, ctx, ignored_val
                        ))
                    {
                        return false;
                    }
                    ++cur_v;
                }
                p_off += (fe - fb);
            }

            out_val = (need_value || has_const)
                ? walk_weighted_finish_product(
                      p, weight, scale, g_base, next_cursor, varying, grad_offset, has_const, ctx
                  )
                : 0.0;
            return true;
        }

        /// @brief Computes weighted parameter gradients $\langle W, \partial K / \partial \theta \rangle_F$ for `term`.
        /// @param[in] term Compiled AST node.
        /// @param[in] x Input point matrix.
        /// @param[in] cache Optional precomputed distance/ARD cache.
        /// @param[in] weight Lower-triangular adjoint weight matrix `W`.
        /// @param[out] out_grads Output gradient span for `term`.
        /// @param[in,out] kernel_scratch Reusable `n x n` scratch matrix.
        /// @param[in,out] nested Nested buffer pool.
        /// @param[in,out] weighted_bufs Working buffer pool for adjoint weights and intermediate Grams.
        /// @param[in] kept_offset Starting slot in `weighted_bufs` when `has_kept` is true.
        /// @param[in] kept_count Number of pre-retained factor Grams when `has_kept` is true.
        /// @param[in] has_kept True when `apply_keeping` pre-retained this product's factor Grams.
        /// @param[in] walk_cursor Next free buffer slot in `weighted_bufs`.
        /// @param[in,out] col_partials Scratch span for deterministic OpenMP reductions.
        /// @param[in] need_value True when caller requires $\langle W, K_{\text{term}} \rangle_F$.
        /// @param[in] ard_s2_buf Optional forward-captured ARD $s^2$ storage.
        /// @param[in] ard_s2_ready Optional readiness bytes for `ard_s2_buf`.
        /// @return Contracted term value, or GprError on failure.
        [[nodiscard]] std::expected<double, GprError> walk_weighted(
            const Term& term,
            MatrixRef<T> x,
            const DistanceCache<T>* cache,
            MatrixRef<T> weight,
            std::span<double> out_grads,
            MatrixMut<T> kernel_scratch,
            NestedBuffers<T>& nested,
            NestedBuffers<T>& weighted_bufs,
            std::size_t kept_offset,
            std::size_t kept_count,
            bool has_kept,
            std::size_t walk_cursor,
            std::span<double> col_partials,
            bool need_value,
            std::span<const T> ard_s2_buf = {},
            std::span<const std::uint8_t> ard_s2_ready = {}
        ) const
        {
            std::optional<GprError> err;
            const WalkWeightedCtx ctx{
                .x = x,
                .cache = cache,
                .ard_s2_buf = ard_s2_buf,
                .ard_s2_ready = ard_s2_ready,
                .out_grads = out_grads,
                .kernel_scratch = kernel_scratch,
                .nested = nested,
                .weighted_bufs = weighted_bufs,
                .col_partials = col_partials,
                .kept_root = has_kept ? &term : nullptr,
                .kept_offset = kept_offset,
                .kept_count = kept_count,
                .err = err,
            };
            double val = 0.0;
            if (!walk_weighted_rec(term, weight, 0U, walk_cursor, need_value, ctx, val))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return val;
        }
    };

    // --- CompiledKernel<T> public methods ---

    /// @brief Constructs a compiled kernel from its private `Impl` state.
    /// @details Internal constructor invoked by `CompiledKernel::from_spec`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] impl Owning pointer to compiled AST state.
    template <KernelScalarConcept T>
    CompiledKernel<T>::CompiledKernel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl))
    {
    }

    /// @brief Move-constructs a `CompiledKernel<T>`.
    /// @details Transfers ownership of the internal `Impl` pointer without copying leaves or bindings.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] other Source compiled kernel moved from.
    template <KernelScalarConcept T> CompiledKernel<T>::CompiledKernel(CompiledKernel&& other) noexcept = default;

    /// @brief Move-assigns a `CompiledKernel<T>`.
    /// @details Transfers ownership of the internal `Impl` pointer without copying leaves or bindings.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] other Source compiled kernel moved from.
    /// @return Reference to `*this`.
    template <KernelScalarConcept T>
    CompiledKernel<T>& CompiledKernel<T>::operator=(CompiledKernel&& other) noexcept = default;

    /// @brief Destroys the `CompiledKernel<T>` and its private `Impl`.
    /// @details Defined out-of-line where `CompiledKernel<T>::Impl` is a complete type.
    /// @tparam T Storage floating-point scalar type.
    template <KernelScalarConcept T> CompiledKernel<T>::~CompiledKernel() = default;

    /// @brief Copy-constructs a `CompiledKernel<T>` with independent mutable leaf state.
    /// @details Deep-copies the compiled `Impl` AST, leaves, bindings, and cached lengthscales.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] other Compiled kernel instance to clone.
    template <KernelScalarConcept T>
    CompiledKernel<T>::CompiledKernel(const CompiledKernel& other) : impl_(std::make_unique<Impl>(*other.impl_))
    {
    }

    /// @brief Copy-assigns a `CompiledKernel<T>` with independent mutable leaf state.
    /// @details Deep-copies `other`'s `Impl` state while guarding against self-assignment.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] other Compiled kernel instance to clone.
    /// @return Reference to `*this`.
    template <KernelScalarConcept T> CompiledKernel<T>& CompiledKernel<T>::operator=(const CompiledKernel& other)
    {
        if (this != &other)
        {
            impl_ = std::make_unique<Impl>(*other.impl_);
        }
        return *this;
    }

    /// @brief Compiles a `KernelSpec` expression tree into a flattened `CompiledKernel<T>`.
    /// @details Flattens associative sums and products, derives `CoordMode`, `reads_distances`, `needs_ard_sq_diff`,
    /// `nested_depth` (adding one level when `CustomKernel` requires a temporary squared-distance matrix), and
    /// populates cached ARD lengthscale vectors.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] spec Declaration-layer kernel expression tree.
    /// @return Compiled execution-layer kernel instance.
    template <KernelScalarConcept T> CompiledKernel<T> CompiledKernel<T>::from_spec(const KernelSpec& spec)
    {
        auto impl = std::make_unique<Impl>();
        std::size_t next_param_idx = 0;
        impl->root = Impl::compile_node(spec, impl->leaves, impl->bindings, next_param_idx);
        // Infer tree-wide coordinate mode and cache compatibility flags across all compiled leaves.
        CoordMode mode = CoordMode::Either;
        bool has_custom_leaf = false;
        bool reads_dist = false;
        bool needs_ard = false;
        for (const auto& leaf : impl->leaves)
        {
            mode = combine_coord_mode(mode, detail::leaf_coord_mode(leaf));
            if (std::holds_alternative<CustomKernel>(leaf))
            {
                has_custom_leaf = true;
            }
            if (leaf_reads_distances(leaf))
            {
                reads_dist = true;
            }
            if (leaf_needs_ard(leaf))
            {
                needs_ard = true;
            }
        }
        impl->coord_mode = mode;
        impl->reads_dist = reads_dist;
        impl->needs_ard = needs_ard;
        // Reserve one extra nested buffer level when any leaf is CustomKernel so point paths can materialize dist.
        impl->nested_depth = Impl::compute_nested_depth(impl->root) + (has_custom_leaf ? 1U : 0U);
        impl->refresh_cached_ls();
        impl->leaf_dims.assign(impl->leaves.size(), 0U);
        impl->leaf_ard_slots.assign(impl->leaves.size(), Impl::k_no_ard_slot);
        std::size_t next_ard_slot = 0U;
        for (std::size_t i = 0U; i < impl->leaves.size(); ++i)
        {
            if (leaf_needs_ard(impl->leaves[i]))
            {
                impl->leaf_dims[i] = impl->cached_ls[i].size();
                impl->leaf_ard_slots[i] = next_ard_slot++;
            }
        }
        impl->ard_leaf_count = next_ard_slot;
        return CompiledKernel<T>(std::move(impl));
    }

    /// @brief Returns the exponential evaluation backend used by built-in leaves.
    /// @details Defaults to `KernelExp::Accurate`.
    /// @tparam T Storage floating-point scalar type.
    /// @return Active `KernelExp` mode.
    template <KernelScalarConcept T> KernelExp CompiledKernel<T>::kernel_exp() const noexcept
    {
        return impl_->exp_mode;
    }

    /// @brief Sets the exponential evaluation backend used by built-in leaves.
    /// @details Selects between exact `std::exp` (`Accurate`) and polynomial approximation (`FastApprox`).
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] mode Exponential evaluation backend to use.
    template <KernelScalarConcept T> void CompiledKernel<T>::set_kernel_exp(KernelExp mode) noexcept
    {
        impl_->exp_mode = mode;
    }

    /// @brief Returns the total number of log-space hyperparameters in the compiled kernel.
    /// @details Equals `parameter_bindings().size()`.
    /// @tparam T Storage floating-point scalar type.
    /// @return Total hyperparameter count.
    template <KernelScalarConcept T> std::size_t CompiledKernel<T>::num_params() const noexcept
    {
        return impl_->bindings.size();
    }

    /// @brief Returns the number of compiled kernel leaves in depth-first left-to-right order.
    /// @details Counts every leaf occurrence in the flattened AST.
    /// @tparam T Storage floating-point scalar type.
    /// @return Number of leaf nodes.
    template <KernelScalarConcept T> std::size_t CompiledKernel<T>::num_leaves() const noexcept
    {
        return impl_->leaves.size();
    }

    /// @brief Returns the input coordinate mode required by the compiled kernel tree.
    /// @details Reflects the combined requirements across all leaf kernels (`Dist`, `Points`, `Either`, or `Mixed`).
    /// @tparam T Storage floating-point scalar type.
    /// @return Combined `CoordMode`.
    template <KernelScalarConcept T> CoordMode CompiledKernel<T>::coord_mode() const noexcept
    {
        return impl_->coord_mode;
    }

    /// @brief Returns the minimum number of `NestedBuffers<T>` levels required for tree evaluation.
    /// @details Callers allocating workspace ahead of time use this depth to guarantee zero hot-path allocations.
    /// @tparam T Storage floating-point scalar type.
    /// @return Required nested buffer count.
    template <KernelScalarConcept T> std::size_t CompiledKernel<T>::nested_depth() const noexcept
    {
        return impl_->nested_depth;
    }

    /// @brief Returns true if any leaf in the tree can consume a scalar squared-distance cache.
    /// @details Used by `DistanceCache` construction to skip allocating unused scalar distance buffers.
    /// @tparam T Storage floating-point scalar type.
    /// @return True when scalar squared distances are read by at least one leaf.
    template <KernelScalarConcept T> bool CompiledKernel<T>::reads_distances() const noexcept
    {
        return impl_->reads_dist;
    }

    /// @brief Returns true if any leaf in the tree can consume packed ARD squared-difference caches.
    /// @details Used by `DistanceCache` construction to skip allocating unused ARD packed buffers.
    /// @tparam T Storage floating-point scalar type.
    /// @return True when packed ARD squared differences are read by at least one leaf.
    template <KernelScalarConcept T> bool CompiledKernel<T>::needs_ard_sq_diff() const noexcept
    {
        return impl_->needs_ard;
    }

    /// @brief Returns the number of ARD kernel leaves in the compiled tree.
    /// @tparam T Storage floating-point scalar type.
    /// @return Count of ARD leaves.
    template <KernelScalarConcept T> std::size_t CompiledKernel<T>::ard_leaf_count() const noexcept
    {
        return impl_->ard_leaf_count;
    }

    /// @brief Returns per-leaf ARD dimension counts in depth-first leaf order (`0` for non-ARD leaves).
    /// @tparam T Storage floating-point scalar type.
    /// @return Immutable span of length `num_leaves()`.
    template <KernelScalarConcept T> std::span<const std::size_t> CompiledKernel<T>::ard_dims_per_leaf() const noexcept
    {
        return impl_->leaf_dims;
    }

    /// @brief Allocates a `NestedBuffers<T>` workspace sized for this compiled kernel and matrix shape `(rows, cols)`.
    /// @details Pre-allocates `nested_depth()` contiguous matrices.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] rows Matrix row capacity.
    /// @param[in] cols Matrix column capacity.
    /// @return Pre-sized `NestedBuffers<T>` instance.
    template <KernelScalarConcept T>
    NestedBuffers<T> CompiledKernel<T>::make_nested_buffers(std::size_t rows, std::size_t cols) const
    {
        return NestedBuffers<T>(impl_->nested_depth, rows, cols);
    }

    /// @brief Writes all flattened log-space hyperparameters into `out`.
    /// @details Collects parameter slices across `leaves` in depth-first left-to-right order.
    /// @tparam T Storage floating-point scalar type.
    /// @param[out] out Destination span of length `num_params()`.
    /// @return Empty expected on success, or length mismatch error.
    /// @retval GprErrorCode::LengthMismatch If `out.size() != num_params()`.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::get_params(std::span<double> out) const
    {
        if (auto cnt = require_count(out.size(), impl_->bindings.size(), "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        std::size_t offset = 0;
        for (const auto& leaf : impl_->leaves)
        {
            const std::size_t count = detail::leaf_num_params(leaf);
            if (auto res = detail::leaf_get_params(leaf, out.subspan(offset, count)); !res.has_value())
            {
                return res;
            }
            offset += count;
        }
        return {};
    }

    /// @brief Atomically updates all leaf hyperparameters from log-space `params`.
    /// @details Validates and updates a staged copy of `leaves` first, committing `impl_->leaves` and refreshing
    /// `cached_ls` only if every leaf accepts its parameter slice.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] params Log-space hyperparameter span of length `num_params()`.
    /// @return Empty expected on success, or length/bounds error leaving state unchanged.
    /// @retval GprErrorCode::LengthMismatch If `params.size() != num_params()`.
    /// @retval GprErrorCode::InvalidHyperparameter If any value violates its open parameter interval.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::set_params(std::span<const double> params)
    {
        if (auto res = detail::set_leaves_params_atomic(impl_->leaves, params, impl_->bindings.size());
            !res.has_value())
        {
            return res;
        }
        impl_->refresh_cached_ls();
        return {};
    }

    /// @brief Writes open hyperparameter intervals for all bound parameters into `out`.
    /// @details Iterates over `leaves` in depth-first left-to-right order.
    /// @tparam T Storage floating-point scalar type.
    /// @param[out] out Destination span of length `num_params()`.
    /// @return Empty expected on success, or length mismatch error.
    /// @retval GprErrorCode::LengthMismatch If `out.size() != num_params()`.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::bounds_into(std::span<Interval> out) const
    {
        if (auto cnt = require_count(out.size(), impl_->bindings.size(), "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        std::size_t offset = 0;
        for (const auto& leaf : impl_->leaves)
        {
            const std::size_t count = detail::leaf_num_params(leaf);
            if (auto res = detail::leaf_bounds_into(leaf, out.subspan(offset, count)); !res.has_value())
            {
                return res;
            }
            offset += count;
        }
        return {};
    }

    /// @brief Returns the flat list of parameter-to-leaf bindings for this compiled kernel.
    /// @details Entries are ordered consecutively by global index `[0, num_params())`.
    /// @tparam T Storage floating-point scalar type.
    /// @return Const reference to the parameter binding vector.
    template <KernelScalarConcept T>
    const std::vector<ParameterBinding>& CompiledKernel<T>::parameter_bindings() const noexcept
    {
        return impl_->bindings;
    }

    /// @brief Evaluates the square Gram matrix from squared Euclidean distances using caller-supplied buffers.
    /// @details Checks non-aliasing scratch, validates distance compatibility and finiteness, and evaluates `root`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] dist Input square squared-distance matrix.
    /// @param[out] out Destination square Gram matrix view.
    /// @param[in] uplo Active matrix triangle (`Lower`, `Upper`, or `Full`).
    /// @param[in,out] scratch Non-aliasing scratch matrix matching `out` shape.
    /// @param[in,out] nested Pre-allocated nested buffer pool for composite subtrees.
    /// @return Empty expected on success, or validation/workspace/evaluation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply(
        MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo, MatrixMut<T> scratch, NestedBuffers<T>& nested
    ) const
    {
        if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value())
        {
            return sc;
        }
        if (auto v = impl_->validate_dist_call(dist, out.as_ref(), uplo, true); !v.has_value())
        {
            return v;
        }
        return impl_->eval_term_dist(impl_->root, dist, out, uplo, scratch, nested, 0);
    }

    /// @brief Evaluates the square Gram matrix from squared Euclidean distances using temporary nested buffers.
    /// @details Convenience overload allocating `make_nested_buffers(out.rows(), out.cols())`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] dist Input square squared-distance matrix.
    /// @param[out] out Destination square Gram matrix view.
    /// @param[in] uplo Active matrix triangle (`Lower`, `Upper`, or `Full`).
    /// @param[in,out] scratch Non-aliasing scratch matrix matching `out` shape.
    /// @return Empty expected on success, or validation/evaluation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply(
        MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo, MatrixMut<T> scratch
    ) const
    {
        auto nested = make_nested_buffers(out.rows(), out.cols());
        return apply(dist, out, uplo, scratch, nested);
    }

    /// @brief Evaluates the square training Gram matrix from point coordinates using caller-supplied buffers.
    /// @details Checks scratch non-aliasing, validates coordinates and ARD dimensions, and evaluates `root`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point coordinate matrix of shape `(n, d)`.
    /// @param[out] out Destination square Gram matrix of shape `(n, n)`.
    /// @param[in] uplo Active matrix triangle (`Lower`, `Upper`, or `Full`).
    /// @param[in,out] scratch Non-aliasing scratch matrix of shape `(n, n)`.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @return Empty expected on success, or validation/workspace/evaluation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_points(
        MatrixRef<T> x, MatrixMut<T> out, Triangle uplo, MatrixMut<T> scratch, NestedBuffers<T>& nested
    ) const
    {
        if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value())
        {
            return sc;
        }
        if (auto v = impl_->validate_points_call(x, out.as_ref()); !v.has_value())
        {
            return v;
        }
        return impl_->eval_term_points(impl_->root, x, out, uplo, scratch, nested, 0);
    }

    /// @brief Evaluates the square training Gram matrix from point coordinates using temporary nested buffers.
    /// @details Convenience overload allocating `make_nested_buffers(out.rows(), out.cols())`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point coordinate matrix of shape `(n, d)`.
    /// @param[out] out Destination square Gram matrix of shape `(n, n)`.
    /// @param[in] uplo Active matrix triangle (`Lower`, `Upper`, or `Full`).
    /// @param[in,out] scratch Non-aliasing scratch matrix of shape `(n, n)`.
    /// @return Empty expected on success, or validation/evaluation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_points(
        MatrixRef<T> x, MatrixMut<T> out, Triangle uplo, MatrixMut<T> scratch
    ) const
    {
        auto nested = make_nested_buffers(out.rows(), out.cols());
        return apply_points(x, out, uplo, scratch, nested);
    }

    /// @brief Evaluates the rectangular cross-covariance matrix from squared distances using caller buffers.
    /// @details Checks scratch non-aliasing, validates distance compatibility and finiteness, and evaluates `root`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] dist Input rectangular squared-distance matrix of shape `(n1, n2)`.
    /// @param[out] out Destination cross-covariance matrix of shape `(n1, n2)`.
    /// @param[in,out] scratch Non-aliasing scratch matrix of shape `(n1, n2)`.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @return Empty expected on success, or validation/workspace/evaluation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_cross(
        MatrixRef<T> dist, MatrixMut<T> out, MatrixMut<T> scratch, NestedBuffers<T>& nested
    ) const
    {
        if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value())
        {
            return sc;
        }
        if (auto v = impl_->validate_dist_call(dist, out.as_ref(), Triangle::Full, false); !v.has_value())
        {
            return v;
        }
        return impl_->eval_term_cross_dist(impl_->root, dist, out, scratch, nested, 0);
    }

    /// @brief Evaluates the rectangular cross-covariance matrix from squared distances using temporary nested buffers.
    /// @details Convenience overload allocating `make_nested_buffers(out.rows(), out.cols())`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] dist Input rectangular squared-distance matrix.
    /// @param[out] out Destination cross-covariance matrix view.
    /// @param[in,out] scratch Non-aliasing scratch matrix matching `out` shape.
    /// @return Empty expected on success, or validation/evaluation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_cross(
        MatrixRef<T> dist, MatrixMut<T> out, MatrixMut<T> scratch
    ) const
    {
        auto nested = make_nested_buffers(out.rows(), out.cols());
        return apply_cross(dist, out, scratch, nested);
    }

    /// @brief Evaluates the rectangular cross-covariance matrix from point matrices `x1` and `x2`.
    /// @details Uses caller-supplied `scratch` and `nested` buffers with zero heap allocations.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x1 First point matrix of shape `(n1, d)`.
    /// @param[in] x2 Second point matrix of shape `(n2, d)`.
    /// @param[out] out Destination cross-covariance matrix of shape `(n1, n2)`.
    /// @param[in,out] scratch Non-aliasing scratch matrix of shape `(n1, n2)`.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @return Empty expected on success, or validation/workspace/evaluation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_cross_points(
        MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out, MatrixMut<T> scratch, NestedBuffers<T>& nested
    ) const
    {
        if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value())
        {
            return sc;
        }
        if (auto v = impl_->validate_cross_points_call(x1, x2, out.as_ref()); !v.has_value())
        {
            return v;
        }
        return impl_->eval_term_cross_points(impl_->root, x1, x2, out, scratch, nested, 0);
    }

    /// @brief Evaluates the rectangular cross-covariance matrix from point matrices using temporary nested buffers.
    /// @details Convenience overload allocating `make_nested_buffers(out.rows(), out.cols())`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x1 First point matrix of shape `(n1, d)`.
    /// @param[in] x2 Second point matrix of shape `(n2, d)`.
    /// @param[out] out Destination cross-covariance matrix of shape `(n1, n2)`.
    /// @param[in,out] scratch Non-aliasing scratch matrix of shape `(n1, n2)`.
    /// @return Empty expected on success, or validation/evaluation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_cross_points(
        MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out, MatrixMut<T> scratch
    ) const
    {
        auto nested = make_nested_buffers(out.rows(), out.cols());
        return apply_cross_points(x1, x2, out, scratch, nested);
    }

    /// @brief Fills the diagonal $k(x_i, x_i)$ for query points `x` using caller-provided scratch.
    /// @details Validates empty inputs, output span length, ARD feature count, scratch length `(nested_depth + 1) * n`,
    /// and coordinate finiteness before running `eval_fill_diag` with zero heap allocations.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[out] out Output diagonal span of length `n`.
    /// @param[in,out] scratch Scratch vector span of length at least `(nested_depth() + 1) * n`.
    /// @return Empty expected on success, or validation/length/evaluation error.
    /// @retval GprErrorCode::EmptyInput If `x` has zero rows or columns.
    /// @retval GprErrorCode::LengthMismatch If `out` or `scratch` is improperly sized.
    /// @retval GprErrorCode::DimensionMismatch If `x.cols()` mismatches an ARD leaf's dimension.
    /// @retval GprErrorCode::NonFiniteInput If any entry of `x` is non-finite.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::fill_diag(
        MatrixRef<T> x, std::span<T> out, std::span<T> scratch
    ) const
    {
        // Enforce deterministic input checks: empty -> out length -> ARD dim -> scratch size -> finiteness.
        if (x.rows() == 0 || x.cols() == 0)
        {
            return std::unexpected(GprError::empty_input());
        }
        if (auto cnt = require_count(out.size(), x.rows(), "out"); !cnt.has_value())
        {
            return cnt;
        }
        if (auto ard = impl_->validate_ard_dims(x.cols()); !ard.has_value())
        {
            return ard;
        }
        const std::size_t min_scratch = (impl_->nested_depth + 1U) * out.size();
        if (scratch.size() < min_scratch)
        {
            return std::unexpected(GprError::length_mismatch("diagonal evaluation scratch buffer is too small"));
        }
        if (auto fin = require_matrix_finite_simd(x); !fin.has_value())
        {
            return fin;
        }
        return impl_->eval_fill_diag(impl_->root, x, out, scratch, 0);
    }

    /// @brief Fills the diagonal $k(x_i, x_i)$ for query points `x` using a temporary scratch buffer.
    /// @details Allocates `(nested_depth() + 1) * out.size()` elements and forwards to the zero-alloc `fill_diag`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[out] out Output diagonal span of length `n`.
    /// @return Empty expected on success, or validation/evaluation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::fill_diag(MatrixRef<T> x, std::span<T> out) const
    {
        const std::size_t min_scratch = (impl_->nested_depth + 1U) * out.size();
        std::vector<T> scratch_vec(min_scratch);
        return fill_diag(x, out, scratch_vec);
    }

    /// @brief Evaluates the first log-hyperparameter derivative $\partial K / \partial \theta_{\text{param\_idx}}$
    /// from squared Euclidean distances.
    /// @details Validates scratch non-aliasing, distance compatibility/finiteness, and parameter bounds before
    /// traversing `root`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] dist Input square squared-distance matrix.
    /// @param[out] d_k Output derivative matrix view.
    /// @param[in] param_idx Global hyperparameter index in `[0, num_params())`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch_a First non-aliasing scratch buffer matching `d_k` shape.
    /// @param[in,out] scratch_b Second non-aliasing scratch buffer matching `d_k` shape.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @return Empty expected on success, or validation/index/workspace error.
    /// @retval GprErrorCode::InvalidConfig If `scratch_a` and `scratch_b` alias each other or `d_k`.
    /// @retval GprErrorCode::IndexOutOfRange If `param_idx >= num_params()`.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::grad(
        MatrixRef<T> dist,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b,
        NestedBuffers<T>& nested
    ) const
    {
        if (auto sc_a = require_scratch(d_k.as_ref(), scratch_a, "scratch_a"); !sc_a.has_value())
        {
            return sc_a;
        }
        if (auto sc_b = require_scratch(d_k.as_ref(), scratch_b, "scratch_b"); !sc_b.has_value())
        {
            return sc_b;
        }
        if (scratch_a.data() == scratch_b.data())
        {
            return std::unexpected(GprError::invalid_config("scratch buffers must not alias"));
        }
        if (auto v = impl_->validate_dist_call(dist, d_k.as_ref(), uplo, true); !v.has_value())
        {
            return v;
        }
        if (param_idx >= impl_->bindings.size())
        {
            return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
        }
        return impl_->eval_grad_dist(impl_->root, dist, d_k, param_idx, uplo, scratch_a, scratch_b, nested, 0);
    }

    /// @brief Evaluates $\partial K / \partial \theta_{\text{param\_idx}}$ from squared distances using temporary
    /// nested buffers.
    /// @details Convenience overload allocating `make_nested_buffers(d_k.rows(), d_k.cols())`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] dist Input square squared-distance matrix.
    /// @param[out] d_k Output derivative matrix view.
    /// @param[in] param_idx Global hyperparameter index in `[0, num_params())`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch_a First non-aliasing scratch buffer matching `d_k` shape.
    /// @param[in,out] scratch_b Second non-aliasing scratch buffer matching `d_k` shape.
    /// @return Empty expected on success, or validation/index error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::grad(
        MatrixRef<T> dist,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b
    ) const
    {
        auto nested = make_nested_buffers(d_k.rows(), d_k.cols());
        return grad(dist, d_k, param_idx, uplo, scratch_a, scratch_b, nested);
    }

    /// @brief Evaluates $\partial K / \partial \theta_{\text{param\_idx}}$ directly from point coordinates.
    /// @details Validates non-aliasing scratch buffers, point inputs, ARD dimensions, and parameter index bounds.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[out] d_k Output derivative matrix of shape `(n, n)`.
    /// @param[in] param_idx Global hyperparameter index in `[0, num_params())`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch_a First non-aliasing scratch buffer of shape `(n, n)`.
    /// @param[in,out] scratch_b Second non-aliasing scratch buffer of shape `(n, n)`.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @return Empty expected on success, or validation/index/workspace error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::grad_points(
        MatrixRef<T> x,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b,
        NestedBuffers<T>& nested
    ) const
    {
        if (auto sc_a = require_scratch(d_k.as_ref(), scratch_a, "scratch_a"); !sc_a.has_value())
        {
            return sc_a;
        }
        if (auto sc_b = require_scratch(d_k.as_ref(), scratch_b, "scratch_b"); !sc_b.has_value())
        {
            return sc_b;
        }
        if (scratch_a.data() == scratch_b.data())
        {
            return std::unexpected(GprError::invalid_config("scratch buffers must not alias"));
        }
        if (auto v = impl_->validate_points_call(x, d_k.as_ref()); !v.has_value())
        {
            return v;
        }
        if (param_idx >= impl_->bindings.size())
        {
            return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
        }
        return impl_->eval_grad_points(impl_->root, x, d_k, param_idx, uplo, scratch_a, scratch_b, nested, 0);
    }

    /// @brief Evaluates $\partial K / \partial \theta_{\text{param\_idx}}$ from point coordinates using temporary
    /// nested buffers.
    /// @details Convenience overload allocating `make_nested_buffers(d_k.rows(), d_k.cols())`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[out] d_k Output derivative matrix of shape `(n, n)`.
    /// @param[in] param_idx Global hyperparameter index in `[0, num_params())`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch_a First non-aliasing scratch buffer of shape `(n, n)`.
    /// @param[in,out] scratch_b Second non-aliasing scratch buffer of shape `(n, n)`.
    /// @return Empty expected on success, or validation/index error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::grad_points(
        MatrixRef<T> x,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b
    ) const
    {
        auto nested = make_nested_buffers(d_k.rows(), d_k.cols());
        return grad_points(x, d_k, param_idx, uplo, scratch_a, scratch_b, nested);
    }

    /// @brief Evaluates the second log-hyperparameter derivative $\partial^2 K / (\partial \theta_i \partial \theta_j)$
    /// from squared Euclidean distances.
    /// @details Requires three mutually distinct non-aliasing scratch buffers matching `d2_k` shape.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] dist Input square squared-distance matrix.
    /// @param[out] d2_k Output second-derivative matrix view.
    /// @param[in] i First global hyperparameter index in `[0, num_params())`.
    /// @param[in] j Second global hyperparameter index in `[0, num_params())`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch_a First non-aliasing scratch buffer.
    /// @param[in,out] scratch_b Second non-aliasing scratch buffer.
    /// @param[in,out] scratch_c Third non-aliasing scratch buffer.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @return Empty expected on success, or validation/index/workspace error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::hess(
        MatrixRef<T> dist,
        MatrixMut<T> d2_k,
        std::size_t i,
        std::size_t j,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b,
        MatrixMut<T> scratch_c,
        NestedBuffers<T>& nested
    ) const
    {
        if (auto sc_a = require_scratch(d2_k.as_ref(), scratch_a, "scratch_a"); !sc_a.has_value())
        {
            return sc_a;
        }
        if (auto sc_b = require_scratch(d2_k.as_ref(), scratch_b, "scratch_b"); !sc_b.has_value())
        {
            return sc_b;
        }
        if (auto sc_c = require_scratch(d2_k.as_ref(), scratch_c, "scratch_c"); !sc_c.has_value())
        {
            return sc_c;
        }
        if (scratch_a.data() == scratch_b.data() || scratch_a.data() == scratch_c.data()
            || scratch_b.data() == scratch_c.data())
        {
            return std::unexpected(GprError::invalid_config("scratch buffers must not alias"));
        }
        if (auto v = impl_->validate_dist_call(dist, d2_k.as_ref(), uplo, true); !v.has_value())
        {
            return v;
        }
        if (i >= impl_->bindings.size() || j >= impl_->bindings.size())
        {
            return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
        }
        return impl_->eval_hess_dist(impl_->root, dist, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested, 0);
    }

    /// @brief Evaluates $\partial^2 K / (\partial \theta_i \partial \theta_j)$ from squared distances using temporary
    /// nested buffers.
    /// @details Convenience overload allocating `make_nested_buffers(d2_k.rows(), d2_k.cols())`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] dist Input square squared-distance matrix.
    /// @param[out] d2_k Output second-derivative matrix view.
    /// @param[in] i First global hyperparameter index.
    /// @param[in] j Second global hyperparameter index.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch_a First non-aliasing scratch buffer.
    /// @param[in,out] scratch_b Second non-aliasing scratch buffer.
    /// @param[in,out] scratch_c Third non-aliasing scratch buffer.
    /// @return Empty expected on success, or validation/index error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::hess(
        MatrixRef<T> dist,
        MatrixMut<T> d2_k,
        std::size_t i,
        std::size_t j,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b,
        MatrixMut<T> scratch_c
    ) const
    {
        auto nested = make_nested_buffers(d2_k.rows(), d2_k.cols());
        return hess(dist, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested);
    }

    /// @brief Evaluates $\partial^2 K / (\partial \theta_i \partial \theta_j)$ directly from point coordinates.
    /// @details Requires three mutually distinct non-aliasing scratch buffers and valid parameter indices `(i, j)`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[out] d2_k Output second-derivative matrix of shape `(n, n)`.
    /// @param[in] i First global hyperparameter index in `[0, num_params())`.
    /// @param[in] j Second global hyperparameter index in `[0, num_params())`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch_a First non-aliasing scratch buffer.
    /// @param[in,out] scratch_b Second non-aliasing scratch buffer.
    /// @param[in,out] scratch_c Third non-aliasing scratch buffer.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @return Empty expected on success, or validation/index/workspace error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::hess_points(
        MatrixRef<T> x,
        MatrixMut<T> d2_k,
        std::size_t i,
        std::size_t j,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b,
        MatrixMut<T> scratch_c,
        NestedBuffers<T>& nested
    ) const
    {
        if (auto sc_a = require_scratch(d2_k.as_ref(), scratch_a, "scratch_a"); !sc_a.has_value())
        {
            return sc_a;
        }
        if (auto sc_b = require_scratch(d2_k.as_ref(), scratch_b, "scratch_b"); !sc_b.has_value())
        {
            return sc_b;
        }
        if (auto sc_c = require_scratch(d2_k.as_ref(), scratch_c, "scratch_c"); !sc_c.has_value())
        {
            return sc_c;
        }
        if (scratch_a.data() == scratch_b.data() || scratch_a.data() == scratch_c.data()
            || scratch_b.data() == scratch_c.data())
        {
            return std::unexpected(GprError::invalid_config("scratch buffers must not alias"));
        }
        if (auto v = impl_->validate_points_call(x, d2_k.as_ref()); !v.has_value())
        {
            return v;
        }
        if (i >= impl_->bindings.size() || j >= impl_->bindings.size())
        {
            return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
        }
        return impl_->eval_hess_points(impl_->root, x, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested, 0);
    }

    /// @brief Evaluates $\partial^2 K / (\partial \theta_i \partial \theta_j)$ from point coordinates using temporary
    /// nested buffers.
    /// @details Convenience overload allocating `make_nested_buffers(d2_k.rows(), d2_k.cols())`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[out] d2_k Output second-derivative matrix of shape `(n, n)`.
    /// @param[in] i First global hyperparameter index.
    /// @param[in] j Second global hyperparameter index.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch_a First non-aliasing scratch buffer.
    /// @param[in,out] scratch_b Second non-aliasing scratch buffer.
    /// @param[in,out] scratch_c Third non-aliasing scratch buffer.
    /// @return Empty expected on success, or validation/index error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::hess_points(
        MatrixRef<T> x,
        MatrixMut<T> d2_k,
        std::size_t i,
        std::size_t j,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b,
        MatrixMut<T> scratch_c
    ) const
    {
        auto nested = make_nested_buffers(d2_k.rows(), d2_k.cols());
        return hess_points(x, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested);
    }

    /// @brief Evaluates the square training Gram matrix using precomputed `DistanceCache<T>` when populated.
    /// @details Validates scratch, points, ARD dimensions, and cache shapes before walking `root` with `&cache`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[in] cache Precomputed scalar distance and packed ARD cache.
    /// @param[out] out Destination square Gram matrix of shape `(n, n)`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch Non-aliasing scratch buffer of shape `(n, n)`.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @return Empty expected on success, or validation/cache/workspace error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_cached(
        MatrixRef<T> x,
        const DistanceCache<T>& cache,
        MatrixMut<T> out,
        Triangle uplo,
        MatrixMut<T> scratch,
        NestedBuffers<T>& nested
    ) const
    {
        if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value())
        {
            return sc;
        }
        if (auto v = impl_->validate_points_call(x, out.as_ref()); !v.has_value())
        {
            return v;
        }
        if (auto cs = impl_->validate_cache_shape(x, cache); !cs.has_value())
        {
            return cs;
        }
        return impl_->eval_term_points(impl_->root, x, out, uplo, scratch, nested, 0, &cache);
    }

    /// @brief Evaluates the training Gram matrix using `DistanceCache<T>` and temporary nested buffers.
    /// @details Convenience overload allocating `make_nested_buffers(out.rows(), out.cols())`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[in] cache Precomputed distance and ARD cache.
    /// @param[out] out Destination square Gram matrix of shape `(n, n)`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch Non-aliasing scratch buffer of shape `(n, n)`.
    /// @return Empty expected on success, or validation/cache error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_cached(
        MatrixRef<T> x, const DistanceCache<T>& cache, MatrixMut<T> out, Triangle uplo, MatrixMut<T> scratch
    ) const
    {
        auto nested = make_nested_buffers(out.rows(), out.cols());
        return apply_cached(x, cache, out, uplo, scratch, nested);
    }

    /// @brief Evaluates $\partial K / \partial \theta_{\text{param\_idx}}$ using precomputed `DistanceCache<T>`.
    /// @details Validates scratch non-aliasing, point shapes, cache shapes, and parameter bounds before walking `root`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[in] cache Precomputed distance and ARD cache.
    /// @param[out] d_k Output derivative matrix of shape `(n, n)`.
    /// @param[in] param_idx Global hyperparameter index in `[0, num_params())`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch_a First non-aliasing scratch buffer.
    /// @param[in,out] scratch_b Second non-aliasing scratch buffer.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @return Empty expected on success, or validation/cache/index error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::grad_cached(
        MatrixRef<T> x,
        const DistanceCache<T>& cache,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b,
        NestedBuffers<T>& nested
    ) const
    {
        if (auto sc_a = require_scratch(d_k.as_ref(), scratch_a, "scratch_a"); !sc_a.has_value())
        {
            return sc_a;
        }
        if (auto sc_b = require_scratch(d_k.as_ref(), scratch_b, "scratch_b"); !sc_b.has_value())
        {
            return sc_b;
        }
        if (scratch_a.data() == scratch_b.data())
        {
            return std::unexpected(GprError::invalid_config("scratch buffers must not alias"));
        }
        if (auto v = impl_->validate_points_call(x, d_k.as_ref()); !v.has_value())
        {
            return v;
        }
        if (auto cs = impl_->validate_cache_shape(x, cache); !cs.has_value())
        {
            return cs;
        }
        if (param_idx >= impl_->bindings.size())
        {
            return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
        }
        return impl_->eval_grad_points(impl_->root, x, d_k, param_idx, uplo, scratch_a, scratch_b, nested, 0, &cache);
    }

    /// @brief Evaluates $\partial K / \partial \theta_{\text{param\_idx}}$ using `DistanceCache<T>` and temporary
    /// nested buffers.
    /// @details Convenience overload allocating `make_nested_buffers(d_k.rows(), d_k.cols())`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[in] cache Precomputed distance and ARD cache.
    /// @param[out] d_k Output derivative matrix of shape `(n, n)`.
    /// @param[in] param_idx Global hyperparameter index.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch_a First non-aliasing scratch buffer.
    /// @param[in,out] scratch_b Second non-aliasing scratch buffer.
    /// @return Empty expected on success, or validation/cache/index error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::grad_cached(
        MatrixRef<T> x,
        const DistanceCache<T>& cache,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b
    ) const
    {
        auto nested = make_nested_buffers(d_k.rows(), d_k.cols());
        return grad_cached(x, cache, d_k, param_idx, uplo, scratch_a, scratch_b, nested);
    }

    /// @brief Evaluates $\partial^2 K / (\partial \theta_i \partial \theta_j)$ using precomputed `DistanceCache<T>`.
    /// @details Validates three non-aliasing scratch buffers, point inputs, cache compatibility, and parameter indices.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[in] cache Precomputed distance and ARD cache.
    /// @param[out] d2_k Output second-derivative matrix of shape `(n, n)`.
    /// @param[in] i First global hyperparameter index.
    /// @param[in] j Second global hyperparameter index.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch_a First non-aliasing scratch buffer.
    /// @param[in,out] scratch_b Second non-aliasing scratch buffer.
    /// @param[in,out] scratch_c Third non-aliasing scratch buffer.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @return Empty expected on success, or validation/cache/index error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::hess_cached(
        MatrixRef<T> x,
        const DistanceCache<T>& cache,
        MatrixMut<T> d2_k,
        std::size_t i,
        std::size_t j,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b,
        MatrixMut<T> scratch_c,
        NestedBuffers<T>& nested
    ) const
    {
        if (auto sc_a = require_scratch(d2_k.as_ref(), scratch_a, "scratch_a"); !sc_a.has_value())
        {
            return sc_a;
        }
        if (auto sc_b = require_scratch(d2_k.as_ref(), scratch_b, "scratch_b"); !sc_b.has_value())
        {
            return sc_b;
        }
        if (auto sc_c = require_scratch(d2_k.as_ref(), scratch_c, "scratch_c"); !sc_c.has_value())
        {
            return sc_c;
        }
        if (scratch_a.data() == scratch_b.data() || scratch_a.data() == scratch_c.data()
            || scratch_b.data() == scratch_c.data())
        {
            return std::unexpected(GprError::invalid_config("scratch buffers must not alias"));
        }
        if (auto v = impl_->validate_points_call(x, d2_k.as_ref()); !v.has_value())
        {
            return v;
        }
        if (auto cs = impl_->validate_cache_shape(x, cache); !cs.has_value())
        {
            return cs;
        }
        if (i >= impl_->bindings.size() || j >= impl_->bindings.size())
        {
            return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
        }
        return impl_->eval_hess_points(
            impl_->root, x, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested, 0, &cache
        );
    }

    /// @brief Evaluates $\partial^2 K / (\partial \theta_i \partial \theta_j)$ using `DistanceCache<T>` and temporary
    /// nested buffers.
    /// @details Convenience overload allocating `make_nested_buffers(d2_k.rows(), d2_k.cols())`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[in] cache Precomputed distance and ARD cache.
    /// @param[out] d2_k Output second-derivative matrix of shape `(n, n)`.
    /// @param[in] i First global hyperparameter index.
    /// @param[in] j Second global hyperparameter index.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch_a First non-aliasing scratch buffer.
    /// @param[in,out] scratch_b Second non-aliasing scratch buffer.
    /// @param[in,out] scratch_c Third non-aliasing scratch buffer.
    /// @return Empty expected on success, or validation/cache/index error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::hess_cached(
        MatrixRef<T> x,
        const DistanceCache<T>& cache,
        MatrixMut<T> d2_k,
        std::size_t i,
        std::size_t j,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b,
        MatrixMut<T> scratch_c
    ) const
    {
        auto nested = make_nested_buffers(d2_k.rows(), d2_k.cols());
        return hess_cached(x, cache, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested);
    }

    /// @brief Returns the compiled leaf index owning flattened global parameter `param_idx`.
    /// @details Used by incremental leaf-cache evaluators to map changed parameter indices to dirty leaves.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] param_idx Global hyperparameter index.
    /// @return Zero-based leaf index in `[0, num_leaves())`, or `GprErrorCode::IndexOutOfRange`.
    /// @retval GprErrorCode::IndexOutOfRange If `param_idx >= num_params()`.
    template <KernelScalarConcept T>
    std::expected<std::size_t, GprError> CompiledKernel<T>::leaf_index_for_param(std::size_t param_idx) const
    {
        if (param_idx >= impl_->bindings.size())
        {
            return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
        }
        return impl_->bindings[param_idx].leaf_id;
    }

    /// @brief Evaluates a single compiled leaf Gram matrix directly from point coordinates `x`.
    /// @details Supports selective per-leaf recomputation for incremental optimizers.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf_idx Zero-based leaf index in `[0, num_leaves())`.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[out] out Destination square leaf Gram matrix of shape `(n, n)`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @return Empty expected on success, or index/validation/evaluation error.
    /// @retval GprErrorCode::IndexOutOfRange If `leaf_idx >= num_leaves()`.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_leaf_points(
        std::size_t leaf_idx, MatrixRef<T> x, MatrixMut<T> out, Triangle uplo, NestedBuffers<T>& nested
    ) const
    {
        if (leaf_idx >= impl_->leaves.size())
        {
            return std::unexpected(GprError::index_out_of_range("kernel leaf index out of range"));
        }
        if (auto v = impl_->validate_points_call(x, out.as_ref()); !v.has_value())
        {
            return v;
        }
        return impl_->eval_leaf_at(leaf_idx, x, out, uplo, nested, 0);
    }

    /// @brief Evaluates a single compiled leaf Gram matrix using precomputed `DistanceCache<T>`.
    /// @details Validates leaf bounds, point shape, and cache shape before evaluating `leaves[leaf_idx]`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf_idx Zero-based leaf index in `[0, num_leaves())`.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[in] cache Precomputed distance and ARD cache.
    /// @param[out] out Destination square leaf Gram matrix of shape `(n, n)`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @return Empty expected on success, or index/validation/cache error.
    /// @retval GprErrorCode::IndexOutOfRange If `leaf_idx >= num_leaves()`.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_leaf_cached(
        std::size_t leaf_idx,
        MatrixRef<T> x,
        const DistanceCache<T>& cache,
        MatrixMut<T> out,
        Triangle uplo,
        NestedBuffers<T>& nested
    ) const
    {
        if (leaf_idx >= impl_->leaves.size())
        {
            return std::unexpected(GprError::index_out_of_range("kernel leaf index out of range"));
        }
        if (auto v = impl_->validate_points_call(x, out.as_ref()); !v.has_value())
        {
            return v;
        }
        if (auto cs = impl_->validate_cache_shape(x, cache); !cs.has_value())
        {
            return cs;
        }
        return impl_->eval_leaf_at(leaf_idx, x, out, uplo, nested, 0, &cache);
    }

    /// @brief Evaluates a single compiled leaf using `DistanceCache<T>` and incremental ARD $s^2$ state.
    /// @details Validates leaf bounds, point shape, and cache shape before delegating to
    /// `eval_leaf_ard_incremental`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf_idx Zero-based leaf index in `[0, num_leaves())`.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[in] cache Precomputed distance and ARD cache.
    /// @param[out] out Destination square leaf Gram matrix of shape `(n, n)`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @param[in,out] ard_s2_cache Caller-owned packed $s^2$ buffer for this leaf.
    /// @param[in,out] prev_ls Caller-owned lengthscale snapshot buffer for this leaf.
    /// @param[in,out] s2_valid True while `ard_s2_cache` matches `prev_ls`.
    /// @param[in,out] inc_steps Consecutive incremental updates applied since last rebuild.
    /// @param[in] commit_s2 True to update `ard_s2_cache`/`prev_ls`; false for non-destructive trial evaluation.
    /// @return Empty expected on success, or index/validation/cache error.
    /// @retval GprErrorCode::IndexOutOfRange If `leaf_idx >= num_leaves()`.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_leaf_cached_incremental(
        std::size_t leaf_idx,
        MatrixRef<T> x,
        const DistanceCache<T>& cache,
        MatrixMut<T> out,
        Triangle uplo,
        NestedBuffers<T>& nested,
        std::span<T> ard_s2_cache,
        std::span<double> prev_ls,
        bool& s2_valid,
        std::size_t& inc_steps,
        bool commit_s2
    ) const
    {
        return apply_leaf_incremental(
            leaf_idx, x, &cache, out, uplo, nested, ard_s2_cache, prev_ls, s2_valid, inc_steps, commit_s2
        );
    }

    /// @brief Evaluates a single compiled leaf incrementally using an optional `DistanceCache<T>`.
    /// @details Validates leaf bounds, point shape, and cache shape before delegating to `eval_leaf_ard_incremental`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf_idx Zero-based leaf index in `[0, num_leaves())`.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[in] cache Optional precomputed distance and ARD cache pointer, or `nullptr` for uncached evaluation.
    /// @param[out] out Destination square leaf Gram matrix of shape `(n, n)`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @param[in,out] ard_s2_cache Caller-owned packed $s^2$ buffer for this leaf.
    /// @param[in,out] prev_ls Caller-owned lengthscale snapshot buffer for this leaf.
    /// @param[in,out] s2_valid True while `ard_s2_cache` matches `prev_ls`.
    /// @param[in,out] inc_steps Consecutive incremental updates applied since last rebuild.
    /// @param[in] commit_s2 True to update `ard_s2_cache`/`prev_ls`; false for non-destructive trial evaluation.
    /// @return Empty expected on success, or index/validation/cache error.
    /// @retval GprErrorCode::IndexOutOfRange If `leaf_idx >= num_leaves()`.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_leaf_incremental(
        std::size_t leaf_idx,
        MatrixRef<T> x,
        const DistanceCache<T>* cache,
        MatrixMut<T> out,
        Triangle uplo,
        NestedBuffers<T>& nested,
        std::span<T> ard_s2_cache,
        std::span<double> prev_ls,
        bool& s2_valid,
        std::size_t& inc_steps,
        bool commit_s2
    ) const
    {
        if (leaf_idx >= impl_->leaves.size())
        {
            return std::unexpected(GprError::index_out_of_range("kernel leaf index out of range"));
        }
        if (auto v = impl_->validate_points_shape(x, out.as_ref()); !v.has_value())
        {
            return v;
        }
        if (cache != nullptr)
        {
            if (auto cs = impl_->validate_cache_shape(x, *cache); !cs.has_value())
            {
                return cs;
            }
        }
        return impl_->eval_leaf_ard_incremental(
            leaf_idx, x, cache, out, uplo, nested, ard_s2_cache, prev_ls, s2_valid, inc_steps, commit_s2
        );
    }

    /// @brief Commits accepted lengthscales of `leaf_idx` to its `ard_s2_cache` baseline.
    /// @details No-op for non-ARD leaves. Validates leaf index, point matrix, and optional cache shape.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf_idx Zero-based leaf index in `[0, num_leaves())`.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[in] cache Optional precomputed distance and ARD cache pointer, or `nullptr`.
    /// @param[in,out] ard_s2_cache Caller-owned packed $s^2$ buffer for this leaf.
    /// @param[in,out] prev_ls Caller-owned lengthscale snapshot buffer for this leaf.
    /// @param[in,out] s2_valid True while `ard_s2_cache` matches `prev_ls`.
    /// @param[in,out] inc_steps Consecutive incremental updates applied since last rebuild.
    /// @return Empty expected on success, or index/validation/cache error.
    /// @retval GprErrorCode::IndexOutOfRange If `leaf_idx >= num_leaves()`.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::commit_leaf_ard_s2(
        std::size_t leaf_idx,
        MatrixRef<T> x,
        const DistanceCache<T>* cache,
        std::span<T> ard_s2_cache,
        std::span<double> prev_ls,
        bool& s2_valid,
        std::size_t& inc_steps
    ) const
    {
        if (leaf_idx >= impl_->leaves.size())
        {
            return std::unexpected(GprError::index_out_of_range("kernel leaf index out of range"));
        }
        if (x.is_empty() || x.data() == nullptr)
        {
            return std::unexpected(GprError::empty_input());
        }
        if (cache != nullptr)
        {
            if (auto cs = impl_->validate_cache_shape(x, *cache); !cs.has_value())
            {
                return cs;
            }
        }
        return impl_->commit_leaf_ard_s2_impl(leaf_idx, x, cache, ard_s2_cache, prev_ls, s2_valid, inc_steps);
    }

    /// @brief Combines precomputed per-leaf Gram matrices into the full composite Gram matrix.
    /// @details Validates output squareness, non-aliasing scratch, `leaf_grams.size() == num_leaves()`, and every
    /// leaf Gram's `(n, n)` shape before walking `root` via `eval_combine_term`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf_grams Span of precomputed square Gram matrix views ordered by leaf index.
    /// @param[out] out Destination composite Gram matrix of shape `(n, n)`.
    /// @param[in] uplo Active matrix triangle to combine.
    /// @param[in,out] scratch Non-aliasing scratch matrix of shape `(n, n)`.
    /// @param[in,out] nested Pre-allocated nested buffer pool.
    /// @return Empty expected on success, or empty/shape/workspace error.
    /// @retval GprErrorCode::EmptyInput If `out` is empty or null.
    /// @retval GprErrorCode::LengthMismatch If `out` is non-square, leaf count mismatches, or any leaf Gram shape
    /// differs from `out`.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::combine_from_leaf_grams(
        std::span<const MatrixRef<T>> leaf_grams,
        MatrixMut<T> out,
        Triangle uplo,
        MatrixMut<T> scratch,
        NestedBuffers<T>& nested
    ) const
    {
        if (out.is_empty() || out.data() == nullptr)
        {
            return std::unexpected(GprError::empty_input());
        }
        if (out.rows() != out.cols())
        {
            return std::unexpected(GprError::length_mismatch("output matrix must be square"));
        }
        if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value())
        {
            return sc;
        }
        if (leaf_grams.size() != impl_->leaves.size())
        {
            return std::unexpected(GprError::length_mismatch("leaf Gram count does not match compiled kernel leaves"));
        }
        for (const auto& gram : leaf_grams)
        {
            if (gram.rows() != out.rows() || gram.cols() != out.cols() || gram.data() == nullptr)
            {
                return std::unexpected(GprError::length_mismatch("leaf Gram matrix shape mismatch"));
            }
        }
        return impl_->eval_combine_term(impl_->root, leaf_grams, out, uplo, nested, 0);
    }

    /// @brief Returns the `keep_plan` count of top-level multi-factor products whose factor Grams should be retained.
    /// @details Maximizes retained products without exceeding the unkept reverse-walk `n x n` buffer budget.
    /// @tparam T Storage floating-point scalar type.
    /// @return Optimal count of top-level products to keep during `apply_keeping` / `apply_cached_keeping`.
    template <KernelScalarConcept T> std::size_t CompiledKernel<T>::kept_products() const noexcept
    {
        return impl_->compute_kept_products();
    }

    /// @brief Returns the number of `n x n` buffers required to retain factor Grams for `products` top-level products.
    /// @details Callers pass this depth to `kept_bufs` in `apply_keeping` and `apply_cached_keeping`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] products Number of top-level multi-factor products to retain.
    /// @return Number of `n x n` buffers needed for retained factor Grams.
    template <KernelScalarConcept T> std::size_t CompiledKernel<T>::kept_buffers(std::size_t products) const noexcept
    {
        return impl_->kept_grams_count(products);
    }

    /// @brief Returns total `n x n` buffers required in `weighted_bufs` when calling `weighted_grads`.
    /// @details Equals `kept_grams_count(products) + walk_buffers_count(products)`, partitioning `weighted_bufs` into a
    /// retained-Gram prefix followed by reverse-walk working buffers.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] products Number of top-level multi-factor products retained during forward evaluation.
    /// @return Total `NestedBuffers<T>` depth required for `weighted_grads`.
    template <KernelScalarConcept T>
    std::size_t CompiledKernel<T>::weighted_buffers(std::size_t products) const noexcept
    {
        return impl_->kept_grams_count(products) + impl_->walk_buffers_count(products);
    }

    /// @brief Evaluates the training Gram matrix from point coordinates while saving `keep_plan` factor Grams.
    /// @details Saves varying factor Grams of the first `products` top-level multi-factor products into `kept_bufs` so
    /// a subsequent `weighted_grads` call avoids re-evaluating those factors.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[out] out Destination square Gram matrix of shape `(n, n)`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch Non-aliasing scratch buffer of shape `(n, n)`.
    /// @param[in,out] nested Pre-allocated intermediate buffer pool.
    /// @param[in,out] kept_bufs Retention buffer pool of depth at least `kept_buffers(products)`.
    /// @param[in] products Number of top-level multi-factor products to retain.
    /// @param[out] ard_s2_buf Optional packed forward ARD $s^2$ storage.
    /// @param[in,out] ard_s2_ready Optional per-ARD-leaf readiness bytes.
    /// @return Empty expected on success, or validation/workspace/evaluation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_keeping(
        MatrixRef<T> x,
        MatrixMut<T> out,
        Triangle uplo,
        MatrixMut<T> scratch,
        NestedBuffers<T>& nested,
        NestedBuffers<T>& kept_bufs,
        std::size_t products,
        std::span<T> ard_s2_buf,
        std::span<std::uint8_t> ard_s2_ready
    ) const
    {
        if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value())
        {
            return sc;
        }
        if (auto v = impl_->validate_points_call(x, out.as_ref()); !v.has_value())
        {
            return v;
        }
        return impl_->eval_keeping_impl(
            x, nullptr, out, uplo, scratch, nested, kept_bufs, products, ard_s2_buf, ard_s2_ready
        );
    }

    /// @brief Evaluates the training Gram matrix using `DistanceCache<T>` while saving `keep_plan` factor Grams.
    /// @details Combines distance/ARD cache acceleration with factor Gram retention in `kept_bufs`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[in] cache Precomputed distance and ARD cache.
    /// @param[out] out Destination square Gram matrix of shape `(n, n)`.
    /// @param[in] uplo Active matrix triangle to populate.
    /// @param[in,out] scratch Non-aliasing scratch buffer of shape `(n, n)`.
    /// @param[in,out] nested Pre-allocated intermediate buffer pool.
    /// @param[in,out] kept_bufs Retention buffer pool of depth at least `kept_buffers(products)`.
    /// @param[in] products Number of top-level multi-factor products to retain.
    /// @param[out] ard_s2_buf Optional packed forward ARD $s^2$ storage.
    /// @param[in,out] ard_s2_ready Optional per-ARD-leaf readiness bytes.
    /// @return Empty expected on success, or validation/cache/workspace error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_cached_keeping(
        MatrixRef<T> x,
        const DistanceCache<T>& cache,
        MatrixMut<T> out,
        Triangle uplo,
        MatrixMut<T> scratch,
        NestedBuffers<T>& nested,
        NestedBuffers<T>& kept_bufs,
        std::size_t products,
        std::span<T> ard_s2_buf,
        std::span<std::uint8_t> ard_s2_ready
    ) const
    {
        if (auto sc = require_scratch(out.as_ref(), scratch, "scratch"); !sc.has_value())
        {
            return sc;
        }
        if (auto v = impl_->validate_points_call(x, out.as_ref()); !v.has_value())
        {
            return v;
        }
        if (auto cs = impl_->validate_cache_shape(x, cache); !cs.has_value())
        {
            return cs;
        }
        return impl_->eval_keeping_impl(
            x, &cache, out, uplo, scratch, nested, kept_bufs, products, ard_s2_buf, ard_s2_ready
        );
    }

    /// @brief Computes $\langle W, \partial K / \partial \theta_p \rangle_F$ for all parameters in a single tree pass.
    /// @details Walks top-level `root_terms()`, matching each multi-factor `ProductTerm` with its retained factor Grams
    /// in the prefix `[0, kept_total)` of `weighted_bufs` and delegating adjoint contraction to `walk_weighted` using
    /// working slots starting at `kept_total`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[in] cache Optional precomputed distance and ARD cache (`nullptr` for uncached evaluation).
    /// @param[in] weight_lower Lower triangle of the symmetric adjoint matrix $W$ of shape `(n, n)`.
    /// @param[out] out_grads Output span of length `num_params()` receiving $\langle W, \partial K / \partial \theta_p
    /// \rangle_F$.
    /// @param[in,out] kernel_scratch Non-aliasing square scratch buffer of shape `(n, n)`.
    /// @param[in,out] nested Pre-allocated intermediate buffer pool.
    /// @param[in,out] weighted_bufs Pre-allocated pool containing kept factor Grams in `[0, kept_buffers(products))`
    /// followed by walk scratch up to `weighted_buffers(products)`.
    /// @param[in] products Number of top-level multi-factor products previously retained in `weighted_bufs`.
    /// @param[in,out] col_partials Per-column partial sum buffer of length at least `n` for deterministic reduction.
    /// @param[in] ard_s2_buf Optional forward-captured ARD $s^2$ storage passed from `apply_keeping`.
    /// @param[in] ard_s2_ready Optional readiness bytes identifying valid slices inside `ard_s2_buf`.
    /// @return Empty expected on success, or length/shape/cache/workspace error.
    /// @retval GprErrorCode::LengthMismatch If `out_grads.size() != num_params()` or cache/input dimensions mismatch.
    /// @retval GprErrorCode::WorkspaceTooSmall If `col_partials` or `weighted_bufs` is smaller than required.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::weighted_grads(
        MatrixRef<T> x,
        const DistanceCache<T>* cache,
        MatrixRef<T> weight_lower,
        std::span<double> out_grads,
        MatrixMut<T> kernel_scratch,
        NestedBuffers<T>& nested,
        NestedBuffers<T>& weighted_bufs,
        std::size_t products,
        std::span<double> col_partials,
        std::span<const T> ard_s2_buf,
        std::span<const std::uint8_t> ard_s2_ready
    ) const
    {
        // Enforce deterministic input, cache, scratch, col_partials, and weighted_bufs capacity checks.
        if (out_grads.size() != impl_->bindings.size())
        {
            return std::unexpected(GprError::length_mismatch("weighted_grads output length mismatch"));
        }
        if (auto v = impl_->validate_points_call(x, weight_lower); !v.has_value())
        {
            return v;
        }
        if (cache != nullptr)
        {
            if (auto cs = impl_->validate_cache_shape(x, *cache); !cs.has_value())
            {
                return cs;
            }
        }
        if (auto sc = require_scratch(weight_lower, kernel_scratch, "kernel_scratch"); !sc.has_value())
        {
            return sc;
        }
        if (col_partials.size() < x.rows())
        {
            return std::unexpected(GprError::workspace_too_small());
        }
        const std::size_t needed_bufs = weighted_buffers(products);
        if (weighted_bufs.depth() < needed_bufs || weighted_bufs.rows() < x.rows() || weighted_bufs.cols() < x.rows())
        {
            return std::unexpected(GprError::workspace_too_small());
        }

        // Partition weighted_bufs: [0, kept_total) holds pre-kept factor Grams; walk scratch begins at kept_total.
        const std::size_t kept_total = impl_->kept_grams_count(products);
        std::size_t p_off = 0;
        std::size_t kept_idx = 0;
        std::size_t kept_off = 0;
        for (const auto& t : impl_->root_terms())
        {
            const auto [tb, te] = Impl::param_range(t);
            const std::size_t count = te - tb;
            std::size_t v = 0;
            bool has_kept = false;
            if (const auto* p = std::get_if<typename Impl::ProductTerm>(&t))
            {
                v = impl_->varying_factors(p->terms);
                if (v >= 2U && kept_idx < products)
                {
                    has_kept = true;
                    ++kept_idx;
                }
            }
            auto res = impl_->walk_weighted(
                t, x, cache, weight_lower, out_grads.subspan(p_off, count), kernel_scratch, nested, weighted_bufs,
                has_kept ? kept_off : 0U, has_kept ? v : 0U, has_kept, kept_total, col_partials, false, ard_s2_buf,
                ard_s2_ready
            );
            if (!res.has_value())
            {
                return std::unexpected(std::move(res).error());
            }
            if (has_kept)
            {
                kept_off += v;
            }
            p_off += count;
        }
        return {};
    }

    template class NestedBuffers<double>;
    template class NestedBuffers<float>;
    template class CompiledKernel<double>;
    template class CompiledKernel<float>;

}  // namespace cxxgpr
