// SPDX-License-Identifier: Apache-2.0
/// @file compiled.cpp
/// @brief Execution-layer compiled kernel tree evaluation, differentiation, and adjoint walks.
/// @details Implements `NestedBuffers`, associative sum/product flattening inside `CompiledKernel::Impl`, cached and
/// uncached kernel Gram evaluation, analytical first and second log-hyperparameter derivatives, `keep_plan` factor Gram
/// retention, and single-pass `weighted_grads` reverse contraction walks.

#include "cxxgpr/kernel/compiled.hpp"

#include "cxxgpr/core/checked.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <variant>

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
        /// @param leaf Kernel leaf variant.
        /// @return True when `leaf` consumes scalar squared distances.
        [[nodiscard]] bool leaf_reads_distances(const KernelLeaf& leaf) noexcept
        {
            return std::holds_alternative<RbfKernel>(leaf) || std::holds_alternative<MaternKernel>(leaf)
                || std::holds_alternative<PeriodicKernel>(leaf) || std::holds_alternative<RationalQuadraticKernel>(leaf)
                || std::holds_alternative<CustomKernel>(leaf);
        }

        /// @brief Returns true if evaluating `leaf` can consume packed per-dimension ARD squared differences.
        /// @details Matches `RbfArdKernel`, `MaternArdKernel`, and `RationalQuadraticArdKernel`.
        /// @param leaf Kernel leaf variant.
        /// @return True when `leaf` consumes packed ARD squared differences.
        [[nodiscard]] bool leaf_needs_ard(const KernelLeaf& leaf) noexcept
        {
            return std::holds_alternative<RbfArdKernel>(leaf) || std::holds_alternative<MaternArdKernel>(leaf)
                || std::holds_alternative<RationalQuadraticArdKernel>(leaf);
        }

        /// @brief Combines two leaf or subtree `CoordMode` values across composition operators.
        /// @details `Either` acts as an identity element, matching modes preserve themselves, and conflicting
        /// `Dist` and `Points` requirements promote to `CoordMode::Mixed`.
        /// @param a First child coordinate mode.
        /// @param b Second child coordinate mode.
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
        /// @param x Input point matrix of shape `(n, d)`.
        /// @param dist Target square squared-distance matrix of shape `(n, n)`.
        template <KernelScalarConcept T> void fill_square_sq_dist(MatrixRef<T> x, MatrixMut<T> dist) noexcept
        {
            const std::size_t n = x.rows();
            const std::size_t d = x.cols();
            // Exploit symmetry: compute strictly lower entries and mirror across the diagonal.
            for (std::size_t col = 0; col < n; ++col)
            {
                dist(col, col) = T{0};
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    const std::size_t r1 = row;
                    const std::size_t r2 = col;
                    T acc = T{0};
                    for (std::size_t k = 0; k < d; ++k)
                    {
                        const T diff = x(r1, k) - x(r2, k);
                        acc += diff * diff;
                    }
                    dist(r1, r2) = acc;
                    dist(r2, r1) = acc;
                }
            }
        }

        /// @brief Populates a rectangular cross squared Euclidean distance matrix between `x1` and `x2`.
        /// @details Iterates in column-major order across `(x1.rows(), x2.rows())` pairs.
        /// @tparam T Storage floating-point scalar type.
        /// @param x1 First point matrix of shape `(n1, d)`.
        /// @param x2 Second point matrix of shape `(n2, d)`.
        /// @param dist Target rectangular squared-distance matrix of shape `(n1, n2)`.
        template <KernelScalarConcept T>
        void fill_cross_sq_dist(MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> dist) noexcept
        {
            const std::size_t n1 = x1.rows();
            const std::size_t n2 = x2.rows();
            const std::size_t d = x1.cols();
            for (std::size_t col = 0; col < n2; ++col)
            {
                for (std::size_t row = 0; row < n1; ++row)
                {
                    T acc = T{0};
                    for (std::size_t k = 0; k < d; ++k)
                    {
                        const T diff = x1(row, k) - x2(col, k);
                        acc += diff * diff;
                    }
                    dist(row, col) = acc;
                }
            }
        }

        /// @brief Validates that a scratch matrix matches `out` shape and does not alias `out`.
        /// @details Returns `WorkspaceTooSmall` if dimensions differ and `InvalidConfig` on pointer aliasing.
        /// @tparam T Storage floating-point scalar type.
        /// @param out Reference output matrix view whose dimensions must be matched.
        /// @param scratch Mutable scratch matrix view to check.
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
        /// @details Parallelizes across columns with OpenMP (`if (n_cols >= 64U)`) so workers operate on disjoint
        /// columns within the requested triangle (`Lower`, `Upper`, or `Full`).
        /// @tparam T Storage floating-point scalar type.
        /// @tparam Op Callable mutating `T& dst_elem` using `T src_elem`.
        /// @param dst Destination matrix updated in place.
        /// @param src Source matrix read over the active region.
        /// @param uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        /// @param op Element-wise update functor.
        template <KernelScalarConcept T, typename Op>
        void apply_binary_op(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo, Op&& op) noexcept
        {
            const std::size_t n_rows = dst.rows();
            const std::size_t n_cols = dst.cols();
            const auto cols_s = static_cast<std::ptrdiff_t>(n_cols);
            // Restrict updates to the active triangle on square outputs and parallelize disjoint columns.
            if (uplo == Triangle::Lower && n_rows == n_cols)
            {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n_cols >= 64U)
#endif
                for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s)
                {
                    const auto col = static_cast<std::size_t>(col_s);
                    for (std::size_t row = col; row < n_rows; ++row)
                    {
                        op(dst(row, col), src(row, col));
                    }
                }
                return;
            }
            if (uplo == Triangle::Upper && n_rows == n_cols)
            {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n_cols >= 64U)
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
#    pragma omp parallel for schedule(static) if (n_cols >= 64U)
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
        /// @param dst Destination matrix.
        /// @param src Source matrix.
        /// @param uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        template <KernelScalarConcept T> void copy_into(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo) noexcept
        {
            apply_binary_op(dst, src, uplo, [](T& d, T s) { d = s; });
        }

        /// @brief Accumulates `dst += src` over the specified matrix region `uplo`.
        /// @details Delegates column-parallel addition to `apply_binary_op`.
        /// @tparam T Storage floating-point scalar type.
        /// @param dst Destination matrix accumulated in place.
        /// @param src Summand matrix.
        /// @param uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        template <KernelScalarConcept T> void add_into(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo) noexcept
        {
            apply_binary_op(dst, src, uplo, [](T& d, T s) { d += s; });
        }

        /// @brief Accumulates Hadamard product `dst *= src` over the specified matrix region `uplo`.
        /// @details Delegates column-parallel element-wise multiplication to `apply_binary_op`.
        /// @tparam T Storage floating-point scalar type.
        /// @param dst Destination matrix scaled in place.
        /// @param src Multiplier matrix.
        /// @param uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        template <KernelScalarConcept T> void mul_into(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo) noexcept
        {
            apply_binary_op(dst, src, uplo, [](T& d, T s) { d *= s; });
        }

        /// @brief Multiplies the lower triangle (including diagonal) of `dst` by scalar `scale`.
        /// @details Parallelizes across columns with OpenMP (`if (n >= 64U)`).
        /// @tparam T Storage floating-point scalar type.
        /// @param dst Square matrix whose lower triangle is scaled in place.
        /// @param scale Scalar multiplier.
        template <KernelScalarConcept T> void scale_lower(MatrixMut<T> dst, T scale) noexcept
        {
            const std::size_t n = dst.rows();
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
            // Scale columns in parallel over the lower triangle only.
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 64U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                for (std::size_t row = col; row < n; ++row)
                {
                    dst(row, col) *= scale;
                }
            }
        }

        /// @brief Computes the symmetric Frobenius inner product $\langle A, B \rangle_F$ from lower triangles.
        /// @details Evaluates each column's `diag + 2 * strictly_lower` inner product in `double` across OpenMP
        /// workers into `col_partials`, then reduces `col_partials` serially in ascending column index order so the
        /// floating-point result is bit-for-bit independent of thread count.
        /// @tparam T Storage floating-point scalar type.
        /// @param a First symmetric matrix (only lower triangle read).
        /// @param b Second symmetric matrix (only lower triangle read).
        /// @param col_partials Pre-allocated per-column partial sum buffer of length at least `a.rows()`.
        /// @return Full symmetric Frobenius inner product in `double`.
        template <KernelScalarConcept T>
        [[nodiscard]] double lower_frobenius(MatrixRef<T> a, MatrixRef<T> b, std::span<double> col_partials) noexcept
        {
            const std::size_t n = a.rows();
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
            // Compute column-wise symmetric contributions (diag + 2 * strictly lower) in parallel.
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 64U)
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
        /// parallel and sums them serially in ascending column order for deterministic OpenMP reductions.
        /// @tparam T Storage floating-point scalar type.
        /// @param w Symmetric matrix whose lower triangle is summed.
        /// @param col_partials Pre-allocated per-column partial sum buffer of length at least `w.rows()`.
        /// @return Sum of all $n^2$ symmetric entries in `double`.
        template <KernelScalarConcept T>
        [[nodiscard]] double lower_sum(MatrixRef<T> w, std::span<double> col_partials) noexcept
        {
            const std::size_t n = w.rows();
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
            // Accumulate diag + 2 * strictly lower per column in parallel.
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 64U)
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
        /// @param w Symmetric square matrix.
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
    /// @param depth Number of nested buffer levels to allocate.
    /// @param rows Row capacity per buffer level.
    /// @param cols Column capacity per buffer level.
    template <KernelScalarConcept T>
    NestedBuffers<T>::NestedBuffers(std::size_t depth, std::size_t rows, std::size_t cols)
    {
        resize(depth, rows, cols);
    }

    /// @brief Reconfigures nested buffer dimensions and grows backing storage monotonically if needed.
    /// @details Never shrinks existing capacity so repeated resize calls for smaller subproblems avoid reallocations.
    /// @tparam T Storage floating-point scalar type.
    /// @param depth Number of nested buffer levels.
    /// @param rows Row dimension per level.
    /// @param cols Column dimension per level.
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
    /// @param index Zero-based buffer level index (`< depth()`).
    /// @param rows Active view row count (`<= rows()`).
    /// @param cols Active view column count (`<= cols()`).
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

        /// @brief Compiled leaf kernels stored in depth-first left-to-right evaluation order.
        std::vector<KernelLeaf> leaves;
        /// @brief Pre-extracted ARD lengthscale vectors per leaf to avoid heap allocations in hot loops.
        std::vector<std::vector<double>> cached_ls;
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

        /// @brief Refreshes `cached_ls` snapshots from current ARD leaf hyperparameters.
        /// @details Populates per-leaf vectors for ARD leaves and clears non-ARD entries so hot-path evaluation reads
        /// contiguous spans without calling `lengthscales()` allocations.
        void refresh_cached_ls()
        {
            cached_ls.resize(leaves.size());
            for (std::size_t idx = 0; idx < leaves.size(); ++idx)
            {
                if (const auto* k = std::get_if<RbfArdKernel>(&leaves[idx]))
                {
                    cached_ls[idx] = k->lengthscales();
                }
                else if (const auto* km = std::get_if<MaternArdKernel>(&leaves[idx]))
                {
                    cached_ls[idx] = km->lengthscales();
                }
                else if (const auto* krq = std::get_if<RationalQuadraticArdKernel>(&leaves[idx]))
                {
                    cached_ls[idx] = krq->lengthscales();
                }
                else
                {
                    cached_ls[idx].clear();
                }
            }
        }

        /// @brief Validates that populated distance and ARD caches match the shape of point matrix `x`.
        /// @details Checks square distance cache dimensions `(n, n)`, ARD point count `n`, ARD feature count `d`, and
        /// packed triangular length `d * n * (n + 1) / 2`.
        /// @param x Input point matrix of shape `(n, d)`.
        /// @param cache Precomputed distance and ARD cache to validate.
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
        /// @param term Compiled AST node.
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
        /// @param term Compiled AST node.
        /// @param param_idx Global hyperparameter index to query.
        /// @return True if `term` owns `param_idx`.
        static bool contains_param(const Term& term, std::size_t param_idx) noexcept
        {
            const auto [begin, end] = param_range(term);
            return param_idx >= begin && param_idx < end;
        }

        /// @brief Computes the nested intermediate buffer depth required to evaluate `term` without allocation.
        /// @details Leaf terms require depth 0 (they evaluate directly into caller-provided `out` or `scratch`); each
        /// non-leaf child adds one nested level above its own required depth.
        /// @param term Compiled AST node.
        /// @return Maximum nested buffer depth required across all subtrees of `term`.
        static std::size_t compute_nested_depth(const Term& term) noexcept
        {
            const std::vector<Term>* children = nullptr;
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                children = &s->terms;
            }
            else if (const auto* p = std::get_if<ProductTerm>(&term))
            {
                children = &p->terms;
            }
            else
            {
                return 0;
            }
            // Only non-leaf children borrow an extra nested buffer level before folding into the parent buffer.
            std::size_t max_child = 0;
            for (const auto& child : *children)
            {
                if (!std::holds_alternative<LeafRef>(child))
                {
                    max_child = std::max(max_child, 1U + compute_nested_depth(child));
                }
            }
            return max_child;
        }

        /// @brief Recursively compiles a declaration-layer `KernelSpec` into an associatively flattened `Term`.
        /// @details Assigns contiguous leaf indices and `ParameterBinding` records in depth-first left-to-right order,
        /// and flattens nested `SumNode` or `ProductNode` chains via `collect_sum` and `collect_product`.
        /// @param spec Input kernel expression specification.
        /// @param out_leaves Output vector accumulating compiled leaf objects.
        /// @param out_bindings Output vector accumulating parameter-to-leaf bindings.
        /// @param next_param_idx Running global parameter counter incremented for each bound parameter.
        /// @return Compiled `Term` representing `spec`.
        static Term compile_node(
            const KernelSpec& spec,
            std::vector<KernelLeaf>& out_leaves,
            std::vector<ParameterBinding>& out_bindings,
            std::size_t& next_param_idx
        )
        {
            if (const auto* leaf = std::get_if<KernelLeaf>(&spec.node_))
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
                return LeafRef{.leaf_idx = leaf_idx, .param_begin = p_begin, .param_count = p_count};
            }
            if (std::holds_alternative<KernelSpec::SumNode>(spec.node_))
            {
                // Flatten nested binary sums into one flat SumTerm while preserving parameter order.
                std::vector<Term> flat;
                const std::size_t p_begin = next_param_idx;
                collect_sum(spec, flat, out_leaves, out_bindings, next_param_idx);
                return SumTerm{
                    .terms = std::move(flat),
                    .param_begin = p_begin,
                    .param_end = next_param_idx,
                };
            }
            // Flatten nested binary products into one flat ProductTerm while preserving parameter order.
            std::vector<Term> flat;
            const std::size_t p_begin = next_param_idx;
            collect_product(spec, flat, out_leaves, out_bindings, next_param_idx);
            return ProductTerm{
                .terms = std::move(flat),
                .param_begin = p_begin,
                .param_end = next_param_idx,
            };
        }

        /// @brief Associatively flattens nested `KernelSpec::SumNode` trees into `out_terms`.
        /// @details Traverses `lhs` then `rhs` so leaf and parameter ordering stays strictly depth-first left-to-right.
        /// @param spec Current `KernelSpec` node being flattened.
        /// @param out_terms Destination list of non-sum summand `Term` nodes.
        /// @param out_leaves Accumulator of compiled leaf objects.
        /// @param out_bindings Accumulator of global parameter bindings.
        /// @param next_param_idx Running global parameter counter.
        static void collect_sum(
            const KernelSpec& spec,
            std::vector<Term>& out_terms,
            std::vector<KernelLeaf>& out_leaves,
            std::vector<ParameterBinding>& out_bindings,
            std::size_t& next_param_idx
        )
        {
            if (const auto* s = std::get_if<KernelSpec::SumNode>(&spec.node_))
            {
                collect_sum(*s->lhs, out_terms, out_leaves, out_bindings, next_param_idx);
                collect_sum(*s->rhs, out_terms, out_leaves, out_bindings, next_param_idx);
                return;
            }
            out_terms.push_back(compile_node(spec, out_leaves, out_bindings, next_param_idx));
        }

        /// @brief Associatively flattens nested `KernelSpec::ProductNode` trees into `out_terms`.
        /// @details Traverses `lhs` then `rhs` so leaf and parameter ordering stays strictly depth-first left-to-right.
        /// @param spec Current `KernelSpec` node being flattened.
        /// @param out_terms Destination list of non-product factor `Term` nodes.
        /// @param out_leaves Accumulator of compiled leaf objects.
        /// @param out_bindings Accumulator of global parameter bindings.
        /// @param next_param_idx Running global parameter counter.
        static void collect_product(
            const KernelSpec& spec,
            std::vector<Term>& out_terms,
            std::vector<KernelLeaf>& out_leaves,
            std::vector<ParameterBinding>& out_bindings,
            std::size_t& next_param_idx
        )
        {
            if (const auto* p = std::get_if<KernelSpec::ProductNode>(&spec.node_))
            {
                collect_product(*p->lhs, out_terms, out_leaves, out_bindings, next_param_idx);
                collect_product(*p->rhs, out_terms, out_leaves, out_bindings, next_param_idx);
                return;
            }
            out_terms.push_back(compile_node(spec, out_leaves, out_bindings, next_param_idx));
        }

        /// @brief Verifies that every ARD leaf in `leaves` expects `x_dim` input features.
        /// @details Checks `RbfArdKernel`, `MaternArdKernel`, and `RationalQuadraticArdKernel` lengthscale counts.
        /// @param x_dim Feature dimensionality of the point coordinate matrix.
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
                    expected_dim = k_rq->lengthscales().size();
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
        /// @param dist Input squared-distance matrix view.
        /// @param out Reference output matrix view.
        /// @param uplo Active matrix triangle when `square == true`.
        /// @param square True for square self-covariance/derivative calls, false for cross-covariance calls.
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
            for (std::size_t col = 0; col < n_cols; ++col)
            {
                const std::size_t row_begin = (square && uplo == Triangle::Lower) ? col : 0U;
                const std::size_t row_end = (square && uplo == Triangle::Upper) ? (col + 1U) : n_rows;
                for (std::size_t row = row_begin; row < row_end; ++row)
                {
                    if (!std::isfinite(dist(row, col)))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                }
            }
            return {};
        }

        /// @brief Validates point matrix shape, square output shape, ARD dimensions, and coordinate finiteness.
        /// @details Enforces empty -> shape -> ARD dimension -> finite scan order before point-based tree evaluation.
        /// @param x Input point coordinate matrix of shape `(n, d)`.
        /// @param out Reference square output matrix of shape `(n, n)`.
        /// @return Empty expected on success, or validation error.
        /// @retval GprErrorCode::EmptyInput If `x` has zero rows or columns.
        /// @retval GprErrorCode::LengthMismatch If `out` is not `(x.rows(), x.rows())`.
        /// @retval GprErrorCode::DimensionMismatch If `x.cols()` mismatches an ARD leaf's dimensionality.
        /// @retval GprErrorCode::NonFiniteInput If any entry in `x` is non-finite.
        [[nodiscard]] std::expected<void, GprError> validate_points_call(MatrixRef<T> x, MatrixRef<T> out) const
        {
            if (x.rows() == 0 || x.cols() == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (out.rows() != x.rows() || out.cols() != x.rows())
            {
                return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
            }
            if (auto ard = validate_ard_dims(x.cols()); !ard.has_value())
            {
                return ard;
            }
            for (std::size_t col = 0; col < x.cols(); ++col)
            {
                for (std::size_t row = 0; row < x.rows(); ++row)
                {
                    if (!std::isfinite(x(row, col)))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                }
            }
            return {};
        }

        /// @brief Validates point matrices `x1` and `x2` and rectangular output `out` for cross-covariance evaluation.
        /// @details Enforces empty -> output shape -> matching feature dimension -> ARD dimension -> finiteness checks.
        /// @param x1 First point matrix of shape `(n1, d)`.
        /// @param x2 Second point matrix of shape `(n2, d)`.
        /// @param out Reference output matrix of shape `(n1, n2)`.
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
            for (std::size_t col = 0; col < x1.cols(); ++col)
            {
                for (std::size_t row = 0; row < x1.rows(); ++row)
                {
                    if (!std::isfinite(x1(row, col)))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                }
            }
            for (std::size_t col = 0; col < x2.cols(); ++col)
            {
                for (std::size_t row = 0; row < x2.rows(); ++row)
                {
                    if (!std::isfinite(x2(row, col)))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                }
            }
            return {};
        }

        // --- Evaluation helpers ---

        /// @brief Verifies that `nested` has enough buffer depth and element capacity for `level_idx`.
        /// @details Returns `GprErrorCode::WorkspaceTooSmall` without allocating if capacity is insufficient.
        /// @param nested Caller-supplied nested buffer pool.
        /// @param level_idx Required zero-based buffer level index.
        /// @param rows Required row capacity.
        /// @param cols Required column capacity.
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

        /// @brief Recursively evaluates a compiled `Term` into `out` from squared Euclidean distances.
        /// @details Evaluates the first child directly into `out`, folds subsequent leaf children through `scratch`,
        /// and folds nested composite children through `nested.level(level_idx, ...)`.
        /// @param term Compiled AST node to evaluate.
        /// @param dist Square squared-distance matrix.
        /// @param out Destination Gram matrix view.
        /// @param uplo Active matrix triangle to populate.
        /// @param scratch Reusable scratch matrix for direct leaf siblings.
        /// @param nested Pre-allocated intermediate buffer pool for nested subtrees.
        /// @param level_idx Current nesting depth inside `nested`.
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
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return detail::leaf_apply_dist(leaves[leaf->leaf_idx], dist, out, uplo, exp_mode);
            }
            const bool is_sum = std::holds_alternative<SumTerm>(term);
            const auto& terms = is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
            if (terms.empty())
            {
                return std::unexpected(GprError::unsupported_kernel_operation("composite kernel has no terms"));
            }
            // Evaluate the first child directly into `out`, then fold remaining siblings into `out`.
            if (auto res = eval_term_dist(terms[0], dist, out, uplo, scratch, nested, level_idx); !res.has_value())
            {
                return res;
            }
            for (std::size_t k = 1; k < terms.size(); ++k)
            {
                if (std::holds_alternative<LeafRef>(terms[k]))
                {
                    if (auto res = eval_term_dist(terms[k], dist, scratch, uplo, out, nested, level_idx);
                        !res.has_value())
                    {
                        return res;
                    }
                }
                else
                {
                    if (auto ok = ensure_nested(nested, level_idx, out.rows(), out.cols()); !ok.has_value())
                    {
                        return ok;
                    }
                    auto buf = nested.level(level_idx, out.rows(), out.cols());
                    if (auto res = eval_term_dist(terms[k], dist, buf, uplo, scratch, nested, level_idx + 1);
                        !res.has_value())
                    {
                        return res;
                    }
                    if (is_sum)
                    {
                        add_into(out, buf.as_ref(), uplo);
                    }
                    else
                    {
                        mul_into(out, buf.as_ref(), uplo);
                    }
                    continue;
                }
                if (is_sum)
                {
                    add_into(out, scratch.as_ref(), uplo);
                }
                else
                {
                    mul_into(out, scratch.as_ref(), uplo);
                }
            }
            return {};
        }

        /// @brief Evaluates a single compiled leaf at `leaf_idx` using caches when available.
        /// @details Prefers `DistanceCache::dist_view()` for distance leaves and `DistanceCache::ard_packed()` for ARD
        /// leaves; materializes square distances in `nested` for uncached `CustomKernel` leaves.
        /// @param leaf_idx Zero-based index into `leaves`.
        /// @param x Input point matrix of shape `(n, d)`.
        /// @param out Destination square matrix of shape `(n, n)`.
        /// @param uplo Active matrix triangle to populate.
        /// @param nested Pre-allocated buffer pool used if `CustomKernel` needs uncached squared distances.
        /// @param level_idx Nesting depth inside `nested` for `CustomKernel` distance buffer.
        /// @param cache Optional precomputed distance and ARD cache.
        /// @return Empty expected on success, or workspace/evaluation error.
        [[nodiscard]] std::expected<void, GprError> eval_leaf_at(
            std::size_t leaf_idx,
            MatrixRef<T> x,
            MatrixMut<T> out,
            Triangle uplo,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            const DistanceCache<T>* cache = nullptr
        ) const
        {
            const KernelLeaf& kl = leaves[leaf_idx];
            // Route to cached distance or packed ARD tables whenever compatible caches are provided.
            if (cache != nullptr)
            {
                if (cache->has_dist() && leaf_reads_distances(kl))
                {
                    return detail::leaf_apply_dist(kl, cache->dist_view(), out, uplo, exp_mode);
                }
                if (cache->has_ard() && leaf_needs_ard(kl))
                {
                    return detail::leaf_apply_ard_packed(
                        kl, cached_ls[leaf_idx], cache->ard_packed(), x.rows(), out, uplo, exp_mode
                    );
                }
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

        /// @brief Reconstructs the composite Gram matrix from precomputed per-leaf Gram matrices.
        /// @details Walks the flattened AST, copying leaf views directly into `out` and combining composite children
        /// via `nested` buffers without re-evaluating any kernel leaf.
        /// @param term Current compiled AST node.
        /// @param leaf_grams Precomputed Gram matrix views indexed by `leaf_idx`.
        /// @param out Destination matrix view.
        /// @param uplo Active matrix triangle to combine.
        /// @param nested Pre-allocated intermediate buffer pool for nested subtrees.
        /// @param level_idx Current nesting depth inside `nested`.
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
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                copy_into(out, leaf_grams[leaf->leaf_idx], uplo);
                return {};
            }
            const bool is_sum = std::holds_alternative<SumTerm>(term);
            const auto& terms = is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
            if (terms.empty())
            {
                return std::unexpected(GprError::unsupported_kernel_operation("composite kernel has no terms"));
            }
            if (auto res = eval_combine_term(terms[0], leaf_grams, out, uplo, nested, level_idx); !res.has_value())
            {
                return res;
            }
            for (std::size_t k = 1; k < terms.size(); ++k)
            {
                if (const auto* leaf = std::get_if<LeafRef>(&terms[k]))
                {
                    if (is_sum)
                    {
                        add_into(out, leaf_grams[leaf->leaf_idx], uplo);
                    }
                    else
                    {
                        mul_into(out, leaf_grams[leaf->leaf_idx], uplo);
                    }
                }
                else
                {
                    if (auto ok = ensure_nested(nested, level_idx, out.rows(), out.cols()); !ok.has_value())
                    {
                        return ok;
                    }
                    auto buf = nested.level(level_idx, out.rows(), out.cols());
                    if (auto res = eval_combine_term(terms[k], leaf_grams, buf, uplo, nested, level_idx + 1U);
                        !res.has_value())
                    {
                        return res;
                    }
                    if (is_sum)
                    {
                        add_into(out, buf.as_ref(), uplo);
                    }
                    else
                    {
                        mul_into(out, buf.as_ref(), uplo);
                    }
                }
            }
            return {};
        }

        /// @brief Recursively evaluates a compiled `Term` from point coordinates `x` (and optional `cache`).
        /// @details Evaluates `terms[0]` into `out`, leaf siblings via `scratch`, and composite siblings via `nested`.
        /// @param term Compiled AST node to evaluate.
        /// @param x Input point coordinate matrix.
        /// @param out Destination square Gram matrix view.
        /// @param uplo Active matrix triangle to populate.
        /// @param scratch Reusable scratch matrix for direct leaf siblings.
        /// @param nested Pre-allocated intermediate buffer pool for nested subtrees.
        /// @param level_idx Current nesting depth inside `nested`.
        /// @param cache Optional precomputed distance and ARD cache.
        /// @return Empty expected on success, or workspace/evaluation error.
        [[nodiscard]] std::expected<void, GprError> eval_term_points(
            const Term& term,
            MatrixRef<T> x,
            MatrixMut<T> out,
            Triangle uplo,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            const DistanceCache<T>* cache = nullptr
        ) const
        {
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return eval_leaf_at(leaf->leaf_idx, x, out, uplo, nested, level_idx, cache);
            }
            const bool is_sum = std::holds_alternative<SumTerm>(term);
            const auto& terms = is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
            if (terms.empty())
            {
                return std::unexpected(GprError::unsupported_kernel_operation("composite kernel has no terms"));
            }
            if (auto res = eval_term_points(terms[0], x, out, uplo, scratch, nested, level_idx, cache);
                !res.has_value())
            {
                return res;
            }
            for (std::size_t k = 1; k < terms.size(); ++k)
            {
                if (std::holds_alternative<LeafRef>(terms[k]))
                {
                    if (auto res = eval_term_points(terms[k], x, scratch, uplo, out, nested, level_idx, cache);
                        !res.has_value())
                    {
                        return res;
                    }
                    if (is_sum)
                    {
                        add_into(out, scratch.as_ref(), uplo);
                    }
                    else
                    {
                        mul_into(out, scratch.as_ref(), uplo);
                    }
                }
                else
                {
                    if (auto ok = ensure_nested(nested, level_idx, out.rows(), out.cols()); !ok.has_value())
                    {
                        return ok;
                    }
                    auto buf = nested.level(level_idx, out.rows(), out.cols());
                    if (auto res = eval_term_points(terms[k], x, buf, uplo, scratch, nested, level_idx + 1, cache);
                        !res.has_value())
                    {
                        return res;
                    }
                    if (is_sum)
                    {
                        add_into(out, buf.as_ref(), uplo);
                    }
                    else
                    {
                        mul_into(out, buf.as_ref(), uplo);
                    }
                }
            }
            return {};
        }

        /// @brief Recursively evaluates a compiled `Term` cross-covariance from rectangular squared distances.
        /// @details Always evaluates full rectangular matrices (`Triangle::Full`) using `scratch` and `nested`.
        /// @param term Compiled AST node to evaluate.
        /// @param dist Rectangular squared-distance matrix.
        /// @param out Destination cross-covariance matrix view.
        /// @param scratch Reusable scratch matrix for direct leaf siblings.
        /// @param nested Pre-allocated intermediate buffer pool for nested subtrees.
        /// @param level_idx Current nesting depth inside `nested`.
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
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return detail::leaf_apply_cross_dist(leaves[leaf->leaf_idx], dist, out, exp_mode);
            }
            const bool is_sum = std::holds_alternative<SumTerm>(term);
            const auto& terms = is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
            if (terms.empty())
            {
                return std::unexpected(GprError::unsupported_kernel_operation("composite kernel has no terms"));
            }
            if (auto res = eval_term_cross_dist(terms[0], dist, out, scratch, nested, level_idx); !res.has_value())
            {
                return res;
            }
            for (std::size_t k = 1; k < terms.size(); ++k)
            {
                if (std::holds_alternative<LeafRef>(terms[k]))
                {
                    if (auto res = eval_term_cross_dist(terms[k], dist, scratch, out, nested, level_idx);
                        !res.has_value())
                    {
                        return res;
                    }
                    if (is_sum)
                    {
                        add_into(out, scratch.as_ref(), Triangle::Full);
                    }
                    else
                    {
                        mul_into(out, scratch.as_ref(), Triangle::Full);
                    }
                }
                else
                {
                    if (auto ok = ensure_nested(nested, level_idx, out.rows(), out.cols()); !ok.has_value())
                    {
                        return ok;
                    }
                    auto buf = nested.level(level_idx, out.rows(), out.cols());
                    if (auto res = eval_term_cross_dist(terms[k], dist, buf, scratch, nested, level_idx + 1);
                        !res.has_value())
                    {
                        return res;
                    }
                    if (is_sum)
                    {
                        add_into(out, buf.as_ref(), Triangle::Full);
                    }
                    else
                    {
                        mul_into(out, buf.as_ref(), Triangle::Full);
                    }
                }
            }
            return {};
        }

        /// @brief Recursively evaluates a compiled `Term` cross-covariance from two point coordinate matrices.
        /// @details Uses `nested` to materialize cross squared distances for `CustomKernel` leaves and combines child
        /// cross-covariances across sums and products.
        /// @param term Compiled AST node to evaluate.
        /// @param x1 First point matrix of shape `(n1, d)`.
        /// @param x2 Second point matrix of shape `(n2, d)`.
        /// @param out Destination cross-covariance matrix of shape `(n1, n2)`.
        /// @param scratch Reusable scratch matrix for direct leaf siblings.
        /// @param nested Pre-allocated intermediate buffer pool for nested subtrees and custom distances.
        /// @param level_idx Current nesting depth inside `nested`.
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
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                const KernelLeaf& kl = leaves[leaf->leaf_idx];
                if (const auto* custom = std::get_if<CustomKernel>(&kl))
                {
                    if (auto ok = ensure_nested(nested, level_idx, x1.rows(), x2.rows()); !ok.has_value())
                    {
                        return ok;
                    }
                    auto dist_buf = nested.level(level_idx, x1.rows(), x2.rows());
                    fill_cross_sq_dist(x1, x2, dist_buf);
                    return custom->template term<T>().apply_cross(dist_buf.as_ref(), out);
                }
                return detail::leaf_apply_cross_points(kl, x1, x2, out, exp_mode, cached_ls[leaf->leaf_idx]);
            }
            const bool is_sum = std::holds_alternative<SumTerm>(term);
            const auto& terms = is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
            if (terms.empty())
            {
                return std::unexpected(GprError::unsupported_kernel_operation("composite kernel has no terms"));
            }
            if (auto res = eval_term_cross_points(terms[0], x1, x2, out, scratch, nested, level_idx); !res.has_value())
            {
                return res;
            }
            for (std::size_t k = 1; k < terms.size(); ++k)
            {
                if (std::holds_alternative<LeafRef>(terms[k]))
                {
                    if (auto res = eval_term_cross_points(terms[k], x1, x2, scratch, out, nested, level_idx);
                        !res.has_value())
                    {
                        return res;
                    }
                    if (is_sum)
                    {
                        add_into(out, scratch.as_ref(), Triangle::Full);
                    }
                    else
                    {
                        mul_into(out, scratch.as_ref(), Triangle::Full);
                    }
                }
                else
                {
                    if (auto ok = ensure_nested(nested, level_idx, out.rows(), out.cols()); !ok.has_value())
                    {
                        return ok;
                    }
                    auto buf = nested.level(level_idx, out.rows(), out.cols());
                    if (auto res = eval_term_cross_points(terms[k], x1, x2, buf, scratch, nested, level_idx + 1);
                        !res.has_value())
                    {
                        return res;
                    }
                    if (is_sum)
                    {
                        add_into(out, buf.as_ref(), Triangle::Full);
                    }
                    else
                    {
                        mul_into(out, buf.as_ref(), Triangle::Full);
                    }
                }
            }
            return {};
        }

        /// @brief Recursively evaluates the self-covariance diagonal $k(x_i, x_i)$ of a compiled `Term`.
        /// @details Evaluates `terms[0]` into `out` and subsequent terms into disjoint slices of `scratch_vec` indexed
        /// by `level_idx * out.size()` so zero heap allocations occur when caller scratch is provided.
        /// @param term Compiled AST node.
        /// @param x Input point coordinate matrix of shape `(n, d)`.
        /// @param out Output diagonal span of length `n`.
        /// @param scratch_vec Caller-supplied scratch vector sliced by recursion level.
        /// @param level_idx Current nesting depth into `scratch_vec`.
        /// @return Empty expected on success, or workspace/evaluation error.
        /// @retval GprErrorCode::LengthMismatch If `scratch_vec` is shorter than `(level_idx + 1) * out.size()`.
        [[nodiscard]] std::expected<void, GprError> eval_fill_diag(
            const Term& term, MatrixRef<T> x, std::span<T> out, std::span<T> scratch_vec, std::size_t level_idx
        ) const
        {
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return detail::leaf_fill_diag(leaves[leaf->leaf_idx], x, out, cached_ls[leaf->leaf_idx]);
            }
            const bool is_sum = std::holds_alternative<SumTerm>(term);
            const auto& terms = is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
            if (terms.empty())
            {
                return std::unexpected(GprError::unsupported_kernel_operation("composite kernel has no terms"));
            }
            if (auto res = eval_fill_diag(terms[0], x, out, scratch_vec, level_idx); !res.has_value())
            {
                return res;
            }
            if (terms.size() == 1U)
            {
                return {};
            }
            // Slice scratch_vec by recursion depth so sibling terms never clobber outer levels.
            const std::size_t slice_offset = level_idx * out.size();
            if (scratch_vec.size() < slice_offset + out.size())
            {
                return std::unexpected(GprError::length_mismatch("diagonal evaluation scratch buffer is too small"));
            }
            for (std::size_t k = 1; k < terms.size(); ++k)
            {
                const std::span<T> sub(scratch_vec.data() + slice_offset, out.size());
                if (auto res = eval_fill_diag(terms[k], x, sub, scratch_vec, level_idx + 1U); !res.has_value())
                {
                    return res;
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
            return {};
        }

        /// @brief Recursively evaluates the first log-hyperparameter derivative $\partial K / \partial \theta_p$
        /// from squared Euclidean distances.
        /// @details Routes sum nodes directly to the single summand owning `param_idx`, and applies the product rule
        /// $(\partial f_{\text{owner}} / \partial \theta_p) \odot \prod_{m \ne \text{owner}} f_m$ on product nodes.
        /// @param term Compiled AST node.
        /// @param dist Square squared-distance matrix.
        /// @param d_k Output derivative matrix view.
        /// @param param_idx Global hyperparameter index to differentiate with respect to.
        /// @param uplo Active matrix triangle to populate.
        /// @param scratch_a First reusable square scratch buffer.
        /// @param scratch_b Second reusable square scratch buffer.
        /// @param nested Pre-allocated intermediate buffer pool for nested subtrees.
        /// @param level_idx Current nesting depth inside `nested`.
        /// @return Empty expected on success, or index/workspace/evaluation error.
        /// @retval GprErrorCode::IndexOutOfRange If `param_idx` is not owned by any child of `term`.
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
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return detail::leaf_grad_dist(
                    leaves[leaf->leaf_idx], dist, d_k, param_idx - leaf->param_begin, uplo, exp_mode
                );
            }
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                // In a sum, only the summand containing param_idx contributes to dK/dtheta.
                for (const auto& child : s->terms)
                {
                    if (contains_param(child, param_idx))
                    {
                        return eval_grad_dist(
                            child, dist, d_k, param_idx, uplo, scratch_a, scratch_b, nested, level_idx
                        );
                    }
                }
                return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
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
                return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
            }
            // Differentiate the owning factor into d_k, then multiply by all sibling factor Grams.
            if (auto res =
                    eval_grad_dist(p.terms[owner], dist, d_k, param_idx, uplo, scratch_a, scratch_b, nested, level_idx);
                !res.has_value())
            {
                return res;
            }
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (m == owner)
                {
                    continue;
                }
                if (auto res = eval_term_dist(p.terms[m], dist, scratch_a, uplo, scratch_b, nested, level_idx);
                    !res.has_value())
                {
                    return res;
                }
                mul_into(d_k, scratch_a.as_ref(), uplo);
            }
            return {};
        }

        /// @brief Recursively evaluates $\partial K / \partial \theta_p$ from point coordinates (and optional cache).
        /// @details Uses cached distance or packed ARD derivatives on compatible leaves, materializes distances into
        /// `nested` for uncached `CustomKernel`, and folds sibling factors across sums and products.
        /// @param term Compiled AST node.
        /// @param x Input point matrix of shape `(n, d)`.
        /// @param d_k Output derivative matrix of shape `(n, n)`.
        /// @param param_idx Global hyperparameter index to differentiate with respect to.
        /// @param uplo Active matrix triangle to populate.
        /// @param scratch_a First reusable square scratch buffer.
        /// @param scratch_b Second reusable square scratch buffer.
        /// @param nested Pre-allocated intermediate buffer pool for nested subtrees.
        /// @param level_idx Current nesting depth inside `nested`.
        /// @param cache Optional precomputed distance and ARD cache.
        /// @return Empty expected on success, or index/workspace/evaluation error.
        /// @retval GprErrorCode::IndexOutOfRange If `param_idx` is not owned by any child of `term`.
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
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                const KernelLeaf& kl = leaves[leaf->leaf_idx];
                if (cache != nullptr)
                {
                    if (cache->has_dist() && leaf_reads_distances(kl))
                    {
                        return detail::leaf_grad_dist(
                            kl, cache->dist_view(), d_k, param_idx - leaf->param_begin, uplo, exp_mode
                        );
                    }
                    if (cache->has_ard() && leaf_needs_ard(kl))
                    {
                        return detail::leaf_grad_ard_packed(
                            kl, cached_ls[leaf->leaf_idx], cache->ard_packed(), x.rows(), d_k,
                            param_idx - leaf->param_begin, uplo, exp_mode
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
                    return custom->template term<T>().grad(dist_buf.as_ref(), d_k, param_idx - leaf->param_begin, uplo);
                }
                return detail::leaf_grad_points(
                    kl, x, d_k, param_idx - leaf->param_begin, uplo, exp_mode, cached_ls[leaf->leaf_idx]
                );
            }
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                // Route sum gradient directly to the child owning param_idx.
                for (const auto& child : s->terms)
                {
                    if (contains_param(child, param_idx))
                    {
                        return eval_grad_points(
                            child, x, d_k, param_idx, uplo, scratch_a, scratch_b, nested, level_idx, cache
                        );
                    }
                }
                return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
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
                return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
            }
            // Product rule: evaluate the owning factor's derivative, then scale by sibling factor values.
            if (auto res = eval_grad_points(
                    p.terms[owner], x, d_k, param_idx, uplo, scratch_a, scratch_b, nested, level_idx, cache
                );
                !res.has_value())
            {
                return res;
            }
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (m == owner)
                {
                    continue;
                }
                if (auto res = eval_term_points(p.terms[m], x, scratch_a, uplo, scratch_b, nested, level_idx, cache);
                    !res.has_value())
                {
                    return res;
                }
                mul_into(d_k, scratch_a.as_ref(), uplo);
            }
            return {};
        }

        /// @brief Recursively evaluates the second log-hyperparameter derivative $\partial^2 K / (\partial \theta_i
        /// \partial \theta_j)$ from squared Euclidean distances.
        /// @details Zeroes the active triangle when `i` and `j` belong to distinct summands of a `SumTerm`; on
        /// `ProductTerm`, computes $(\partial^2 f_m / \partial \theta_i \partial \theta_j) \prod_{r \ne m} f_r$ when
        /// co-located and $(\partial f_{m_i}/\partial \theta_i)(\partial f_{m_j}/\partial \theta_j) \prod_{r \ne
        /// m_i,m_j} f_r$ across distinct factors.
        /// @param term Compiled AST node.
        /// @param dist Square squared-distance matrix.
        /// @param d2_k Output second-derivative matrix view.
        /// @param i First global hyperparameter index.
        /// @param j Second global hyperparameter index.
        /// @param uplo Active matrix triangle to populate.
        /// @param scratch_a First reusable square scratch buffer.
        /// @param scratch_b Second reusable square scratch buffer.
        /// @param scratch_c Third reusable square scratch buffer.
        /// @param nested Pre-allocated intermediate buffer pool for nested subtrees.
        /// @param level_idx Current nesting depth inside `nested`.
        /// @return Empty expected on success, or index/workspace/evaluation error.
        /// @retval GprErrorCode::IndexOutOfRange If `i` or `j` is outside `term`'s parameter range.
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
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                return detail::leaf_hess_dist(
                    leaves[leaf->leaf_idx], dist, d2_k, i - leaf->param_begin, j - leaf->param_begin, uplo, exp_mode
                );
            }
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                for (const auto& child : s->terms)
                {
                    const bool has_i = contains_param(child, i);
                    const bool has_j = contains_param(child, j);
                    if (has_i && has_j)
                    {
                        return eval_hess_dist(
                            child, dist, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested, level_idx
                        );
                    }
                    if (has_i || has_j)
                    {
                        // Input shape, finiteness, and CoordMode were already validated at entry.
                        // Cross-derivatives between separate summands vanish identically.
                        for (std::size_t col = 0; col < d2_k.cols(); ++col)
                        {
                            for (std::size_t row = 0; row < d2_k.rows(); ++row)
                            {
                                if (uplo == Triangle::Lower && row < col)
                                {
                                    continue;
                                }
                                if (uplo == Triangle::Upper && row > col)
                                {
                                    continue;
                                }
                                d2_k(row, col) = T{0};
                            }
                        }
                        return {};
                    }
                }
                return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
            }
            const auto& p = std::get<ProductTerm>(term);
            std::size_t owner_i = p.terms.size();
            std::size_t owner_j = p.terms.size();
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (contains_param(p.terms[m], i))
                {
                    owner_i = m;
                }
                if (contains_param(p.terms[m], j))
                {
                    owner_j = m;
                }
            }
            if (owner_i == p.terms.size() || owner_j == p.terms.size())
            {
                return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
            }
            // Case 1: both i and j belong to the same factor -> second derivative of owner_i times remaining factors.
            if (owner_i == owner_j)
            {
                if (auto res = eval_hess_dist(
                        p.terms[owner_i], dist, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested, level_idx
                    );
                    !res.has_value())
                {
                    return res;
                }
                for (std::size_t m = 0; m < p.terms.size(); ++m)
                {
                    if (m == owner_i)
                    {
                        continue;
                    }
                    if (auto res = eval_term_dist(p.terms[m], dist, scratch_b, uplo, scratch_c, nested, level_idx);
                        !res.has_value())
                    {
                        return res;
                    }
                    mul_into(d2_k, scratch_b.as_ref(), uplo);
                }
                return {};
            }
            // Case 2: i and j belong to distinct factors -> product of first derivatives times remaining factors.
            MatrixMut<T> grad_rhs = scratch_a;
            MatrixMut<T> work_a = scratch_b;
            MatrixMut<T> work_b = scratch_c;
            if (auto res = eval_grad_dist(p.terms[owner_i], dist, d2_k, i, uplo, work_a, work_b, nested, level_idx);
                !res.has_value())
            {
                return res;
            }
            if (auto res = eval_grad_dist(p.terms[owner_j], dist, grad_rhs, j, uplo, work_a, work_b, nested, level_idx);
                !res.has_value())
            {
                return res;
            }
            mul_into(d2_k, grad_rhs.as_ref(), uplo);
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (m == owner_i || m == owner_j)
                {
                    continue;
                }
                if (auto res = eval_term_dist(p.terms[m], dist, scratch_b, uplo, scratch_c, nested, level_idx);
                    !res.has_value())
                {
                    return res;
                }
                mul_into(d2_k, scratch_b.as_ref(), uplo);
            }
            return {};
        }

        /// @brief Recursively evaluates $\partial^2 K / (\partial \theta_i \partial \theta_j)$ from point coordinates.
        /// @details Handles cached distance/ARD acceleration, `CustomKernel` distance materialization, cross-summand
        /// zeroing, and same-factor vs cross-factor product Hessian rules.
        /// @param term Compiled AST node.
        /// @param x Input point coordinate matrix of shape `(n, d)`.
        /// @param d2_k Output second-derivative matrix of shape `(n, n)`.
        /// @param i First global hyperparameter index.
        /// @param j Second global hyperparameter index.
        /// @param uplo Active matrix triangle to populate.
        /// @param scratch_a First reusable square scratch buffer.
        /// @param scratch_b Second reusable square scratch buffer.
        /// @param scratch_c Third reusable square scratch buffer.
        /// @param nested Pre-allocated intermediate buffer pool for nested subtrees.
        /// @param level_idx Current nesting depth inside `nested`.
        /// @param cache Optional precomputed distance and ARD cache.
        /// @return Empty expected on success, or index/workspace/evaluation error.
        /// @retval GprErrorCode::IndexOutOfRange If `i` or `j` is outside `term`'s parameter range.
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
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                const KernelLeaf& kl = leaves[leaf->leaf_idx];
                if (cache != nullptr)
                {
                    if (cache->has_dist() && leaf_reads_distances(kl))
                    {
                        return detail::leaf_hess_dist(
                            kl, cache->dist_view(), d2_k, i - leaf->param_begin, j - leaf->param_begin, uplo, exp_mode
                        );
                    }
                    if (cache->has_ard() && leaf_needs_ard(kl))
                    {
                        return detail::leaf_hess_ard_packed(
                            kl, cached_ls[leaf->leaf_idx], cache->ard_packed(), x.rows(), d2_k, i - leaf->param_begin,
                            j - leaf->param_begin, uplo, exp_mode
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
                    return custom->template term<T>().hess(
                        dist_buf.as_ref(), d2_k, i - leaf->param_begin, j - leaf->param_begin, uplo
                    );
                }
                return detail::leaf_hess_points(
                    kl, x, d2_k, i - leaf->param_begin, j - leaf->param_begin, uplo, exp_mode, cached_ls[leaf->leaf_idx]
                );
            }
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                for (const auto& child : s->terms)
                {
                    const bool has_i = contains_param(child, i);
                    const bool has_j = contains_param(child, j);
                    if (has_i && has_j)
                    {
                        return eval_hess_points(
                            child, x, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested, level_idx, cache
                        );
                    }
                    if (has_i || has_j)
                    {
                        // Input shape, finiteness, and all ARD dimensions were already validated at
                        // entry.
                        // Mixed partial derivatives across separate summands vanish identically.
                        for (std::size_t col = 0; col < d2_k.cols(); ++col)
                        {
                            for (std::size_t row = 0; row < d2_k.rows(); ++row)
                            {
                                if (uplo == Triangle::Lower && row < col)
                                {
                                    continue;
                                }
                                if (uplo == Triangle::Upper && row > col)
                                {
                                    continue;
                                }
                                d2_k(row, col) = T{0};
                            }
                        }
                        return {};
                    }
                }
                return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
            }
            const auto& p = std::get<ProductTerm>(term);
            std::size_t owner_i = p.terms.size();
            std::size_t owner_j = p.terms.size();
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (contains_param(p.terms[m], i))
                {
                    owner_i = m;
                }
                if (contains_param(p.terms[m], j))
                {
                    owner_j = m;
                }
            }
            if (owner_i == p.terms.size() || owner_j == p.terms.size())
            {
                return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
            }
            // Same-factor Hessian times remaining sibling factor Grams.
            if (owner_i == owner_j)
            {
                if (auto res = eval_hess_points(
                        p.terms[owner_i], x, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested, level_idx, cache
                    );
                    !res.has_value())
                {
                    return res;
                }
                for (std::size_t m = 0; m < p.terms.size(); ++m)
                {
                    if (m == owner_i)
                    {
                        continue;
                    }
                    if (auto res =
                            eval_term_points(p.terms[m], x, scratch_b, uplo, scratch_c, nested, level_idx, cache);
                        !res.has_value())
                    {
                        return res;
                    }
                    mul_into(d2_k, scratch_b.as_ref(), uplo);
                }
                return {};
            }
            // Cross-factor Hessian: multiply gradients of owner_i and owner_j by any remaining factors.
            MatrixMut<T> grad_rhs = scratch_a;
            MatrixMut<T> work_a = scratch_b;
            MatrixMut<T> work_b = scratch_c;
            if (auto res =
                    eval_grad_points(p.terms[owner_i], x, d2_k, i, uplo, work_a, work_b, nested, level_idx, cache);
                !res.has_value())
            {
                return res;
            }
            if (auto res =
                    eval_grad_points(p.terms[owner_j], x, grad_rhs, j, uplo, work_a, work_b, nested, level_idx, cache);
                !res.has_value())
            {
                return res;
            }
            mul_into(d2_k, grad_rhs.as_ref(), uplo);
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (m == owner_i || m == owner_j)
                {
                    continue;
                }
                if (auto res = eval_term_points(p.terms[m], x, scratch_b, uplo, scratch_c, nested, level_idx, cache);
                    !res.has_value())
                {
                    return res;
                }
                mul_into(d2_k, scratch_b.as_ref(), uplo);
            }
            return {};
        }

        // --- keep_plan and weighted_grads ---

        /// @brief Returns true if `term` is a `LeafRef` wrapping a `ConstantKernel`.
        /// @details Constant factors fold into a single scalar multiplier in product retention and adjoint contraction.
        /// @param term Compiled AST node.
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
        /// @param terms Child factors of a `ProductTerm`.
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
        /// @param terms Child factors of a `ProductTerm`.
        /// @return Combined scalar constant multiplier cast to `T`.
        [[nodiscard]] T constant_scale(const std::vector<Term>& terms) const noexcept
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
        /// @param term Compiled AST node.
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
        /// @param term Compiled AST node.
        /// @param keeps True if this node's varying factor Grams were pre-retained by `apply_keeping`.
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
        /// @param products Number of eligible top-level `ProductTerm` nodes to retain.
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
        /// @param products Number of eligible top-level `ProductTerm` nodes whose factor Grams are kept.
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
        /// @param leaf_idx Zero-based index into `leaves`.
        /// @param x Input point matrix of shape `(n, d)`.
        /// @param cache Optional precomputed distance and ARD cache.
        /// @param d_k Output derivative matrix view.
        /// @param local_idx Local hyperparameter index within `leaves[leaf_idx]`.
        /// @param uplo Active matrix triangle to write.
        /// @param nested Pre-allocated nested buffer pool for uncached `CustomKernel` distance construction.
        /// @param level_idx Current nesting depth inside `nested`.
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
        /// @param x Input point matrix of shape `(n, d)`.
        /// @param cache Optional precomputed distance and ARD cache.
        /// @param out Destination square Gram matrix of shape `(n, n)`.
        /// @param uplo Active matrix triangle to populate.
        /// @param scratch Reusable square scratch matrix.
        /// @param nested Pre-allocated intermediate buffer pool for nested subtrees.
        /// @param kept_bufs Pre-allocated retention buffer pool receiving kept factor Grams.
        /// @param products Maximum number of top-level multi-factor products to retain.
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
            std::size_t products
        ) const
        {
            const auto roots = root_terms();
            if (roots.empty())
            {
                return std::unexpected(GprError::unsupported_kernel_operation("composite kernel has no terms"));
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
                            if (auto res = eval_term_points(f, x, kbuf, uplo, tmp, nested, sub_level, cache);
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
                        if (auto res = eval_term_points(t, x, target, uplo, scratch, nested, 0U, cache);
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
                        if (auto res = eval_term_points(t, x, nbuf, uplo, scratch, nested, 1U, cache); !res.has_value())
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

        /// @brief Single-pass adjoint contraction walk computing $\langle W, \partial K / \partial \theta_p \rangle_F$
        /// for all parameters under `term`.
        /// @details Propagates the incoming lower-triangular weight matrix `weight` top-down: sum nodes forward
        /// `weight` unchanged; leaf nodes contract `weight` with each parameter derivative into `out_grads`; product
        /// nodes form per-factor adjoint weights $W_{\text{eff}} = \text{scale} \cdot W \odot \prod_{r \ne m} G_r$
        /// (reusing `kept_bufs` when `has_kept == true`) and optionally return $\langle W, K_{\text{term}} \rangle_F$
        /// when
        /// `need_value` is set or a sibling `ConstantKernel` requires its log-parameter gradient.
        /// @param term Compiled AST node being traversed.
        /// @param x Input point coordinate matrix of shape `(n, d)`.
        /// @param cache Optional precomputed distance and ARD cache.
        /// @param weight Incoming symmetric adjoint weight matrix (lower triangle read).
        /// @param out_grads Output span receiving $\langle W, \partial K / \partial \theta \rangle_F$ for `term`.
        /// @param kernel_scratch Reusable `n x n` scratch matrix for single-leaf gradient/value evaluation.
        /// @param nested Pre-allocated buffer pool for uncached `CustomKernel` distance matrices.
        /// @param weighted_bufs Pre-allocated `keep_plan` + walk buffer pool (`kept_bufs` followed by walk scratch).
        /// @param kept_offset Starting buffer index in `weighted_bufs` if `has_kept` is true.
        /// @param kept_count Number of retained factor Grams starting at `kept_offset`.
        /// @param has_kept True when `term`'s varying factor Grams were pre-retained in `weighted_bufs`.
        /// @param walk_cursor First free buffer index in `weighted_bufs` beyond retained Grams and outer frames.
        /// @param col_partials Per-column scratch array of length at least `n` for deterministic serial reductions.
        /// @param need_value True when an enclosing product needs $\langle W, K_{\text{term}} \rangle_F$ back.
        /// @return Contracted term value $\langle W, K_{\text{term}} \rangle_F$ (when `need_value` or needed by a
        /// sibling `ConstantKernel`), `0.0` otherwise, or an error.
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
            bool need_value
        ) const
        {
            const std::size_t n = x.rows();
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                // Sum rule is linear in W: pass identical `weight` to every summand and sum their contractions.
                std::size_t p_off = 0;
                double value = 0.0;
                for (const auto& child : s->terms)
                {
                    const auto [cb, ce] = param_range(child);
                    const std::size_t count = ce - cb;
                    auto res = walk_weighted(
                        child, x, cache, weight, out_grads.subspan(p_off, count), kernel_scratch, nested, weighted_bufs,
                        0, 0, false, walk_cursor, col_partials, need_value
                    );
                    if (!res.has_value())
                    {
                        return res;
                    }
                    value += *res;
                    p_off += count;
                }
                return value;
            }
            if (const auto* leaf = std::get_if<LeafRef>(&term))
            {
                const KernelLeaf& kl = leaves[leaf->leaf_idx];
                if (const auto* ck = std::get_if<ConstantKernel>(&kl))
                {
                    // Since d(c)/d(log c) = c, gradient and <W, K_leaf>_F are both c * sum_{i,j} W_ij.
                    const double val = ck->constant() * lower_sum(weight, col_partials);
                    out_grads[0] = val;
                    return val;
                }
                if (const auto* wk = std::get_if<WhiteKernel>(&kl))
                {
                    // WhiteKernel derivative and value live strictly on the diagonal: v * trace(W).
                    const double val = wk->variance() * lower_diag_sum(weight);
                    out_grads[0] = val;
                    return need_value ? val : 0.0;
                }
                for (std::size_t p = 0; p < leaf->param_count; ++p)
                {
                    if (auto res =
                            grad_leaf_at(leaf->leaf_idx, x, cache, kernel_scratch, p, Triangle::Lower, nested, 0);
                        !res.has_value())
                    {
                        return std::unexpected(std::move(res).error());
                    }
                    out_grads[p] = lower_frobenius(weight, kernel_scratch.as_ref(), col_partials);
                }
                if (need_value)
                {
                    if (auto res = eval_leaf_at(leaf->leaf_idx, x, kernel_scratch, Triangle::Lower, nested, 0, cache);
                        !res.has_value())
                    {
                        return std::unexpected(std::move(res).error());
                    }
                    return lower_frobenius(weight, kernel_scratch.as_ref(), col_partials);
                }
                return 0.0;
            }

            const auto& p = std::get<ProductTerm>(term);
            const std::size_t varying = varying_factors(p.terms);
            const T scale = constant_scale(p.terms);
            const bool has_const = (varying < p.terms.size());

            if (varying == 0U)
            {
                // All factors are ConstantKernel scalars; every constant parameter shares <W, prod c_k>_F.
                const double val = static_cast<double>(scale) * lower_sum(weight, col_partials);
                std::size_t p_off = 0;
                for (const auto& f : p.terms)
                {
                    const auto [fb, fe] = param_range(f);
                    out_grads[p_off] = val;
                    p_off += (fe - fb);
                }
                return val;
            }

            if (varying == 1U)
            {
                // Single varying factor scaled by constant_scale; request its contracted value if constants exist
                // because <scale * W, f_var>_F equals d/d(log c_k) <W, scale * f_var>_F for every constant factor.
                double val = 0.0;
                if (has_const)
                {
                    if (auto ok = ensure_nested(weighted_bufs, walk_cursor, n, n); !ok.has_value())
                    {
                        return std::unexpected(std::move(ok).error());
                    }
                }
                std::size_t p_off = 0;
                for (const auto& f : p.terms)
                {
                    const auto [fb, fe] = param_range(f);
                    const std::size_t count = fe - fb;
                    if (!is_constant_term(f))
                    {
                        MatrixRef<T> eff_w = weight;
                        if (has_const)
                        {
                            auto w_buf = weighted_bufs.level(walk_cursor, n, n);
                            copy_into(w_buf, weight, Triangle::Lower);
                            if (scale != T{1})
                            {
                                scale_lower(w_buf, scale);
                            }
                            eff_w = w_buf.as_ref();
                        }
                        auto res = walk_weighted(
                            f, x, cache, eff_w, out_grads.subspan(p_off, count), kernel_scratch, nested, weighted_bufs,
                            0, 0, false, has_const ? (walk_cursor + 1U) : walk_cursor, col_partials,
                            need_value || has_const
                        );
                        if (!res.has_value())
                        {
                            return res;
                        }
                        val = *res;
                    }
                    p_off += count;
                }
                if (has_const)
                {
                    std::size_t c_off = 0;
                    for (const auto& f : p.terms)
                    {
                        const auto [fb, fe] = param_range(f);
                        if (is_constant_term(f))
                        {
                            out_grads[c_off] = val;
                        }
                        c_off += (fe - fb);
                    }
                }
                return val;
            }

            // Two or more varying factors: reuse pre-kept factor Grams when available, otherwise compute each once.
            const std::size_t g_base = has_kept ? kept_offset : walk_cursor;
            const std::size_t next_cursor = has_kept ? walk_cursor : (walk_cursor + varying);
            if (!has_kept)
            {
                std::size_t v_idx = 0;
                for (const auto& f : p.terms)
                {
                    if (is_constant_term(f))
                    {
                        continue;
                    }
                    if (auto ok = ensure_nested(weighted_bufs, g_base + v_idx, n, n); !ok.has_value())
                    {
                        return std::unexpected(std::move(ok).error());
                    }
                    auto g_buf = weighted_bufs.level(g_base + v_idx, n, n);
                    if (auto res = eval_term_points(f, x, g_buf, Triangle::Lower, kernel_scratch, nested, 0, cache);
                        !res.has_value())
                    {
                        return std::unexpected(std::move(res).error());
                    }
                    ++v_idx;
                }
            }
            else if (kept_count < varying)
            {
                return std::unexpected(GprError::workspace_too_small());
            }

            if (auto ok = ensure_nested(weighted_bufs, next_cursor, n, n); !ok.has_value())
            {
                return std::unexpected(std::move(ok).error());
            }
            auto w_eff = weighted_bufs.level(next_cursor, n, n);

            double val = 0.0;
            std::size_t p_off = 0;
            std::size_t cur_v = 0;
            for (const auto& f : p.terms)
            {
                const auto [fb, fe] = param_range(f);
                const std::size_t count = fe - fb;
                if (!is_constant_term(f))
                {
                    // Build adjoint weight W_eff = scale * W * prod_{r != cur_v} G_r and recurse into factor cur_v.
                    copy_into(w_eff, weight, Triangle::Lower);
                    if (scale != T{1})
                    {
                        scale_lower(w_eff, scale);
                    }
                    for (std::size_t r = 0; r < varying; ++r)
                    {
                        if (r == cur_v)
                        {
                            continue;
                        }
                        mul_into(w_eff, weighted_bufs.level(g_base + r, n, n).as_ref(), Triangle::Lower);
                    }
                    if (auto res = walk_weighted(
                            f, x, cache, w_eff.as_ref(), out_grads.subspan(p_off, count), kernel_scratch, nested,
                            weighted_bufs, 0, 0, false, next_cursor + 1U, col_partials, false
                        );
                        !res.has_value())
                    {
                        return res;
                    }
                    ++cur_v;
                }
                p_off += count;
            }

            if (need_value || has_const)
            {
                // Contract scale * W * prod_{r >= 1} G_r with G_0 to obtain <W, K_prod>_F for constants/parents.
                copy_into(w_eff, weight, Triangle::Lower);
                if (scale != T{1})
                {
                    scale_lower(w_eff, scale);
                }
                for (std::size_t r = 1U; r < varying; ++r)
                {
                    mul_into(w_eff, weighted_bufs.level(g_base + r, n, n).as_ref(), Triangle::Lower);
                }
                val = lower_frobenius(w_eff.as_ref(), weighted_bufs.level(g_base, n, n).as_ref(), col_partials);
                if (has_const)
                {
                    std::size_t c_off = 0;
                    for (const auto& f : p.terms)
                    {
                        const auto [fb, fe] = param_range(f);
                        if (is_constant_term(f))
                        {
                            out_grads[c_off] = val;
                        }
                        c_off += (fe - fb);
                    }
                }
            }
            return val;
        }
    };

    // --- CompiledKernel<T> public methods ---

    /// @brief Constructs a compiled kernel from its private `Impl` state.
    /// @details Internal constructor invoked by `CompiledKernel::from_spec`.
    /// @tparam T Storage floating-point scalar type.
    /// @param impl Owning pointer to compiled AST state.
    template <KernelScalarConcept T>
    CompiledKernel<T>::CompiledKernel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl))
    {
    }

    /// @brief Move-constructs a `CompiledKernel<T>`.
    /// @details Transfers ownership of the internal `Impl` pointer without copying leaves or bindings.
    /// @tparam T Storage floating-point scalar type.
    /// @param other Source compiled kernel moved from.
    template <KernelScalarConcept T> CompiledKernel<T>::CompiledKernel(CompiledKernel&& other) noexcept = default;

    /// @brief Move-assigns a `CompiledKernel<T>`.
    /// @details Transfers ownership of the internal `Impl` pointer without copying leaves or bindings.
    /// @tparam T Storage floating-point scalar type.
    /// @param other Source compiled kernel moved from.
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
    /// @param other Compiled kernel instance to clone.
    template <KernelScalarConcept T>
    CompiledKernel<T>::CompiledKernel(const CompiledKernel& other) : impl_(std::make_unique<Impl>(*other.impl_))
    {
    }

    /// @brief Copy-assigns a `CompiledKernel<T>` with independent mutable leaf state.
    /// @details Deep-copies `other`'s `Impl` state while guarding against self-assignment.
    /// @tparam T Storage floating-point scalar type.
    /// @param other Compiled kernel instance to clone.
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
    /// @param spec Declaration-layer kernel expression tree.
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
    /// @param mode Exponential evaluation backend to use.
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

    /// @brief Allocates a `NestedBuffers<T>` workspace sized for this compiled kernel and matrix shape `(rows, cols)`.
    /// @details Pre-allocates `nested_depth()` contiguous matrices.
    /// @tparam T Storage floating-point scalar type.
    /// @param rows Matrix row capacity.
    /// @param cols Matrix column capacity.
    /// @return Pre-sized `NestedBuffers<T>` instance.
    template <KernelScalarConcept T>
    NestedBuffers<T> CompiledKernel<T>::make_nested_buffers(std::size_t rows, std::size_t cols) const
    {
        return NestedBuffers<T>(impl_->nested_depth, rows, cols);
    }

    /// @brief Writes all flattened log-space hyperparameters into `out`.
    /// @details Collects parameter slices across `leaves` in depth-first left-to-right order.
    /// @tparam T Storage floating-point scalar type.
    /// @param out Destination span of length `num_params()`.
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
    /// @param params Log-space hyperparameter span of length `num_params()`.
    /// @return Empty expected on success, or length/bounds error leaving state unchanged.
    /// @retval GprErrorCode::LengthMismatch If `params.size() != num_params()`.
    /// @retval GprErrorCode::InvalidHyperparameter If any value violates its open parameter interval.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::set_params(std::span<const double> params)
    {
        if (auto cnt = require_count(params.size(), impl_->bindings.size(), "params"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        // Stage updates on a temporary vector so failed validation leaves all leaves unchanged.
        std::vector<KernelLeaf> updated = impl_->leaves;
        std::size_t offset = 0;
        for (auto& leaf : updated)
        {
            const std::size_t count = detail::leaf_num_params(leaf);
            if (auto res = detail::leaf_set_params(leaf, params.subspan(offset, count)); !res.has_value())
            {
                return res;
            }
            offset += count;
        }
        impl_->leaves = std::move(updated);
        impl_->refresh_cached_ls();
        return {};
    }

    /// @brief Writes open hyperparameter intervals for all bound parameters into `out`.
    /// @details Iterates over `leaves` in depth-first left-to-right order.
    /// @tparam T Storage floating-point scalar type.
    /// @param out Destination span of length `num_params()`.
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
    /// @param dist Input square squared-distance matrix.
    /// @param out Destination square Gram matrix view.
    /// @param uplo Active matrix triangle (`Lower`, `Upper`, or `Full`).
    /// @param scratch Non-aliasing scratch matrix matching `out` shape.
    /// @param nested Pre-allocated nested buffer pool for composite subtrees.
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
    /// @param dist Input square squared-distance matrix.
    /// @param out Destination square Gram matrix view.
    /// @param uplo Active matrix triangle (`Lower`, `Upper`, or `Full`).
    /// @param scratch Non-aliasing scratch matrix matching `out` shape.
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
    /// @param x Input point coordinate matrix of shape `(n, d)`.
    /// @param out Destination square Gram matrix of shape `(n, n)`.
    /// @param uplo Active matrix triangle (`Lower`, `Upper`, or `Full`).
    /// @param scratch Non-aliasing scratch matrix of shape `(n, n)`.
    /// @param nested Pre-allocated nested buffer pool.
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
    /// @param x Input point coordinate matrix of shape `(n, d)`.
    /// @param out Destination square Gram matrix of shape `(n, n)`.
    /// @param uplo Active matrix triangle (`Lower`, `Upper`, or `Full`).
    /// @param scratch Non-aliasing scratch matrix of shape `(n, n)`.
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
    /// @param dist Input rectangular squared-distance matrix of shape `(n1, n2)`.
    /// @param out Destination cross-covariance matrix of shape `(n1, n2)`.
    /// @param scratch Non-aliasing scratch matrix of shape `(n1, n2)`.
    /// @param nested Pre-allocated nested buffer pool.
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
    /// @param dist Input rectangular squared-distance matrix.
    /// @param out Destination cross-covariance matrix view.
    /// @param scratch Non-aliasing scratch matrix matching `out` shape.
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
    /// @param x1 First point matrix of shape `(n1, d)`.
    /// @param x2 Second point matrix of shape `(n2, d)`.
    /// @param out Destination cross-covariance matrix of shape `(n1, n2)`.
    /// @param scratch Non-aliasing scratch matrix of shape `(n1, n2)`.
    /// @param nested Pre-allocated nested buffer pool.
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
    /// @param x1 First point matrix of shape `(n1, d)`.
    /// @param x2 Second point matrix of shape `(n2, d)`.
    /// @param out Destination cross-covariance matrix of shape `(n1, n2)`.
    /// @param scratch Non-aliasing scratch matrix of shape `(n1, n2)`.
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
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param out Output diagonal span of length `n`.
    /// @param scratch Scratch vector span of length at least `(nested_depth() + 1) * n`.
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
        for (std::size_t col = 0; col < x.cols(); ++col)
        {
            for (std::size_t row = 0; row < x.rows(); ++row)
            {
                if (!std::isfinite(x(row, col)))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
        }
        return impl_->eval_fill_diag(impl_->root, x, out, scratch, 0);
    }

    /// @brief Fills the diagonal $k(x_i, x_i)$ for query points `x` using a temporary scratch buffer.
    /// @details Allocates `(nested_depth() + 1) * out.size()` elements and forwards to the zero-alloc `fill_diag`.
    /// @tparam T Storage floating-point scalar type.
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param out Output diagonal span of length `n`.
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
    /// @param dist Input square squared-distance matrix.
    /// @param d_k Output derivative matrix view.
    /// @param param_idx Global hyperparameter index in `[0, num_params())`.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch_a First non-aliasing scratch buffer matching `d_k` shape.
    /// @param scratch_b Second non-aliasing scratch buffer matching `d_k` shape.
    /// @param nested Pre-allocated nested buffer pool.
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
    /// @param dist Input square squared-distance matrix.
    /// @param d_k Output derivative matrix view.
    /// @param param_idx Global hyperparameter index in `[0, num_params())`.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch_a First non-aliasing scratch buffer matching `d_k` shape.
    /// @param scratch_b Second non-aliasing scratch buffer matching `d_k` shape.
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
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param d_k Output derivative matrix of shape `(n, n)`.
    /// @param param_idx Global hyperparameter index in `[0, num_params())`.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch_a First non-aliasing scratch buffer of shape `(n, n)`.
    /// @param scratch_b Second non-aliasing scratch buffer of shape `(n, n)`.
    /// @param nested Pre-allocated nested buffer pool.
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
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param d_k Output derivative matrix of shape `(n, n)`.
    /// @param param_idx Global hyperparameter index in `[0, num_params())`.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch_a First non-aliasing scratch buffer of shape `(n, n)`.
    /// @param scratch_b Second non-aliasing scratch buffer of shape `(n, n)`.
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
    /// @param dist Input square squared-distance matrix.
    /// @param d2_k Output second-derivative matrix view.
    /// @param i First global hyperparameter index in `[0, num_params())`.
    /// @param j Second global hyperparameter index in `[0, num_params())`.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch_a First non-aliasing scratch buffer.
    /// @param scratch_b Second non-aliasing scratch buffer.
    /// @param scratch_c Third non-aliasing scratch buffer.
    /// @param nested Pre-allocated nested buffer pool.
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
    /// @param dist Input square squared-distance matrix.
    /// @param d2_k Output second-derivative matrix view.
    /// @param i First global hyperparameter index.
    /// @param j Second global hyperparameter index.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch_a First non-aliasing scratch buffer.
    /// @param scratch_b Second non-aliasing scratch buffer.
    /// @param scratch_c Third non-aliasing scratch buffer.
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
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param d2_k Output second-derivative matrix of shape `(n, n)`.
    /// @param i First global hyperparameter index in `[0, num_params())`.
    /// @param j Second global hyperparameter index in `[0, num_params())`.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch_a First non-aliasing scratch buffer.
    /// @param scratch_b Second non-aliasing scratch buffer.
    /// @param scratch_c Third non-aliasing scratch buffer.
    /// @param nested Pre-allocated nested buffer pool.
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
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param d2_k Output second-derivative matrix of shape `(n, n)`.
    /// @param i First global hyperparameter index.
    /// @param j Second global hyperparameter index.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch_a First non-aliasing scratch buffer.
    /// @param scratch_b Second non-aliasing scratch buffer.
    /// @param scratch_c Third non-aliasing scratch buffer.
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
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param cache Precomputed scalar distance and packed ARD cache.
    /// @param out Destination square Gram matrix of shape `(n, n)`.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch Non-aliasing scratch buffer of shape `(n, n)`.
    /// @param nested Pre-allocated nested buffer pool.
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
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param cache Precomputed distance and ARD cache.
    /// @param out Destination square Gram matrix of shape `(n, n)`.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch Non-aliasing scratch buffer of shape `(n, n)`.
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
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param cache Precomputed distance and ARD cache.
    /// @param d_k Output derivative matrix of shape `(n, n)`.
    /// @param param_idx Global hyperparameter index in `[0, num_params())`.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch_a First non-aliasing scratch buffer.
    /// @param scratch_b Second non-aliasing scratch buffer.
    /// @param nested Pre-allocated nested buffer pool.
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
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param cache Precomputed distance and ARD cache.
    /// @param d_k Output derivative matrix of shape `(n, n)`.
    /// @param param_idx Global hyperparameter index.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch_a First non-aliasing scratch buffer.
    /// @param scratch_b Second non-aliasing scratch buffer.
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
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param cache Precomputed distance and ARD cache.
    /// @param d2_k Output second-derivative matrix of shape `(n, n)`.
    /// @param i First global hyperparameter index.
    /// @param j Second global hyperparameter index.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch_a First non-aliasing scratch buffer.
    /// @param scratch_b Second non-aliasing scratch buffer.
    /// @param scratch_c Third non-aliasing scratch buffer.
    /// @param nested Pre-allocated nested buffer pool.
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
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param cache Precomputed distance and ARD cache.
    /// @param d2_k Output second-derivative matrix of shape `(n, n)`.
    /// @param i First global hyperparameter index.
    /// @param j Second global hyperparameter index.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch_a First non-aliasing scratch buffer.
    /// @param scratch_b Second non-aliasing scratch buffer.
    /// @param scratch_c Third non-aliasing scratch buffer.
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
    /// @param param_idx Global hyperparameter index.
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
    /// @param leaf_idx Zero-based leaf index in `[0, num_leaves())`.
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param out Destination square leaf Gram matrix of shape `(n, n)`.
    /// @param uplo Active matrix triangle to populate.
    /// @param nested Pre-allocated nested buffer pool.
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
    /// @param leaf_idx Zero-based leaf index in `[0, num_leaves())`.
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param cache Precomputed distance and ARD cache.
    /// @param out Destination square leaf Gram matrix of shape `(n, n)`.
    /// @param uplo Active matrix triangle to populate.
    /// @param nested Pre-allocated nested buffer pool.
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

    /// @brief Combines precomputed per-leaf Gram matrices into the full composite Gram matrix.
    /// @details Validates output squareness, non-aliasing scratch, `leaf_grams.size() == num_leaves()`, and every
    /// leaf Gram's `(n, n)` shape before walking `root` via `eval_combine_term`.
    /// @tparam T Storage floating-point scalar type.
    /// @param leaf_grams Span of precomputed square Gram matrix views ordered by leaf index.
    /// @param out Destination composite Gram matrix of shape `(n, n)`.
    /// @param uplo Active matrix triangle to combine.
    /// @param scratch Non-aliasing scratch matrix of shape `(n, n)`.
    /// @param nested Pre-allocated nested buffer pool.
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
    /// @param products Number of top-level multi-factor products to retain.
    /// @return Number of `n x n` buffers needed for retained factor Grams.
    template <KernelScalarConcept T> std::size_t CompiledKernel<T>::kept_buffers(std::size_t products) const noexcept
    {
        return impl_->kept_grams_count(products);
    }

    /// @brief Returns total `n x n` buffers required in `weighted_bufs` when calling `weighted_grads`.
    /// @details Equals `kept_grams_count(products) + walk_buffers_count(products)`, partitioning `weighted_bufs` into a
    /// retained-Gram prefix followed by reverse-walk working buffers.
    /// @tparam T Storage floating-point scalar type.
    /// @param products Number of top-level multi-factor products retained during forward evaluation.
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
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param out Destination square Gram matrix of shape `(n, n)`.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch Non-aliasing scratch buffer of shape `(n, n)`.
    /// @param nested Pre-allocated intermediate buffer pool.
    /// @param kept_bufs Retention buffer pool of depth at least `kept_buffers(products)`.
    /// @param products Number of top-level multi-factor products to retain.
    /// @return Empty expected on success, or validation/workspace/evaluation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledKernel<T>::apply_keeping(
        MatrixRef<T> x,
        MatrixMut<T> out,
        Triangle uplo,
        MatrixMut<T> scratch,
        NestedBuffers<T>& nested,
        NestedBuffers<T>& kept_bufs,
        std::size_t products
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
        return impl_->eval_keeping_impl(x, nullptr, out, uplo, scratch, nested, kept_bufs, products);
    }

    /// @brief Evaluates the training Gram matrix using `DistanceCache<T>` while saving `keep_plan` factor Grams.
    /// @details Combines distance/ARD cache acceleration with factor Gram retention in `kept_bufs`.
    /// @tparam T Storage floating-point scalar type.
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param cache Precomputed distance and ARD cache.
    /// @param out Destination square Gram matrix of shape `(n, n)`.
    /// @param uplo Active matrix triangle to populate.
    /// @param scratch Non-aliasing scratch buffer of shape `(n, n)`.
    /// @param nested Pre-allocated intermediate buffer pool.
    /// @param kept_bufs Retention buffer pool of depth at least `kept_buffers(products)`.
    /// @param products Number of top-level multi-factor products to retain.
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
        std::size_t products
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
        return impl_->eval_keeping_impl(x, &cache, out, uplo, scratch, nested, kept_bufs, products);
    }

    /// @brief Computes $\langle W, \partial K / \partial \theta_p \rangle_F$ for all parameters in a single tree pass.
    /// @details Walks top-level `root_terms()`, matching each multi-factor `ProductTerm` with its retained factor Grams
    /// in the prefix `[0, kept_total)` of `weighted_bufs` and delegating adjoint contraction to `walk_weighted` using
    /// working slots starting at `kept_total`.
    /// @tparam T Storage floating-point scalar type.
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param cache Optional precomputed distance and ARD cache (`nullptr` for uncached evaluation).
    /// @param weight_lower Lower triangle of the symmetric adjoint matrix $W$ of shape `(n, n)`.
    /// @param out_grads Output span of length `num_params()` receiving $\langle W, \partial K / \partial \theta_p
    /// \rangle_F$.
    /// @param kernel_scratch Non-aliasing square scratch buffer of shape `(n, n)`.
    /// @param nested Pre-allocated intermediate buffer pool.
    /// @param weighted_bufs Pre-allocated pool containing kept factor Grams in `[0, kept_buffers(products))` followed
    /// by walk scratch up to `weighted_buffers(products)`.
    /// @param products Number of top-level multi-factor products previously retained in `weighted_bufs`.
    /// @param col_partials Per-column partial sum buffer of length at least `n` for deterministic reduction.
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
        std::span<double> col_partials
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
                has_kept ? kept_off : 0U, has_kept ? v : 0U, has_kept, kept_total, col_partials, false
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
