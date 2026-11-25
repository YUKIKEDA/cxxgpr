// SPDX-License-Identifier: Apache-2.0
/// @file blas_backend.cpp
/// @brief Implementation of internal dense linear algebra wrappers across MKL, OpenBLAS, and BuiltIn SIMD backends.

#include "linalg/blas_backend.hpp"

#include "cxxgpr/core/views.hpp"
#include "cxxgpr/linalg/parallel.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>

#ifdef _OPENMP
#    include <omp.h>
#endif

#ifdef CXXGPR_HAS_MKL
#    include <mkl.h>
#    include <mkl_cblas.h>
#    include <mkl_lapacke.h>
#    include <mkl_service.h>
#elifdef CXXGPR_HAS_OPENBLAS
#    if __has_include(<cblas.h>)
#        include <cblas.h>
#    elif __has_include(<openblas/cblas.h>)
#        include <openblas/cblas.h>
#    else
#        error "OpenBLAS selected but <cblas.h> could not be located."
#    endif
#endif

#ifdef CXXGPR_HAS_BUILTIN_SIMD_KERNELS
#    include "linalg/packed_blas3.hpp"
#    include "linalg/simd_blas3.hpp"
#endif

namespace cxxgpr::detail
{

    namespace
    {

        /// @brief Column-major mutable Eigen map with dynamic dimensions and runtime outer stride.
        /// @tparam T Scalar floating-point element type.
        template <typename T>
        using ColMap =
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<>>;

        /// @brief Column-major immutable Eigen map with dynamic dimensions and runtime outer stride.
        /// @tparam T Scalar floating-point element type.
        template <typename T>
        using ConstColMap = Eigen::
            Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<>>;

#ifdef CXXGPR_HAS_MKL
        /// @brief Vendor BLAS integer type matching Intel oneMKL CBLAS signatures.
        using BlasInt = MKL_INT;

        /// @brief Scoped thread-local MKL thread-count limiter for small matrices.
        /// @details Prevents libiomp5md worker synchronization overhead on small BLAS/LAPACK calls while keeping
        /// outer OpenMP regions and sibling threads untouched.
        class MklLocalThreadsGuard
        {
        public:
            /// @brief Constructs the guard and optionally forces thread-local MKL execution to one thread.
            /// @param[in] force_single True to restrict the calling thread's MKL calls to one thread.
            explicit MklLocalThreadsGuard(bool force_single) noexcept :
                active_(force_single),
                prev_(force_single ? mkl_set_num_threads_local(1) : 0)
            {
            }

            /// @brief Restores the previous thread-local MKL thread setting on scope exit.
            ~MklLocalThreadsGuard() noexcept
            {
                if (active_)
                {
                    (void)mkl_set_num_threads_local(prev_);
                }
            }

            /// @brief Copy construction is deleted for thread-local MKL guard instances.
            MklLocalThreadsGuard(const MklLocalThreadsGuard&) = delete;
            /// @brief Copy assignment is deleted for thread-local MKL guard instances.
            /// @return Deleted copy-assignment reference.
            MklLocalThreadsGuard& operator=(const MklLocalThreadsGuard&) = delete;

        private:
            /// @brief True when thread-local MKL thread count was overridden on entry.
            bool active_{false};
            /// @brief Saved thread-local MKL thread count restored on destruction.
            int prev_{0};
        };
#elifdef CXXGPR_HAS_OPENBLAS
        /// @brief Vendor BLAS integer type matching OpenBLAS CBLAS signatures.
        using BlasInt = blasint;
#endif

        /// @brief Panel block size used by right-looking blocked Cholesky when LAPACKE is unavailable.
        constexpr std::size_t k_cholesky_panel_block = 64U;

        /// @brief Unblocked column-major lower-triangular Cholesky panel factorization over `n x n`.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] n Panel order.
        /// @param[in,out] a Column-major symmetric positive definite panel overwritten with lower factor `L`.
        /// @param[in] lda Leading dimension of `a`.
        /// @return Zero on success, or the 1-based failing pivot column index.
        template <typename T> std::size_t panel_cholesky_lower(std::size_t n, T* a, std::size_t lda) noexcept
        {
            for (std::size_t j = 0U; j < n; ++j)
            {
                // Compute diagonal L_jj after subtracting prior column contributions.
                T* const col_j = a + (j * lda);
                T diag = col_j[j];
                for (std::size_t k = 0U; k < j; ++k)
                {
                    const T l_jk = a[(k * lda) + j];
                    diag -= l_jk * l_jk;
                }
                if (!(diag > T{0}) || !std::isfinite(diag))
                {
                    return j + 1U;
                }
                const T l_jj = std::sqrt(diag);
                col_j[j] = l_jj;
                const T inv_l_jj = T{1} / l_jj;
                // Accumulate unit-stride trailing column updates down column k before scaling column j.
                for (std::size_t k = 0U; k < j; ++k)
                {
                    const T* const col_k = a + (k * lda);
                    const T l_jk = col_k[j];
                    for (std::size_t i = j + 1U; i < n; ++i)
                    {
                        col_j[i] -= col_k[i] * l_jk;
                    }
                }
                for (std::size_t i = j + 1U; i < n; ++i)
                {
                    col_j[i] *= inv_l_jj;
                }
            }
            return 0U;
        }

        /// @brief Blocked right-looking lower-triangular Cholesky factorization backed by BLAS TRSM and SYRK.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] n Matrix order.
        /// @param[in,out] a Column-major symmetric matrix overwritten in lower triangle with `L`.
        /// @param[in] lda Leading dimension of `a`.
        /// @return Zero on success, or the 1-based failing pivot column index.
        template <typename T> std::size_t blocked_cholesky_lower_impl(std::size_t n, T* a, std::size_t lda) noexcept
        {
            for (std::size_t k = 0U; k < n; k += k_cholesky_panel_block)
            {
                const std::size_t jb = std::min(k_cholesky_panel_block, n - k);
                const std::size_t panel_fail = panel_cholesky_lower(jb, a + (k * lda) + k, lda);
                if (panel_fail != 0U)
                {
                    return k + panel_fail;
                }
                if (k + jb < n)
                {
                    // Solve trailing subdiagonal block L21 and apply symmetric rank-k update to trailing A22.
                    const std::size_t tail = n - (k + jb);
                    T* const l21 = a + (k * lda) + (k + jb);
                    const T* const l11 = a + (k * lda) + k;
                    blas_trsm_lower(true, true, false, tail, jb, T{1}, l11, lda, l21, lda);
                    T* const a22 = a + ((k + jb) * lda) + (k + jb);
                    blas_syrk_lower(false, tail, jb, T{-1}, l21, lda, T{1}, a22, lda);
                }
            }
            return 0U;
        }

        /// @brief Generic Eigen fallback implementation of `blas_gemm`.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] trans_a Transpose flag for A.
        /// @param[in] trans_b Transpose flag for B.
        /// @param[in] m Rows of op(A) and C.
        /// @param[in] n Columns of op(B) and C.
        /// @param[in] k Inner contraction dimension.
        /// @param[in] alpha Multiplier for product.
        /// @param[in] a Column-major matrix A.
        /// @param[in] lda Leading dimension of A.
        /// @param[in] b Column-major matrix B.
        /// @param[in] ldb Leading dimension of B.
        /// @param[in] beta Multiplier for C.
        /// @param[in,out] c Column-major output matrix C.
        /// @param[in] ldc Leading dimension of C.
        template <typename T>
        void eigen_gemm_impl(
            bool trans_a,
            bool trans_b,
            std::size_t m,
            std::size_t n,
            std::size_t k,
            T alpha,
            const T* a,
            std::size_t lda,
            const T* b,
            std::size_t ldb,
            T beta,
            T* c,
            std::size_t ldc
        ) noexcept
        {
            const auto rows_a = static_cast<Eigen::Index>(trans_a ? k : m);
            const auto cols_a = static_cast<Eigen::Index>(trans_a ? m : k);
            const auto rows_b = static_cast<Eigen::Index>(trans_b ? n : k);
            const auto cols_b = static_cast<Eigen::Index>(trans_b ? k : n);
            ConstColMap<T> a_mat(a, rows_a, cols_a, Eigen::OuterStride<>(static_cast<Eigen::Index>(lda)));
            ConstColMap<T> b_mat(b, rows_b, cols_b, Eigen::OuterStride<>(static_cast<Eigen::Index>(ldb)));
            ColMap<T> c_mat(
                c, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n),
                Eigen::OuterStride<>(static_cast<Eigen::Index>(ldc))
            );
            if (beta == T{0})
            {
                c_mat.setZero();
            }
            else if (beta != T{1})
            {
                c_mat *= beta;
            }
            if (trans_a && trans_b)
            {
                c_mat.noalias() += alpha * a_mat.transpose() * b_mat.transpose();
            }
            else if (trans_a)
            {
                c_mat.noalias() += alpha * a_mat.transpose() * b_mat;
            }
            else if (trans_b)
            {
                c_mat.noalias() += alpha * a_mat * b_mat.transpose();
            }
            else
            {
                c_mat.noalias() += alpha * a_mat * b_mat;
            }
        }

        /// @brief Generic Eigen fallback implementation of `blas_syrk_lower`.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] trans_a If true uses A^T * A, else A * A^T.
        /// @param[in] n Order of C.
        /// @param[in] k Inner contraction dimension.
        /// @param[in] alpha Multiplier for rank-k update.
        /// @param[in] a Factor matrix A.
        /// @param[in] lda Leading dimension of A.
        /// @param[in] beta Multiplier for existing C.
        /// @param[in,out] c Lower-triangular matrix C.
        /// @param[in] ldc Leading dimension of C.
        template <typename T>
        void eigen_syrk_lower_impl(
            bool trans_a,
            std::size_t n,
            std::size_t k,
            T alpha,
            const T* a,
            std::size_t lda,
            T beta,
            T* c,
            std::size_t ldc
        ) noexcept
        {
            ColMap<T> c_mat(
                c, static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n),
                Eigen::OuterStride<>(static_cast<Eigen::Index>(ldc))
            );
            if (beta == T{0})
            {
                c_mat.template triangularView<Eigen::Lower>().setZero();
            }
            else if (beta != T{1})
            {
                for (std::size_t col = 0U; col < n; ++col)
                {
                    T* const c_col = c + (col * ldc);
                    for (std::size_t row = col; row < n; ++row)
                    {
                        c_col[row] *= beta;
                    }
                }
            }
            if (k == 0U || alpha == T{0})
            {
                return;
            }
            const auto rows_a = static_cast<Eigen::Index>(trans_a ? k : n);
            const auto cols_a = static_cast<Eigen::Index>(trans_a ? n : k);
            ConstColMap<T> a_mat(a, rows_a, cols_a, Eigen::OuterStride<>(static_cast<Eigen::Index>(lda)));
            if (trans_a)
            {
                c_mat.template selfadjointView<Eigen::Lower>().rankUpdate(a_mat.transpose(), alpha);
            }
            else
            {
                c_mat.template selfadjointView<Eigen::Lower>().rankUpdate(a_mat, alpha);
            }
        }

        /// @brief Pure column-major fallback implementation of `blas_symv_lower`.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] n Order of A, x, and y.
        /// @param[in] alpha Multiplier for A * x.
        /// @param[in] a Lower-triangular matrix A.
        /// @param[in] lda Leading dimension of A.
        /// @param[in] x Input vector.
        /// @param[in] beta Multiplier for y.
        /// @param[in,out] y Output vector.
        template <typename T>
        void eigen_symv_lower_impl(
            std::size_t n, T alpha, const T* a, std::size_t lda, const T* x, T beta, T* y
        ) noexcept
        {
            if (beta == T{0})
            {
                std::fill_n(y, n, T{0});
            }
            else if (beta != T{1})
            {
                for (std::size_t i = 0U; i < n; ++i)
                {
                    y[i] *= beta;
                }
            }
            if (alpha == T{0})
            {
                return;
            }
            for (std::size_t col = 0U; col < n; ++col)
            {
                const T* const a_col = a + (col * lda);
                const T ax = alpha * x[col];
                y[col] += a_col[col] * ax;
                T acc = T{0};
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    const T v = a_col[row];
                    y[row] += v * ax;
                    acc += v * x[row];
                }
                y[col] += alpha * acc;
            }
        }

        /// @brief Generic Eigen fallback implementation of `blas_trsm_lower`.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] right_side Solves X * op(L) = alpha * B if true, else op(L) * X = alpha * B.
        /// @param[in] trans_a Transpose flag for L.
        /// @param[in] unit_diag Unit diagonal assumption for L.
        /// @param[in] m Rows of B.
        /// @param[in] n Columns of B.
        /// @param[in] alpha Multiplier for B.
        /// @param[in] a Triangular matrix L.
        /// @param[in] lda Leading dimension of L.
        /// @param[in,out] b Matrix B overwritten with solution.
        /// @param[in] ldb Leading dimension of B.
        template <typename T>
        void eigen_trsm_lower_impl(
            bool right_side,
            bool trans_a,
            bool unit_diag,
            std::size_t m,
            std::size_t n,
            T alpha,
            const T* a,
            std::size_t lda,
            T* b,
            std::size_t ldb
        ) noexcept
        {
            const std::size_t dim_a = right_side ? n : m;
            ConstColMap<T> a_mat(
                a, static_cast<Eigen::Index>(dim_a), static_cast<Eigen::Index>(dim_a),
                Eigen::OuterStride<>(static_cast<Eigen::Index>(lda))
            );
            ColMap<T> b_mat(
                b, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n),
                Eigen::OuterStride<>(static_cast<Eigen::Index>(ldb))
            );
            if (alpha != T{1})
            {
                b_mat *= alpha;
            }
            if (unit_diag)
            {
                if (right_side)
                {
                    if (trans_a)
                    {
                        a_mat.template triangularView<Eigen::UnitLower>()
                            .transpose()
                            .template solveInPlace<Eigen::OnTheRight>(b_mat);
                    }
                    else
                    {
                        a_mat.template triangularView<Eigen::UnitLower>().template solveInPlace<Eigen::OnTheRight>(
                            b_mat
                        );
                    }
                }
                else if (trans_a)
                {
                    a_mat.template triangularView<Eigen::UnitLower>().transpose().solveInPlace(b_mat);
                }
                else
                {
                    a_mat.template triangularView<Eigen::UnitLower>().solveInPlace(b_mat);
                }
            }
            else
            {
                if (right_side)
                {
                    if (trans_a)
                    {
                        a_mat.template triangularView<Eigen::Lower>()
                            .transpose()
                            .template solveInPlace<Eigen::OnTheRight>(b_mat);
                    }
                    else
                    {
                        a_mat.template triangularView<Eigen::Lower>().template solveInPlace<Eigen::OnTheRight>(b_mat);
                    }
                }
                else if (trans_a)
                {
                    a_mat.template triangularView<Eigen::Lower>().transpose().solveInPlace(b_mat);
                }
                else
                {
                    a_mat.template triangularView<Eigen::Lower>().solveInPlace(b_mat);
                }
            }
        }

        /// @brief Generic Eigen fallback implementation of `blas_gemv`.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] trans_a Transpose flag for A.
        /// @param[in] m Rows of A.
        /// @param[in] n Columns of A.
        /// @param[in] alpha Multiplier for op(A) * x.
        /// @param[in] a Column-major matrix A.
        /// @param[in] lda Leading dimension of A.
        /// @param[in] x Input vector.
        /// @param[in] beta Multiplier for y.
        /// @param[in,out] y Output vector.
        template <typename T>
        void eigen_gemv_impl(
            bool trans_a, std::size_t m, std::size_t n, T alpha, const T* a, std::size_t lda, const T* x, T beta, T* y
        ) noexcept
        {
            ConstColMap<T> a_mat(
                a, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n),
                Eigen::OuterStride<>(static_cast<Eigen::Index>(lda))
            );
            const auto len_x = static_cast<Eigen::Index>(trans_a ? m : n);
            const auto len_y = static_cast<Eigen::Index>(trans_a ? n : m);
            const Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> x_vec(x, len_x);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> y_vec(y, len_y);
            if (beta == T{0})
            {
                y_vec.setZero();
            }
            else if (beta != T{1})
            {
                y_vec *= beta;
            }
            if (trans_a)
            {
                y_vec.noalias() += alpha * (a_mat.transpose() * x_vec);
            }
            else
            {
                y_vec.noalias() += alpha * (a_mat * x_vec);
            }
        }

#ifdef CXXGPR_HAS_BUILTIN_SIMD_KERNELS

        /// @brief Column count shared by direct Cholesky panels and their scratch arrays.
        constexpr std::size_t k_direct_panel_block = 32U;

        /// @brief Factors diagonal submatrix `a[k:k+bs, k:k+bs]` in place via Eigen::LLT and validates diagonal pivots.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in,out] a Target n x n matrix view.
        /// @param[in] k Starting row and column index of the submatrix.
        /// @param[in] bs Submatrix dimension.
        /// @return Zero on success, or 1-based failing column index in `a`.
        template <typename T>
        [[nodiscard]] std::size_t factor_subblock_eigen(MatrixMut<T> a, std::size_t k, std::size_t bs) noexcept
        {
            const auto bs_idx = static_cast<Eigen::Index>(bs);
            ColMap<T> sub_map(
                &a(k, k), bs_idx, bs_idx, Eigen::OuterStride<>(static_cast<Eigen::Index>(a.col_stride()))
            );
            using RefType =
                Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<>>;
            Eigen::LLT<RefType, Eigen::Lower> llt(sub_map);
            if (llt.info() != Eigen::Success)
            {
                return k + 1U;
            }
            for (std::size_t i = 0U; i < bs; ++i)
            {
                const T diag = a(k + i, k + i);
                if (!std::isfinite(diag) || diag <= T{0})
                {
                    return k + i + 1U;
                }
            }
            return 0U;
        }

        /// @brief Solves lower-panel rows `[r_begin, r_end)` x cols `[k, k + bs)` against the factored diagonal block.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in,out] a Target n x n matrix view.
        /// @param[in] k Starting column of the active panel.
        /// @param[in] bs Panel width (at most 32).
        /// @param[in] r_begin Inclusive starting row owned by the calling thread.
        /// @param[in] r_end Exclusive ending row owned by the calling thread.
        template <typename T>
        void solve_panel_rows(
            MatrixMut<T> a, std::size_t k, std::size_t bs, std::size_t r_begin, std::size_t r_end
        ) noexcept
        {
            constexpr std::size_t row_tile = 64U;
            std::array<const T*, k_direct_panel_block> l_ptrs{};
            std::array<T, k_direct_panel_block> coeffs{};
            for (std::size_t r_tile = r_begin; r_tile < r_end; r_tile += row_tile)
            {
                const std::size_t r_stop = std::min(r_tile + row_tile, r_end);
                const std::size_t tile_len = r_stop - r_tile;
                for (std::size_t col = 0U; col < bs; ++col)
                {
                    const std::size_t gc = k + col;
                    T* const col_c = a.col(gc).data();
                    for (std::size_t p = 0U; p < col; ++p)
                    {
                        l_ptrs[p] = a.col(k + p).data();
                        coeffs[p] = a(gc, k + p);
                    }
                    if (col > 0U)
                    {
                        rank_k_update_1col_simd(col_c, l_ptrs.data(), col, coeffs.data(), r_tile, r_stop);
                    }
                    const T inv_diag = T{1} / col_c[gc];
                    scale_in_place_simd(col_c + r_tile, tile_len, inv_diag);
                }
            }
        }

        /// @brief Applies the trailing symmetric rank-bs update to lower-triangular columns `[j_begin, j_end)`.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in,out] a Target n x n matrix view updated in place.
        /// @param[in] k Starting column of the active panel.
        /// @param[in] bs Panel width (at most 32).
        /// @param[in] j_begin Inclusive first trailing column assigned to this thread.
        /// @param[in] j_end Exclusive last trailing column assigned to this thread.
        /// @param[in] n Matrix order.
        template <typename T>
        void rank_k_lower_update(
            MatrixMut<T> a, std::size_t k, std::size_t bs, std::size_t j_begin, std::size_t j_end, std::size_t n
        ) noexcept
        {
            constexpr std::size_t row_tile = 256U;
            std::array<const T*, k_direct_panel_block> l_ptrs{};
            std::array<T, k_direct_panel_block> v0_arr{};
            std::array<T, k_direct_panel_block> v1_arr{};
            std::array<T, k_direct_panel_block> v2_arr{};
            std::array<T, k_direct_panel_block> v3_arr{};

            std::size_t j = j_begin;
            for (; j + 4U <= j_end; j += 4U)
            {
                const std::size_t j0 = j;
                const std::size_t j1 = j + 1U;
                const std::size_t j2 = j + 2U;
                const std::size_t j3 = j + 3U;
                T* const col0 = a.col(j0).data();
                T* const col1 = a.col(j1).data();
                T* const col2 = a.col(j2).data();
                T* const col3 = a.col(j3).data();

                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const T* const lp = a.col(k + p).data();
                    const T v0 = lp[j0];
                    const T v1 = lp[j1];
                    const T v2 = lp[j2];
                    l_ptrs[p] = lp;
                    v0_arr[p] = v0;
                    v1_arr[p] = v1;
                    v2_arr[p] = v2;
                    v3_arr[p] = lp[j3];
                    col0[j0] = scalar_fnmadd(v0, v0, col0[j0]);
                    col0[j1] = scalar_fnmadd(v1, v0, col0[j1]);
                    col0[j2] = scalar_fnmadd(v2, v0, col0[j2]);
                    col1[j1] = scalar_fnmadd(v1, v1, col1[j1]);
                    col1[j2] = scalar_fnmadd(v2, v1, col1[j2]);
                    col2[j2] = scalar_fnmadd(v2, v2, col2[j2]);
                }

                for (std::size_t r_start = j3; r_start < n; r_start += row_tile)
                {
                    const std::size_t r_stop = std::min(r_start + row_tile, n);
                    rank_k_update_4cols_simd(
                        col0, col1, col2, col3, l_ptrs.data(), bs, v0_arr.data(), v1_arr.data(), v2_arr.data(),
                        v3_arr.data(), r_start, r_stop
                    );
                }
            }
            for (; j < j_end; ++j)
            {
                T* const col_j = a.col(j).data();
                for (std::size_t p = 0U; p < bs; ++p)
                {
                    const T* const lp = a.col(k + p).data();
                    l_ptrs[p] = lp;
                    v0_arr[p] = lp[j];
                }
                for (std::size_t r_start = j; r_start < n; r_start += row_tile)
                {
                    const std::size_t r_stop = std::min(r_start + row_tile, n);
                    rank_k_update_1col_simd(col_j, l_ptrs.data(), bs, v0_arr.data(), r_start, r_stop);
                }
            }
        }

        /// @brief Applies one trailing symmetric rank-bs lower tile update inside packed Cholesky factorization.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in,out] a Target n x n matrix view updated in place.
        /// @param[in] k Starting column of the factored panel.
        /// @param[in] bs Factored panel width.
        /// @param[in] row_start Starting row and column index of the trailing submatrix.
        /// @param[in] tail Dimension of the trailing submatrix.
        /// @param[in] tile Block tile extent.
        /// @param[in] n_blocks Number of lower-triangular blocks along the trailing dimension.
        /// @param[in] tile_idx Linear lower tile index.
        /// @param[in,out] last_bj Cached block column index for `packed_b` reuse.
        /// @param[out] packed_a Per-thread packed A buffer.
        /// @param[in,out] packed_b Per-thread packed B buffer.
        template <typename T>
        inline void apply_cholesky_trailing_tile(
            MatrixMut<T> a,
            std::size_t k,
            std::size_t bs,
            std::size_t row_start,
            std::size_t tail,
            std::size_t tile,
            std::size_t n_blocks,
            std::size_t tile_idx,
            std::size_t& last_bj,
            T* packed_a,
            T* packed_b
        ) noexcept
        {
            const auto coords = lower_tile_coords(tile_idx, n_blocks);
            const std::size_t bi = coords.row_block;
            const std::size_t bj = coords.col_block;
            const std::size_t ib = bi * tile;
            const std::size_t jb = bj * tile;
            const std::size_t rm = std::min(tile, tail - ib);
            const std::size_t cn = std::min(tile, tail - jb);
            if (bj != last_bj)
            {
                pack_panel_b_colmajor_t(&a(row_start + jb, k), a.col_stride(), bs, cn, packed_b);
                last_bj = bj;
            }
            pack_panel_a_colmajor(&a(row_start + ib, k), a.col_stride(), rm, bs, packed_a);
            T* const c_dst = &a(row_start + ib, row_start + jb);
            if (bi == bj)
            {
                syrk_packed_lower_diag_sub(rm, bs, packed_a, packed_b, c_dst, a.col_stride());
            }
            else
            {
                gemm_packed_sub(rm, cn, bs, packed_a, packed_b, c_dst, a.col_stride());
            }
        }

        /// @brief BuiltIn three-tier cache-aware SIMD + OpenMP lower Cholesky factorization.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] n Order of symmetric matrix A.
        /// @param[in,out] a_ptr Pointer to column-major matrix A overwritten with L.
        /// @param[in] lda Leading dimension of A.
        /// @return Zero on success, or 1-based failing column index.
        template <typename T>
        [[nodiscard]] std::size_t builtin_cholesky_lower_impl(std::size_t n, T* a_ptr, std::size_t lda) noexcept
        {
            MatrixMut<T> a(a_ptr, n, n, lda);
            if (n >= 384U)
            {
                std::size_t fail_idx = 0U;
                constexpr std::size_t panel_block = k_blas3_kc;
                constexpr std::size_t tail_cutoff = 64U;
#    ifdef _OPENMP
#        pragma omp parallel if (linalg::openmp_max_threads() > 1U) shared(a, fail_idx, n)
                {
                    alignas(64) std::array<T, k_blas3_pack_a_cap<T>> packed_a{};
                    alignas(64) std::array<T, k_blas3_pack_b_cap> packed_b{};
                    for (std::size_t k = 0U; k < n; k += panel_block)
                    {
                        const std::size_t rem = n - k;
                        if (rem <= tail_cutoff)
                        {
#        pragma omp single
                            {
                                fail_idx = factor_subblock_eigen(a, k, rem);
                            }
                            break;
                        }
                        const std::size_t bs = std::min(panel_block, rem);
#        pragma omp single
                        {
                            fail_idx = factor_subblock_eigen(a, k, bs);
                        }
                        if (fail_idx != 0U)
                        {
                            break;
                        }
                        const std::size_t row_start = k + bs;
                        const std::size_t tail = n - row_start;
                        const std::size_t n_strips = (tail + k_blas3_mc - 1U) / k_blas3_mc;
                        const auto n_strips_s = static_cast<std::ptrdiff_t>(n_strips);
#        pragma omp for schedule(dynamic, 1)
                        for (std::ptrdiff_t s = 0; s < n_strips_s; ++s)
                        {
                            const std::size_t r_begin = row_start + (static_cast<std::size_t>(s) * k_blas3_mc);
                            const std::size_t r_end = std::min(r_begin + k_blas3_mc, n);
                            solve_panel_rows_packed(a, k, bs, r_begin, r_end);
                        }

                        const std::size_t tile = (tail >= 384U) ? k_blas3_mc : 24U;
                        const std::size_t n_blocks = (tail + tile - 1U) / tile;
                        const std::size_t total_tiles = (n_blocks * (n_blocks + 1U)) / 2U;
                        const auto total_tiles_s = static_cast<std::ptrdiff_t>(total_tiles);
                        std::size_t last_bj = n_blocks;
#        pragma omp for schedule(dynamic, 1)
                        for (std::ptrdiff_t t_s = 0; t_s < total_tiles_s; ++t_s)
                        {
                            apply_cholesky_trailing_tile(
                                a, k, bs, row_start, tail, tile, n_blocks, static_cast<std::size_t>(t_s), last_bj,
                                packed_a.data(), packed_b.data()
                            );
                        }
                    }
                }
#    else
                alignas(64) std::array<T, k_blas3_pack_a_cap<T>> packed_a{};
                alignas(64) std::array<T, k_blas3_pack_b_cap> packed_b{};
                for (std::size_t k = 0U; k < n; k += panel_block)
                {
                    const std::size_t rem = n - k;
                    if (rem <= tail_cutoff)
                    {
                        fail_idx = factor_subblock_eigen(a, k, rem);
                        break;
                    }
                    const std::size_t bs = std::min(panel_block, rem);
                    fail_idx = factor_subblock_eigen(a, k, bs);
                    if (fail_idx != 0U)
                    {
                        break;
                    }
                    const std::size_t row_start = k + bs;
                    const std::size_t tail = n - row_start;
                    solve_panel_rows_packed(a, k, bs, row_start, n);

                    const std::size_t tile = (tail >= 384U) ? k_blas3_mc : 24U;
                    const std::size_t n_blocks = (tail + tile - 1U) / tile;
                    const std::size_t total_tiles = (n_blocks * (n_blocks + 1U)) / 2U;
                    std::size_t last_bj = n_blocks;
                    for (std::size_t t = 0U; t < total_tiles; ++t)
                    {
                        apply_cholesky_trailing_tile(
                            a, k, bs, row_start, tail, tile, n_blocks, t, last_bj, packed_a.data(), packed_b.data()
                        );
                    }
                }
#    endif
                return fail_idx;
            }

            if (n >= 256U)
            {
                std::size_t fail_idx = 0U;
                constexpr std::size_t tail_cutoff = 64U;
                constexpr std::size_t row_granule = 16U;
#    ifdef _OPENMP
#        pragma omp parallel if (linalg::openmp_max_threads() > 1U) shared(a, fail_idx, n)
                {
                    const auto tid = static_cast<std::size_t>(omp_get_thread_num());
                    const auto nthreads = static_cast<std::size_t>(omp_get_num_threads());
                    for (std::size_t k = 0U; k < n; k += k_direct_panel_block)
                    {
                        const std::size_t rem = n - k;
                        if (rem <= tail_cutoff)
                        {
#        pragma omp single
                            {
                                fail_idx = factor_subblock_eigen(a, k, rem);
                            }
                            break;
                        }
                        const std::size_t bs = std::min(k_direct_panel_block, rem);
#        pragma omp single
                        {
                            fail_idx = factor_subblock_eigen(a, k, bs);
                        }
                        if (fail_idx != 0U)
                        {
                            break;
                        }
                        const std::size_t row_start = k + bs;
                        const std::size_t tail = n - row_start;
                        const std::size_t n_gran = (tail + row_granule - 1U) / row_granule;
                        const std::size_t chunk_begin =
                            row_start + std::min(tail, ((n_gran * tid) / nthreads) * row_granule);
                        const std::size_t chunk_end =
                            row_start + std::min(tail, ((n_gran * (tid + 1U)) / nthreads) * row_granule);
                        if (chunk_begin < chunk_end)
                        {
                            solve_panel_rows(a, k, bs, chunk_begin, chunk_end);
                        }
#        pragma omp barrier
                        const std::size_t rel_begin = linalg::lower_block_start(tail, tid, nthreads);
                        const std::size_t rel_end = linalg::lower_block_start(tail, tid + 1U, nthreads);
                        rank_k_lower_update(a, k, bs, row_start + rel_begin, row_start + rel_end, n);
#        pragma omp barrier
                    }
                }
#    else
                for (std::size_t k = 0U; k < n; k += k_direct_panel_block)
                {
                    const std::size_t rem = n - k;
                    if (rem <= tail_cutoff)
                    {
                        fail_idx = factor_subblock_eigen(a, k, rem);
                        break;
                    }
                    const std::size_t bs = std::min(k_direct_panel_block, rem);
                    fail_idx = factor_subblock_eigen(a, k, bs);
                    if (fail_idx != 0U)
                    {
                        break;
                    }
                    const std::size_t row_start = k + bs;
                    const std::size_t tail = n - row_start;
                    (void)row_granule;
                    solve_panel_rows(a, k, bs, row_start, n);
                    rank_k_lower_update(a, k, bs, row_start, row_start + tail, n);
                }
#    endif
                return fail_idx;
            }

            return factor_subblock_eigen(a, 0U, n);
        }

#    ifdef CXXGPR_HAS_BLAS_BUILTIN

        /// @brief Smallest RHS batch handled by the packed forward solver.
        constexpr std::size_t k_min_packed_rhs = 6U;

        /// @brief Solves `L * X = B` across RHS columns `[col_begin, col_end)` via blocked SIMD updates.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] l Square lower Cholesky factor view.
        /// @param[in,out] rhs Right-hand side matrix view updated in place.
        /// @param[in] col_begin Inclusive start column index.
        /// @param[in] col_end Exclusive end column index.
        template <typename T>
        void solve_lower_cols(MatrixRef<T> l, MatrixMut<T> rhs, std::size_t col_begin, std::size_t col_end) noexcept
        {
            const std::size_t n = l.rows();
            if (n >= 64U && (col_end - col_begin) >= k_min_packed_rhs)
            {
                solve_lower_block_packed(l, rhs, 0U, col_begin, col_end);
                return;
            }
            constexpr std::size_t k_block = 32U;
            constexpr std::size_t row_tile = 256U;
            std::array<const T*, k_block> l_ptrs{};
            std::array<T, k_block> v0_arr{};
            std::array<T, k_block> v1_arr{};
            std::array<T, k_block> v2_arr{};
            std::array<T, k_block> v3_arr{};

            std::size_t c = col_begin;
            for (; c + 4U <= col_end; c += 4U)
            {
                T* const x0 = rhs.col(c).data();
                T* const x1 = rhs.col(c + 1U).data();
                T* const x2 = rhs.col(c + 2U).data();
                T* const x3 = rhs.col(c + 3U).data();

                for (std::size_t kb = 0U; kb < n; kb += k_block)
                {
                    const std::size_t ke = std::min(kb + k_block, n);
                    const std::size_t bs = ke - kb;
                    for (std::size_t p = 0U; p < bs; ++p)
                    {
                        const std::size_t k = kb + p;
                        const T* const lk = l.col(k).data();
                        const T lkk = lk[k];
                        const T v0p = x0[k] / lkk;
                        const T v1p = x1[k] / lkk;
                        const T v2p = x2[k] / lkk;
                        const T v3p = x3[k] / lkk;
                        x0[k] = v0p;
                        x1[k] = v1p;
                        x2[k] = v2p;
                        x3[k] = v3p;
                        l_ptrs[p] = lk;
                        v0_arr[p] = v0p;
                        v1_arr[p] = v1p;
                        v2_arr[p] = v2p;
                        v3_arr[p] = v3p;
                        for (std::size_t r = k + 1U; r < ke; ++r)
                        {
                            const T lrk = lk[r];
                            x0[r] = scalar_fnmadd(lrk, v0p, x0[r]);
                            x1[r] = scalar_fnmadd(lrk, v1p, x1[r]);
                            x2[r] = scalar_fnmadd(lrk, v2p, x2[r]);
                            x3[r] = scalar_fnmadd(lrk, v3p, x3[r]);
                        }
                    }
                    if (ke < n)
                    {
                        for (std::size_t r_start = ke; r_start < n; r_start += row_tile)
                        {
                            const std::size_t r_stop = std::min(r_start + row_tile, n);
                            rank_k_update_4cols_simd(
                                x0, x1, x2, x3, l_ptrs.data(), bs, v0_arr.data(), v1_arr.data(), v2_arr.data(),
                                v3_arr.data(), r_start, r_stop
                            );
                        }
                    }
                }
            }

            const std::size_t single_panel = (n <= 1024U) ? 8U : k_block;
            for (; c < col_end; ++c)
            {
                T* const xc = rhs.col(c).data();
                for (std::size_t kb = 0U; kb < n; kb += single_panel)
                {
                    const std::size_t ke = std::min(kb + single_panel, n);
                    const std::size_t bs = ke - kb;
                    for (std::size_t p = 0U; p < bs; ++p)
                    {
                        const std::size_t k = kb + p;
                        const T* const lk = l.col(k).data();
                        const T v0p = xc[k] / lk[k];
                        xc[k] = v0p;
                        l_ptrs[p] = lk;
                        v0_arr[p] = v0p;
                        for (std::size_t r = k + 1U; r < ke; ++r)
                        {
                            xc[r] = scalar_fnmadd(lk[r], v0p, xc[r]);
                        }
                    }
                    if (ke < n)
                    {
                        for (std::size_t r_start = ke; r_start < n; r_start += row_tile)
                        {
                            const std::size_t r_stop = std::min(r_start + row_tile, n);
                            rank_k_update_1col_simd(xc, l_ptrs.data(), bs, v0_arr.data(), r_start, r_stop);
                        }
                    }
                }
            }
        }

        /// @brief Resolves one 4-row backward-substitution step given precomputed tail dot products.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in,out] x Destination solution column pointer.
        /// @param[in] l0 Column `i0` of L.
        /// @param[in] l1 Column `i0 + 1` of L.
        /// @param[in] l2 Column `i0 + 2` of L.
        /// @param[in] l3 Column `i0 + 3` of L.
        /// @param[in] i0 Starting row index.
        /// @param[in] d0 Precomputed tail dot for `i0`.
        /// @param[in] d1 Precomputed tail dot for `i0 + 1`.
        /// @param[in] d2 Precomputed tail dot for `i0 + 2`.
        /// @param[in] d3 Precomputed tail dot for `i0 + 3`.
        template <typename T>
        inline void finish_backward_4rows(
            T* x, const T* l0, const T* l1, const T* l2, const T* l3, std::size_t i0, T d0, T d1, T d2, T d3
        ) noexcept
        {
            const std::size_t i1 = i0 + 1U;
            const std::size_t i2 = i0 + 2U;
            const std::size_t i3 = i0 + 3U;

            const T s3 = (x[i3] - d3) / l3[i3];
            x[i3] = s3;

            const T a2 = scalar_fmadd(l2[i3], s3, d2);
            const T s2 = (x[i2] - a2) / l2[i2];
            x[i2] = s2;

            const T a1 = scalar_fmadd(l1[i2], s2, scalar_fmadd(l1[i3], s3, d1));
            const T s1 = (x[i1] - a1) / l1[i1];
            x[i1] = s1;

            const T a0 = scalar_fmadd(l0[i1], s1, scalar_fmadd(l0[i2], s2, scalar_fmadd(l0[i3], s3, d0)));
            x[i0] = (x[i0] - a0) / l0[i0];
        }

        /// @brief Solves `L^T * X = B` across RHS columns `[col_begin, col_end)` via deterministic SIMD dots.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] l Square lower Cholesky factor view.
        /// @param[in,out] rhs Right-hand side matrix view updated in place.
        /// @param[in] col_begin Inclusive start column index.
        /// @param[in] col_end Exclusive end column index.
        template <typename T>
        void solve_lower_transpose_cols(
            MatrixRef<T> l, MatrixMut<T> rhs, std::size_t col_begin, std::size_t col_end
        ) noexcept
        {
            const std::size_t n = l.rows();
            std::size_t rem = n;
            while (rem >= 4U)
            {
                const std::size_t i0 = rem - 4U;
                const std::size_t i1 = rem - 3U;
                const std::size_t i2 = rem - 2U;
                const std::size_t i3 = rem - 1U;
                const std::size_t tail_start = rem;
                const std::size_t tail_len = n - tail_start;
                const T* const l0 = l.col(i0).data();
                const T* const l1 = l.col(i1).data();
                const T* const l2 = l.col(i2).data();
                const T* const l3 = l.col(i3).data();
                std::size_t c = col_begin;
                for (; c + 2U <= col_end; c += 2U)
                {
                    T* const x0 = rhs.col(c).data();
                    T* const x1 = rhs.col(c + 1U).data();
                    std::array<std::array<T, 2>, 4> dots{};
                    if (tail_len > 0U)
                    {
                        dot_4x2_simd(
                            l0 + tail_start, l1 + tail_start, l2 + tail_start, l3 + tail_start, x0 + tail_start,
                            x1 + tail_start, tail_len, dots
                        );
                    }
                    finish_backward_4rows(x0, l0, l1, l2, l3, i0, dots[0][0], dots[1][0], dots[2][0], dots[3][0]);
                    finish_backward_4rows(x1, l0, l1, l2, l3, i0, dots[0][1], dots[1][1], dots[2][1], dots[3][1]);
                }
                if (c < col_end)
                {
                    T* const xc = rhs.col(c).data();
                    std::array<T, 4> dots{};
                    if (tail_len > 0U)
                    {
                        dot_4x1_simd(
                            l0 + tail_start, l1 + tail_start, l2 + tail_start, l3 + tail_start, xc + tail_start,
                            tail_len, dots
                        );
                    }
                    finish_backward_4rows(xc, l0, l1, l2, l3, i0, dots[0], dots[1], dots[2], dots[3]);
                }
                rem -= 4U;
            }
            while (rem > 0U)
            {
                --rem;
                const T* const li = l.col(rem).data();
                const std::size_t start = rem + 1U;
                for (std::size_t c = col_begin; c < col_end; ++c)
                {
                    T* const xc = rhs.col(c).data();
                    const T acc = dot_1x1_simd(li + start, xc + start, n - start);
                    xc[rem] = (xc[rem] - acc) / li[rem];
                }
            }
        }

        /// @brief BuiltIn SIMD + OpenMP implementation of `blas_trsm_lower`.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] right_side Solves X * op(L) = alpha * B if true, else op(L) * X = alpha * B.
        /// @param[in] trans_a Transpose flag for L.
        /// @param[in] unit_diag Unit diagonal assumption for L.
        /// @param[in] m Rows of B.
        /// @param[in] n Columns of B.
        /// @param[in] alpha Multiplier for B.
        /// @param[in] a Pointer to L.
        /// @param[in] lda Leading dimension of L.
        /// @param[in,out] b Pointer to B overwritten with solution.
        /// @param[in] ldb Leading dimension of B.
        template <typename T>
        void builtin_trsm_lower_impl(
            bool right_side,
            bool trans_a,
            bool unit_diag,
            std::size_t m,
            std::size_t n,
            T alpha,
            const T* a,
            std::size_t lda,
            T* b,
            std::size_t ldb
        ) noexcept
        {
            if (right_side || unit_diag)
            {
                eigen_trsm_lower_impl(right_side, trans_a, unit_diag, m, n, alpha, a, lda, b, ldb);
                return;
            }

            if (alpha != T{1})
            {
                ColMap<T> b_mat(
                    b, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n),
                    Eigen::OuterStride<>(static_cast<Eigen::Index>(ldb))
                );
                b_mat *= alpha;
            }

            const MatrixRef<T> l(a, m, m, lda);
            MatrixMut<T> rhs(b, m, n, ldb);
            const double approx_flops = static_cast<double>(m) * static_cast<double>(m) * static_cast<double>(n);

            if (!trans_a)
            {
#        ifdef _OPENMP
                if (n >= 8U && approx_flops >= 4194304.0)
                {
                    if (n > 24U && linalg::openmp_max_threads() == 1U)
                    {
                        constexpr std::size_t strip = k_blas3_nc;
                        const std::size_t tail = n % strip;
                        const std::size_t packed_end = (tail < k_min_packed_rhs) ? n - tail : n;
                        solve_lower_cols(l, rhs, 0U, packed_end);
                        if (packed_end < n)
                        {
                            solve_lower_cols(l, rhs, packed_end, n);
                        }
                        return;
                    }
                    const std::size_t col_block = (n >= 512U) ? 24U : ((n >= 192U) ? 12U : 4U);
                    const std::size_t n_blocks = (n + col_block - 1U) / col_block;
                    const auto n_blocks_s = static_cast<std::ptrdiff_t>(n_blocks);
#            pragma omp parallel for schedule(dynamic, 1) if (linalg::openmp_max_threads() > 1U)
                    for (std::ptrdiff_t blk = 0; blk < n_blocks_s; ++blk)
                    {
                        const std::size_t col_begin = static_cast<std::size_t>(blk) * col_block;
                        const std::size_t col_end = std::min(col_begin + col_block, n);
                        solve_lower_cols(l, rhs, col_begin, col_end);
                    }
                    return;
                }
#        else
                (void)approx_flops;
#        endif
                solve_lower_cols(l, rhs, 0U, n);
                return;
            }

#        ifdef _OPENMP
            if (n >= 4U && linalg::openmp_max_threads() > 1U && approx_flops >= 4194304.0)
            {
                const std::size_t small_block = (n <= 12U) ? 2U : 4U;
                const std::size_t col_block = (n >= 64U) ? 16U : small_block;
                const std::size_t n_blocks = (n + col_block - 1U) / col_block;
                const auto n_blocks_s = static_cast<std::ptrdiff_t>(n_blocks);
#            pragma omp parallel for schedule(dynamic, 1)
                for (std::ptrdiff_t blk = 0; blk < n_blocks_s; ++blk)
                {
                    const std::size_t col_begin = static_cast<std::size_t>(blk) * col_block;
                    const std::size_t col_end = std::min(col_begin + col_block, n);
                    solve_lower_transpose_cols(l, rhs, col_begin, col_end);
                }
                return;
            }
#        endif
            solve_lower_transpose_cols(l, rhs, 0U, n);
        }

        /// @brief BuiltIn cache-packed OpenMP SIMD implementation of `blas_syrk_lower`.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] trans_a If true computes A^T * A, else A * A^T.
        /// @param[in] n Order of C.
        /// @param[in] k Inner contraction dimension.
        /// @param[in] alpha Multiplier for rank-k term.
        /// @param[in] a Pointer to matrix A.
        /// @param[in] lda Leading dimension of A.
        /// @param[in] beta Multiplier for existing C.
        /// @param[in,out] c Lower-triangular matrix C.
        /// @param[in] ldc Leading dimension of C.
        template <typename T>
        void builtin_syrk_lower_impl(
            bool trans_a,
            std::size_t n,
            std::size_t k,
            T alpha,
            const T* a,
            std::size_t lda,
            T beta,
            T* c,
            std::size_t ldc
        ) noexcept
        {
            if (n < 64U || k < 64U || alpha == T{0})
            {
                eigen_syrk_lower_impl(trans_a, n, k, alpha, a, lda, beta, c, ldc);
                return;
            }

            const std::size_t tile = (n >= 512U) ? k_blas3_mc : 24U;
            const std::size_t n_blocks = (n + tile - 1U) / tile;
            const std::size_t total_tiles = (n_blocks * (n_blocks + 1U)) / 2U;
            const auto total_tiles_s = static_cast<std::ptrdiff_t>(total_tiles);
            const double flops = static_cast<double>(k) * static_cast<double>(n) * static_cast<double>(n);
#        ifdef _OPENMP
#            pragma omp parallel if (linalg::openmp_max_threads() > 1U && flops >= 262144.0)
#        endif
            {
                alignas(64) std::array<T, k_blas3_mc * k_blas3_nc> c_tile{};
                alignas(64) std::array<T, k_blas3_pack_a_cap<T>> packed_a{};
                alignas(64) std::array<T, k_blas3_pack_b_cap> packed_b{};
#        ifdef _OPENMP
#            pragma omp for schedule(dynamic, 1)
#        endif
                for (std::ptrdiff_t t_s = 0; t_s < total_tiles_s; ++t_s)
                {
                    const auto coords = lower_tile_coords(static_cast<std::size_t>(t_s), n_blocks);
                    const std::size_t bi = coords.row_block;
                    const std::size_t bj = coords.col_block;
                    const std::size_t ib = bi * tile;
                    const std::size_t jb = bj * tile;
                    const std::size_t rm = std::min(tile, n - ib);
                    const std::size_t cn = std::min(tile, n - jb);
                    std::fill_n(c_tile.data(), rm * cn, T{0});

                    for (std::size_t kb = 0U; kb < k; kb += k_blas3_kc)
                    {
                        const std::size_t bs = std::min(k_blas3_kc, k - kb);
                        if (trans_a)
                        {
                            pack_panel_a_transpose(a + (ib * lda) + kb, lda, rm, bs, packed_a.data(), 0U);
                            pack_panel_b_colmajor(a + (jb * lda) + kb, lda, bs, cn, packed_b.data(), 0U);
                        }
                        else
                        {
                            pack_panel_a_colmajor(a + (kb * lda) + ib, lda, rm, bs, packed_a.data());
                            pack_panel_b_colmajor_t(a + (kb * lda) + jb, lda, bs, cn, packed_b.data());
                        }
                        gemm_packed_add(rm, cn, bs, packed_a.data(), packed_b.data(), c_tile.data(), rm);
                    }

                    for (std::size_t col_off = 0U; col_off < cn; ++col_off)
                    {
                        const std::size_t col = jb + col_off;
                        const std::size_t r_start = (bi == bj) ? col_off : 0U;
                        const T* const tile_col = c_tile.data() + (col_off * rm);
                        T* const dst_col = c + (col * ldc);
                        if (beta == T{0})
                        {
                            for (std::size_t r = r_start; r < rm; ++r)
                            {
                                dst_col[ib + r] = alpha * tile_col[r];
                            }
                        }
                        else if (beta == T{1})
                        {
                            for (std::size_t r = r_start; r < rm; ++r)
                            {
                                dst_col[ib + r] = scalar_fmadd(alpha, tile_col[r], dst_col[ib + r]);
                            }
                        }
                        else
                        {
                            for (std::size_t r = r_start; r < rm; ++r)
                            {
                                dst_col[ib + r] = scalar_fmadd(alpha, tile_col[r], beta * dst_col[ib + r]);
                            }
                        }
                    }
                }
            }
            (void)flops;
        }

#    endif  // CXXGPR_HAS_BLAS_BUILTIN

        /// @brief BuiltIn parallel 2-stage SIMD implementation of `blas_compute_w_lower`.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] n Matrix order.
        /// @param[in] l_ptr Square n x n lower Cholesky factor column-major pointer.
        /// @param[in] ldl Leading dimension of L.
        /// @param[in] alpha Solved weight vector A^-1 y of length n.
        /// @param[out] z_ptr Preallocated n x n scratch matrix overwritten with Z = L^-1.
        /// @param[in] ldz Leading dimension of z_scratch.
        /// @param[out] w_ptr Preallocated n x n output matrix receiving lower-triangular W (may alias L).
        /// @param[in] ldw Leading dimension of w_out.
        template <typename T>
        void builtin_compute_w_lower_impl(
            std::size_t n,
            const T* l_ptr,
            std::size_t ldl,
            const T* alpha,
            T* z_ptr,
            std::size_t ldz,
            T* w_ptr,
            std::size_t ldw
        ) noexcept
        {
            const MatrixRef<T> l(l_ptr, n, n, ldl);
            MatrixMut<T> z_scratch(z_ptr, n, n, ldz);
            MatrixMut<T> w_out(w_ptr, n, n, ldw);

            if (n >= 384U)
            {
                const std::size_t col_block = (n >= 768U) ? 24U : 12U;
                const std::size_t n_col_blocks = (n + col_block - 1U) / col_block;
                const auto n_col_blocks_s = static_cast<std::ptrdiff_t>(n_col_blocks);
                const std::size_t tile = (n >= 768U) ? k_blas3_mc : 24U;
                const std::size_t n_tile_blocks = (n + tile - 1U) / tile;
                const std::size_t total_tiles = (n_tile_blocks * (n_tile_blocks + 1U)) / 2U;
                const auto total_tiles_s = static_cast<std::ptrdiff_t>(total_tiles);
#    ifdef _OPENMP
#        pragma omp parallel if (linalg::openmp_max_threads() > 1U)
#    endif
                {
#    ifdef _OPENMP
#        pragma omp for schedule(dynamic, 1)
#    endif
                    for (std::ptrdiff_t b = 0; b < n_col_blocks_s; ++b)
                    {
                        const std::size_t col_begin = static_cast<std::size_t>(b) * col_block;
                        const std::size_t col_end = std::min(col_begin + col_block, n);
                        invert_lower_col_strip_packed(l, z_scratch, col_begin, col_end);
                    }

                    alignas(64) std::array<T, k_blas3_mc * k_blas3_nc> c_tile{};
                    alignas(64) std::array<T, k_blas3_pack_a_cap<T>> packed_a{};
                    alignas(64) std::array<T, k_blas3_pack_b_cap> packed_b{};
#    ifdef _OPENMP
#        pragma omp for schedule(dynamic, 1)
#    endif
                    for (std::ptrdiff_t t_s = 0; t_s < total_tiles_s; ++t_s)
                    {
                        const auto coords = lower_tile_coords(static_cast<std::size_t>(t_s), n_tile_blocks);
                        const std::size_t bi = coords.row_block;
                        const std::size_t bj = coords.col_block;
                        const std::size_t ib = bi * tile;
                        const std::size_t jb = bj * tile;
                        const std::size_t rm = std::min(tile, n - ib);
                        const std::size_t cn = std::min(tile, n - jb);
                        std::fill_n(c_tile.data(), rm * cn, T{0});

                        for (std::size_t kb = ib; kb < n; kb += k_blas3_kc)
                        {
                            const std::size_t bs = std::min(k_blas3_kc, n - kb);
                            pack_panel_a_transpose(
                                &z_scratch(kb, ib), z_scratch.col_stride(), rm, bs, packed_a.data(),
                                (kb == ib) ? rm : 0U
                            );
                            pack_panel_b_colmajor(
                                &z_scratch(kb, jb), z_scratch.col_stride(), bs, cn, packed_b.data(),
                                (kb == jb) ? cn : 0U
                            );
                            gemm_packed_add(rm, cn, bs, packed_a.data(), packed_b.data(), c_tile.data(), rm);
                        }

                        for (std::size_t c = 0U; c < cn; ++c)
                        {
                            const std::size_t col = jb + c;
                            const T alpha_j = alpha[col];
                            const std::size_t r_start = (bi == bj) ? c : 0U;
                            const T* const tile_col = c_tile.data() + (c * rm);
                            T* const w_col = w_out.col(col).data();
                            for (std::size_t r = r_start; r < rm; ++r)
                            {
                                const std::size_t row = ib + r;
                                w_col[row] = scalar_fmadd(alpha[row], alpha_j, -tile_col[r]);
                            }
                        }
                    }
                }
                return;
            }

            auto invert_range = [n, l, z_scratch](std::size_t col_begin, std::size_t col_end) mutable noexcept
            {
                constexpr std::size_t k_block = 32U;
                constexpr std::size_t row_tile = 256U;
                std::array<const T*, k_block> l_ptrs{};
                std::array<T, k_block> v0_arr{};
                std::array<T, k_block> v1_arr{};
                std::array<T, k_block> v2_arr{};
                std::array<T, k_block> v3_arr{};

                std::size_t j = col_begin;
                for (; j + 4U <= col_end; j += 4U)
                {
                    const std::size_t j0 = j;
                    const std::size_t j1 = j + 1U;
                    const std::size_t j2 = j + 2U;
                    const std::size_t j3 = j + 3U;
                    T* const z0 = z_scratch.col(j0).data();
                    T* const z1 = z_scratch.col(j1).data();
                    T* const z2 = z_scratch.col(j2).data();
                    T* const z3 = z_scratch.col(j3).data();

                    for (std::size_t c = 0U; c < 4U; ++c)
                    {
                        const std::size_t jc = j + c;
                        T* const zc = z_scratch.col(jc).data();
                        const T* const l_col_jc = l.col(jc).data();
                        const T z_jj = T{1} / l_col_jc[jc];
                        zc[jc] = z_jj;
                        scale_neg_set_simd(zc, l_col_jc, z_jj, jc + 1U, n);
                    }

                    for (std::size_t c = 0U; c < 3U; ++c)
                    {
                        const std::size_t jc = j + c;
                        T* const zc = z_scratch.col(jc).data();
                        const std::size_t lead_w = j3 - jc;
                        for (std::size_t p = 0U; p < lead_w; ++p)
                        {
                            const std::size_t k = jc + 1U + p;
                            const T* const lk = l.col(k).data();
                            const T zck = zc[k] / lk[k];
                            zc[k] = zck;
                            l_ptrs[p] = lk;
                            v0_arr[p] = zck;
                            for (std::size_t r = k + 1U; r <= j3; ++r)
                            {
                                zc[r] = scalar_fnmadd(lk[r], zck, zc[r]);
                            }
                        }
                        for (std::size_t r_start = j3 + 1U; r_start < n; r_start += row_tile)
                        {
                            const std::size_t r_stop = std::min(r_start + row_tile, n);
                            rank_k_update_1col_simd(zc, l_ptrs.data(), lead_w, v0_arr.data(), r_start, r_stop);
                        }
                    }

                    for (std::size_t kb = j3 + 1U; kb < n; kb += k_block)
                    {
                        const std::size_t ke = std::min(kb + k_block, n);
                        const std::size_t bs = ke - kb;
                        for (std::size_t p = 0U; p < bs; ++p)
                        {
                            const std::size_t k = kb + p;
                            const T* const lk = l.col(k).data();
                            const T lkk = lk[k];
                            const T v0p = z0[k] / lkk;
                            const T v1p = z1[k] / lkk;
                            const T v2p = z2[k] / lkk;
                            const T v3p = z3[k] / lkk;
                            z0[k] = v0p;
                            z1[k] = v1p;
                            z2[k] = v2p;
                            z3[k] = v3p;
                            l_ptrs[p] = lk;
                            v0_arr[p] = v0p;
                            v1_arr[p] = v1p;
                            v2_arr[p] = v2p;
                            v3_arr[p] = v3p;
                            for (std::size_t r = k + 1U; r < ke; ++r)
                            {
                                const T lrk = lk[r];
                                z0[r] = scalar_fnmadd(lrk, v0p, z0[r]);
                                z1[r] = scalar_fnmadd(lrk, v1p, z1[r]);
                                z2[r] = scalar_fnmadd(lrk, v2p, z2[r]);
                                z3[r] = scalar_fnmadd(lrk, v3p, z3[r]);
                            }
                        }
                        if (ke < n)
                        {
                            for (std::size_t r_start = ke; r_start < n; r_start += row_tile)
                            {
                                const std::size_t r_stop = std::min(r_start + row_tile, n);
                                rank_k_update_4cols_simd(
                                    z0, z1, z2, z3, l_ptrs.data(), bs, v0_arr.data(), v1_arr.data(), v2_arr.data(),
                                    v3_arr.data(), r_start, r_stop
                                );
                            }
                        }
                    }
                }

                for (; j < col_end; ++j)
                {
                    T* const z_col = z_scratch.col(j).data();
                    const T* const l_col_j = l.col(j).data();
                    const T z_jj = T{1} / l_col_j[j];
                    z_col[j] = z_jj;
                    scale_neg_set_simd(z_col, l_col_j, z_jj, j + 1U, n);
                    for (std::size_t kb = j + 1U; kb < n; kb += k_block)
                    {
                        const std::size_t ke = std::min(kb + k_block, n);
                        const std::size_t bs = ke - kb;
                        for (std::size_t p = 0U; p < bs; ++p)
                        {
                            const std::size_t k = kb + p;
                            const T* const lk = l.col(k).data();
                            const T z_kj = z_col[k] / lk[k];
                            z_col[k] = z_kj;
                            l_ptrs[p] = lk;
                            v0_arr[p] = z_kj;
                            for (std::size_t r = k + 1U; r < ke; ++r)
                            {
                                z_col[r] = scalar_fnmadd(lk[r], z_kj, z_col[r]);
                            }
                        }
                        if (ke < n)
                        {
                            for (std::size_t r_start = ke; r_start < n; r_start += row_tile)
                            {
                                const std::size_t r_stop = std::min(r_start + row_tile, n);
                                rank_k_update_1col_simd(z_col, l_ptrs.data(), bs, v0_arr.data(), r_start, r_stop);
                            }
                        }
                    }
                }
            };

            if (n >= 96U)
            {
                linalg::parallel_for_lower_blocks(n, 0U, invert_range);
            }
            else
            {
                invert_range(0U, n);
            }

            auto contract_range =
                [n, alpha, z_scratch, w_out](std::size_t col_begin, std::size_t col_end) mutable noexcept
            {
                auto eval_single = [n, &alpha, &z_scratch, &w_out](std::size_t row, std::size_t col_idx) noexcept
                {
                    const T* const z_row = z_scratch.col(row).data();
                    const T* const z_col = z_scratch.col(col_idx).data();
                    const T acc = dot_1x1_simd(z_row + row, z_col + row, n - row);
                    w_out.col(col_idx)[row] = scalar_fmadd(alpha[row], alpha[col_idx], -acc);
                };

                auto finish_4rows = [&alpha](
                                        T* w_col, const T* z0, const T* z1, const T* z2, const T* zj, std::size_t i0,
                                        T alpha_j, T tail0, T tail1, T tail2, T tail3
                                    ) noexcept
                {
                    const std::size_t i1 = i0 + 1U;
                    const std::size_t i2 = i0 + 2U;
                    const std::size_t i3 = i0 + 3U;
                    const T sum0 =
                        scalar_fmadd(z0[i0], zj[i0], scalar_fmadd(z0[i1], zj[i1], scalar_fmadd(z0[i2], zj[i2], tail0)));
                    const T sum1 = scalar_fmadd(z1[i1], zj[i1], scalar_fmadd(z1[i2], zj[i2], tail1));
                    const T sum2 = scalar_fmadd(z2[i2], zj[i2], tail2);
                    w_col[i0] = scalar_fmadd(alpha[i0], alpha_j, -sum0);
                    w_col[i1] = scalar_fmadd(alpha[i1], alpha_j, -sum1);
                    w_col[i2] = scalar_fmadd(alpha[i2], alpha_j, -sum2);
                    w_col[i3] = scalar_fmadd(alpha[i3], alpha_j, -tail3);
                };

                auto eval_4x1 =
                    [n, &alpha, &z_scratch, &w_out, &finish_4rows](std::size_t row, std::size_t col_idx) noexcept
                {
                    const std::size_t i0 = row;
                    const std::size_t i1 = row + 1U;
                    const std::size_t i2 = row + 2U;
                    const std::size_t i3 = row + 3U;
                    const T* const z0 = z_scratch.col(i0).data();
                    const T* const z1 = z_scratch.col(i1).data();
                    const T* const z2 = z_scratch.col(i2).data();
                    const T* const z3 = z_scratch.col(i3).data();
                    const T* const zj = z_scratch.col(col_idx).data();
                    std::array<T, 4> tail_dots{};
                    dot_4x1_simd(z0 + i3, z1 + i3, z2 + i3, z3 + i3, zj + i3, n - i3, tail_dots);
                    finish_4rows(
                        w_out.col(col_idx).data(), z0, z1, z2, zj, i0, alpha[col_idx], tail_dots[0], tail_dots[1],
                        tail_dots[2], tail_dots[3]
                    );
                };

                std::size_t j = col_begin;
                for (; j + 2U <= col_end; j += 2U)
                {
                    const std::size_t j0 = j;
                    const std::size_t j1 = j + 1U;
                    const std::size_t r4_0 = std::min(n, (j0 + 3U) & ~std::size_t{3U});
                    const std::size_t r4_1 = std::min(n, (j1 + 3U) & ~std::size_t{3U});

                    for (std::size_t row = j0; row < r4_0; ++row)
                    {
                        eval_single(row, j0);
                    }
                    std::size_t row_pair_start = r4_1;
                    if (r4_0 < r4_1)
                    {
                        if (r4_0 + 4U <= n)
                        {
                            eval_4x1(r4_0, j0);
                        }
                        else
                        {
                            for (std::size_t row = r4_0; row < n; ++row)
                            {
                                eval_single(row, j0);
                            }
                            row_pair_start = n;
                        }
                    }
                    for (std::size_t row = j1; row < std::min(n, row_pair_start); ++row)
                    {
                        eval_single(row, j1);
                    }

                    const T* const z_j0 = z_scratch.col(j0).data();
                    const T* const z_j1 = z_scratch.col(j1).data();
                    T* const w_col0 = w_out.col(j0).data();
                    T* const w_col1 = w_out.col(j1).data();
                    const T alpha_j0 = alpha[j0];
                    const T alpha_j1 = alpha[j1];

                    std::size_t row = row_pair_start;
                    for (; row + 4U <= n; row += 4U)
                    {
                        const std::size_t i0 = row;
                        const std::size_t i1 = row + 1U;
                        const std::size_t i2 = row + 2U;
                        const std::size_t i3 = row + 3U;
                        const T* const z0 = z_scratch.col(i0).data();
                        const T* const z1 = z_scratch.col(i1).data();
                        const T* const z2 = z_scratch.col(i2).data();
                        const T* const z3 = z_scratch.col(i3).data();

                        std::array<std::array<T, 2>, 4> tail_dots{};
                        dot_4x2_simd(z0 + i3, z1 + i3, z2 + i3, z3 + i3, z_j0 + i3, z_j1 + i3, n - i3, tail_dots);

                        finish_4rows(
                            w_col0, z0, z1, z2, z_j0, i0, alpha_j0, tail_dots[0][0], tail_dots[1][0], tail_dots[2][0],
                            tail_dots[3][0]
                        );
                        finish_4rows(
                            w_col1, z0, z1, z2, z_j1, i0, alpha_j1, tail_dots[0][1], tail_dots[1][1], tail_dots[2][1],
                            tail_dots[3][1]
                        );
                    }
                    for (; row < n; ++row)
                    {
                        eval_single(row, j0);
                        eval_single(row, j1);
                    }
                }

                for (; j < col_end; ++j)
                {
                    const std::size_t r4 = std::min(n, (j + 3U) & ~std::size_t{3U});
                    for (std::size_t row = j; row < r4; ++row)
                    {
                        eval_single(row, j);
                    }
                    std::size_t row = r4;
                    for (; row + 4U <= n; row += 4U)
                    {
                        eval_4x1(row, j);
                    }
                    for (; row < n; ++row)
                    {
                        eval_single(row, j);
                    }
                }
            };
            if (n >= 96U)
            {
                linalg::parallel_for_lower_blocks(n, 0U, contract_range);
            }
            else
            {
                contract_range(0U, n);
            }
        }

#endif  // CXXGPR_HAS_BUILTIN_SIMD_KERNELS

#ifdef CXXGPR_HAS_MKL

        /// @brief Computes lower-triangular W = alpha * alpha^T - A^-1 using MKL LAPACKE potri in 2/3 n^3 FLOPs.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] n Matrix order.
        /// @param[in] l Square lower Cholesky factor column-major pointer.
        /// @param[in] ldl Leading dimension of L.
        /// @param[in] alpha Solved weight vector A^-1 y of length n.
        /// @param[out] w_out Preallocated n x n matrix receiving lower-triangular W (may alias L).
        /// @param[in] ldw Leading dimension of W.
        template <typename T>
        void mkl_compute_w_lower_impl(
            std::size_t n, const T* l, std::size_t ldl, const T* alpha, T* w_out, std::size_t ldw
        ) noexcept
        {
            if (w_out != l)
            {
                for (std::size_t col = 0U; col < n; ++col)
                {
                    const T* const src_col = l + (col * ldl);
                    T* const dst_col = w_out + (col * ldw);
                    std::copy(src_col + col, src_col + n, dst_col + col);
                }
            }
            {
                const MklLocalThreadsGuard mkl_guard(n <= 384U);
                if constexpr (std::same_as<T, double>)
                {
                    (void)LAPACKE_dpotri(
                        LAPACK_COL_MAJOR, 'L', static_cast<lapack_int>(n), w_out, static_cast<lapack_int>(ldw)
                    );
                }
                else
                {
                    (void)LAPACKE_spotri(
                        LAPACK_COL_MAJOR, 'L', static_cast<lapack_int>(n), w_out, static_cast<lapack_int>(ldw)
                    );
                }
            }

            const auto n_s = static_cast<std::ptrdiff_t>(n);
#    ifdef _OPENMP
#        pragma omp parallel for schedule(static) if (n >= 384U)
#    endif
            for (std::ptrdiff_t col_s = 0; col_s < n_s; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                const T alpha_j = alpha[col];
                T* const w_col = w_out + (col * ldw);
                for (std::size_t row = col; row < n; ++row)
                {
                    w_col[row] = (alpha[row] * alpha_j) - w_col[row];
                }
            }
        }

#endif  // CXXGPR_HAS_MKL

#ifdef CXXGPR_HAS_OPENBLAS

        /// @brief Computes lower-triangular W = alpha * alpha^T - A^-1 via blocked trtri + lauum in 2/3 n^3 FLOPs.
        /// @tparam T Scalar floating-point element type (`double` or `float`).
        /// @param[in] n Matrix order.
        /// @param[in] l Square lower Cholesky factor column-major pointer.
        /// @param[in] ldl Leading dimension of L.
        /// @param[in] alpha Solved weight vector A^-1 y of length n.
        /// @param[out] z_scratch Preallocated n x n scratch matrix overwritten with Z = L^-1.
        /// @param[in] ldz Leading dimension of z_scratch.
        /// @param[out] w_out Preallocated n x n matrix receiving lower-triangular W (may alias L).
        /// @param[in] ldw Leading dimension of w_out.
        template <typename T>
        void openblas_compute_w_lower_impl(
            std::size_t n,
            const T* l,
            std::size_t ldl,
            const T* alpha,
            T* z_scratch,
            std::size_t ldz,
            T* w_out,
            std::size_t ldw
        ) noexcept
        {
            constexpr std::size_t k_blk = 64U;
            // Stage 1: Solve Z = L^-1 in block column strips (zeros upper entries inside diagonal blocks, 1/3 n^3).
            for (std::size_t j = 0U; j < n; j += k_blk)
            {
                const std::size_t jb = std::min(k_blk, n - j);
                const std::size_t m = n - j;
                for (std::size_t c = 0U; c < jb; ++c)
                {
                    T* const z_col = z_scratch + ((j + c) * ldz) + j;
                    std::fill_n(z_col, m, T{0});
                    z_col[c] = T{1};
                }
                blas_trsm_lower(
                    false, false, false, m, jb, T{1}, l + (j * ldl) + j, ldl, z_scratch + (j * ldz) + j, ldz
                );
            }

            // Stage 2: Contract lower-triangular W = -Z^T * Z by row blocks (1/3 n^3).
            for (std::size_t i = 0U; i < n; i += k_blk)
            {
                const std::size_t ib = std::min(k_blk, n - i);
                const std::size_t k_len = n - i;
                if (i > 0U)
                {
                    blas_gemm(
                        true, false, ib, i, k_len, T{-1}, z_scratch + (i * ldz) + i, ldz, z_scratch + i, ldz, T{0},
                        w_out + i, ldw
                    );
                }
                blas_syrk_lower(
                    true, ib, k_len, T{-1}, z_scratch + (i * ldz) + i, ldz, T{0}, w_out + (i * ldw) + i, ldw
                );
            }

            // Stage 3: Add symmetric rank-1 alpha * alpha^T into lower triangle of W.
            blas_syr_lower(n, T{1}, alpha, w_out, ldw);
        }

#endif  // CXXGPR_HAS_OPENBLAS

        /// @brief Initializes lower triangle of `cov_ptr` to `kss_ptr + noise_var * I`.
        /// @tparam Storage Input prior matrix scalar type (`double` or `float`).
        /// @tparam Refine Output covariance scalar type (`double` or `float`).
        /// @param[in] m Order of square matrices `kss_ptr` and `cov_ptr`.
        /// @param[in] kss_ptr Column-major `m x m` prior covariance matrix (stride `m`).
        /// @param[in] noise_var Diagonal observation noise variance added when `row == col`.
        /// @param[out] cov_ptr Column-major `m x m` output covariance matrix (stride `ldc`).
        /// @param[in] ldc Column stride of `cov_ptr`.
        template <typename Storage, typename Refine>
        void init_posterior_cov_lower(
            const std::size_t m,
            const Storage* const kss_ptr,
            const Refine noise_var,
            Refine* const cov_ptr,
            const std::size_t ldc
        ) noexcept
        {
            for (std::size_t col = 0U; col < m; ++col)
            {
                const Storage* const kss_col = kss_ptr + (col * m);
                Refine* const dst_col = cov_ptr + (col * ldc);
                for (std::size_t row = col; row < m; ++row)
                {
                    auto entry = static_cast<Refine>(kss_col[row]);
                    if (row == col)
                    {
                        entry += noise_var;
                    }
                    dst_col[row] = entry;
                }
            }
        }

#ifdef CXXGPR_HAS_BUILTIN_SIMD_KERNELS

        /// @brief Computes lower-triangular `kss_ptr + noise_var * I - V^T * V` in one parallel SIMD tile pass.
        /// @tparam Storage Scalar type of `v_ptr` and `kss_ptr` (`double` or `float`).
        /// @tparam Refine Output covariance scalar type (`double` or `float`).
        /// @param[in] n Inner contraction dimension (rows of `n x m` matrix `V`).
        /// @param[in] m Query point count (order of `kss_ptr` and `cov_ptr`).
        /// @param[in] v_ptr Column-major `n x m` matrix `V` (stride `n`).
        /// @param[in] kss_ptr Column-major `m x m` prior covariance matrix (stride `m`).
        /// @param[in] noise_var Diagonal observation noise variance added when `row == col`.
        /// @param[out] cov_ptr Column-major `m x m` output covariance matrix (stride `ldc`).
        /// @param[in] ldc Column stride of `cov_ptr`.
        template <typename Storage, typename Refine>
        void fused_builtin_posterior_cov_lower(
            const std::size_t n,
            const std::size_t m,
            const Storage* const v_ptr,
            const Storage* const kss_ptr,
            const Refine noise_var,
            Refine* const cov_ptr,
            const std::size_t ldc
        ) noexcept
        {
            const std::size_t tile = (m >= 512U) ? k_blas3_mc : 24U;
            const std::size_t n_blocks = (m + tile - 1U) / tile;
            const std::size_t total_tiles = (n_blocks * (n_blocks + 1U)) / 2U;
            const auto total_tiles_s = static_cast<std::ptrdiff_t>(total_tiles);
            const double cov_flops = static_cast<double>(n) * static_cast<double>(m) * static_cast<double>(m);
#    ifdef _OPENMP
#        pragma omp parallel if (linalg::openmp_max_threads() > 1U && cov_flops >= 262144.0)
#    endif
            {
                alignas(64) std::array<Refine, k_blas3_mc * k_blas3_nc> c_tile{};
                alignas(64) std::array<Refine, k_blas3_pack_a_cap<Refine>> packed_a{};
                alignas(64) std::array<Refine, k_blas3_pack_b_cap> packed_b{};
#    ifdef _OPENMP
#        pragma omp for schedule(dynamic, 1)
#    endif
                for (std::ptrdiff_t t_s = 0; t_s < total_tiles_s; ++t_s)
                {
                    const auto coords = lower_tile_coords(static_cast<std::size_t>(t_s), n_blocks);
                    const std::size_t bi = coords.row_block;
                    const std::size_t bj = coords.col_block;
                    const std::size_t ib = bi * tile;
                    const std::size_t jb = bj * tile;
                    const std::size_t rm = std::min(tile, m - ib);
                    const std::size_t cn = std::min(tile, m - jb);
                    std::fill_n(c_tile.data(), rm * cn, Refine{0});

                    for (std::size_t kb = 0U; kb < n; kb += k_blas3_kc)
                    {
                        const std::size_t bs = std::min(k_blas3_kc, n - kb);
                        if constexpr (std::same_as<Storage, Refine>)
                        {
                            pack_panel_a_transpose(v_ptr + (ib * n) + kb, n, rm, bs, packed_a.data(), 0U);
                            pack_panel_b_colmajor(v_ptr + (jb * n) + kb, n, bs, cn, packed_b.data(), 0U);
                        }
                        else
                        {
                            pack_panel_a_transpose_promote_f32_to_f64(
                                v_ptr + (ib * n) + kb, n, rm, bs, packed_a.data()
                            );
                            pack_panel_b_colmajor_promote_f32_to_f64(v_ptr + (jb * n) + kb, n, bs, cn, packed_b.data());
                        }
                        gemm_packed_add(rm, cn, bs, packed_a.data(), packed_b.data(), c_tile.data(), rm);
                    }

                    for (std::size_t c = 0U; c < cn; ++c)
                    {
                        const std::size_t col = jb + c;
                        const std::size_t r_start = (bi == bj) ? c : 0U;
                        const Refine* const tile_col = c_tile.data() + (c * rm);
                        const Storage* const kss_col = kss_ptr + (col * m);
                        Refine* const dst_col = cov_ptr + (col * ldc);
                        for (std::size_t r = r_start; r < rm; ++r)
                        {
                            const std::size_t row = ib + r;
                            Refine entry = static_cast<Refine>(kss_col[row]) - tile_col[r];
                            if (row == col)
                            {
                                entry += noise_var;
                            }
                            dst_col[row] = entry;
                        }
                    }
                }
            }
            (void)cov_flops;
        }

#endif  // CXXGPR_HAS_BUILTIN_SIMD_KERNELS

        /// @brief Same-precision helper forming lower-triangular `kss_ptr + noise_var * I - V^T * V`.
        /// @tparam T Floating-point scalar type (`double` or `float`).
        /// @param[in] n Inner contraction dimension (rows of `n x m` matrix `V`).
        /// @param[in] m Query point count (order of `kss_ptr` and `cov_ptr`).
        /// @param[in] v_ptr Column-major `n x m` matrix `V` (stride `n`).
        /// @param[in] kss_ptr Column-major `m x m` prior covariance matrix (stride `m`).
        /// @param[in] noise_var Diagonal observation noise variance added when `row == col`.
        /// @param[out] cov_ptr Column-major `m x m` output covariance matrix (stride `ldc`).
        /// @param[in] ldc Column stride of `cov_ptr`.
        template <typename T>
        void blas_posterior_cov_lower_same_precision_impl(
            const std::size_t n,
            const std::size_t m,
            const T* const v_ptr,
            const T* const kss_ptr,
            const T noise_var,
            T* const cov_ptr,
            const std::size_t ldc
        ) noexcept
        {
            if (m == 0U)
            {
                return;
            }
            if (n == 0U)
            {
                init_posterior_cov_lower(m, kss_ptr, noise_var, cov_ptr, ldc);
                return;
            }
#ifdef CXXGPR_HAS_BLAS_BUILTIN
            if (m >= 64U && n >= 64U)
            {
                fused_builtin_posterior_cov_lower<T, T>(n, m, v_ptr, kss_ptr, noise_var, cov_ptr, ldc);
                return;
            }
#endif
            init_posterior_cov_lower(m, kss_ptr, noise_var, cov_ptr, ldc);
            blas_syrk_lower(true, m, n, T{-1}, v_ptr, n, T{1}, cov_ptr, ldc);
        }

    }  // namespace

    /// @brief Computes C = alpha * op(A) * op(B) + beta * C for column-major double matrices.
    /// @param[in] trans_a Transpose flag for matrix A.
    /// @param[in] trans_b Transpose flag for matrix B.
    /// @param[in] m Number of rows of op(A) and C.
    /// @param[in] n Number of columns of op(B) and C.
    /// @param[in] k Inner contraction dimension.
    /// @param[in] alpha Scalar multiplier for op(A) * op(B).
    /// @param[in] a Pointer to column-major matrix A.
    /// @param[in] lda Leading dimension of A.
    /// @param[in] b Pointer to column-major matrix B.
    /// @param[in] ldb Leading dimension of B.
    /// @param[in] beta Scalar multiplier for C.
    /// @param[in,out] c Pointer to column-major output matrix C.
    /// @param[in] ldc Leading dimension of C.
    void blas_gemm(
        const bool trans_a,
        const bool trans_b,
        const std::size_t m,
        const std::size_t n,
        const std::size_t k,
        const double alpha,
        const double* const a,
        const std::size_t lda,
        const double* const b,
        const std::size_t ldb,
        const double beta,
        double* const c,
        const std::size_t ldc
    ) noexcept
    {
        if (m == 0U || n == 0U)
        {
            return;
        }
        if (k == 0U)
        {
            ColMap<double> c_mat(
                c, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n),
                Eigen::OuterStride<>(static_cast<Eigen::Index>(ldc))
            );
            if (beta == 0.0)
            {
                c_mat.setZero();
            }
            else if (beta != 1.0)
            {
                c_mat *= beta;
            }
            return;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
#    ifdef CXXGPR_HAS_MKL
        const MklLocalThreadsGuard mkl_guard(m <= 256U && n <= 256U && k <= 256U);
#    endif
        cblas_dgemm(
            CblasColMajor, trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
            static_cast<BlasInt>(m), static_cast<BlasInt>(n), static_cast<BlasInt>(k), alpha, a,
            static_cast<BlasInt>(lda), b, static_cast<BlasInt>(ldb), beta, c, static_cast<BlasInt>(ldc)
        );
#else
        eigen_gemm_impl(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
    }

    /// @brief Single-precision overload of `blas_gemm`.
    /// @param[in] trans_a Transpose flag for matrix A.
    /// @param[in] trans_b Transpose flag for matrix B.
    /// @param[in] m Number of rows of op(A) and C.
    /// @param[in] n Number of columns of op(B) and C.
    /// @param[in] k Inner contraction dimension.
    /// @param[in] alpha Scalar multiplier for op(A) * op(B).
    /// @param[in] a Pointer to column-major matrix A.
    /// @param[in] lda Leading dimension of A.
    /// @param[in] b Pointer to column-major matrix B.
    /// @param[in] ldb Leading dimension of B.
    /// @param[in] beta Scalar multiplier for C.
    /// @param[in,out] c Pointer to column-major output matrix C.
    /// @param[in] ldc Leading dimension of C.
    void blas_gemm(
        const bool trans_a,
        const bool trans_b,
        const std::size_t m,
        const std::size_t n,
        const std::size_t k,
        const float alpha,
        const float* const a,
        const std::size_t lda,
        const float* const b,
        const std::size_t ldb,
        const float beta,
        float* const c,
        const std::size_t ldc
    ) noexcept
    {
        if (m == 0U || n == 0U)
        {
            return;
        }
        if (k == 0U)
        {
            ColMap<float> c_mat(
                c, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n),
                Eigen::OuterStride<>(static_cast<Eigen::Index>(ldc))
            );
            if (beta == 0.0F)
            {
                c_mat.setZero();
            }
            else if (beta != 1.0F)
            {
                c_mat *= beta;
            }
            return;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
#    ifdef CXXGPR_HAS_MKL
        const MklLocalThreadsGuard mkl_guard(m <= 256U && n <= 256U && k <= 256U);
#    endif
        cblas_sgemm(
            CblasColMajor, trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
            static_cast<BlasInt>(m), static_cast<BlasInt>(n), static_cast<BlasInt>(k), alpha, a,
            static_cast<BlasInt>(lda), b, static_cast<BlasInt>(ldb), beta, c, static_cast<BlasInt>(ldc)
        );
#else
        eigen_gemm_impl(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
    }

    /// @brief Updates lower triangle C = alpha * op(A) * op(A)^T + beta * C for column-major double matrices.
    /// @param[in] trans_a If true uses A^T * A, else A * A^T.
    /// @param[in] n Order of symmetric matrix C.
    /// @param[in] k Number of columns (or rows if transposed) summed in the rank-k update.
    /// @param[in] alpha Scalar multiplier for the rank-k term.
    /// @param[in] a Pointer to factor matrix A.
    /// @param[in] lda Leading dimension of A.
    /// @param[in] beta Scalar multiplier for existing entries of C.
    /// @param[in,out] c Pointer to symmetric matrix C updated only in its lower triangle.
    /// @param[in] ldc Leading dimension of C.
    void blas_syrk_lower(
        const bool trans_a,
        const std::size_t n,
        const std::size_t k,
        const double alpha,
        const double* const a,
        const std::size_t lda,
        const double beta,
        double* const c,
        const std::size_t ldc
    ) noexcept
    {
        if (n == 0U)
        {
            return;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
#    ifdef CXXGPR_HAS_MKL
        const MklLocalThreadsGuard mkl_guard(n <= 256U && k <= 256U);
#    endif
        cblas_dsyrk(
            CblasColMajor, CblasLower, trans_a ? CblasTrans : CblasNoTrans, static_cast<BlasInt>(n),
            static_cast<BlasInt>(k), alpha, a, static_cast<BlasInt>(lda), beta, c, static_cast<BlasInt>(ldc)
        );
#else
        builtin_syrk_lower_impl(trans_a, n, k, alpha, a, lda, beta, c, ldc);
#endif
    }

    /// @brief Single-precision overload of `blas_syrk_lower`.
    /// @param[in] trans_a If true uses A^T * A, else A * A^T.
    /// @param[in] n Order of symmetric matrix C.
    /// @param[in] k Number of columns (or rows if transposed) summed in the rank-k update.
    /// @param[in] alpha Scalar multiplier for the rank-k term.
    /// @param[in] a Pointer to factor matrix A.
    /// @param[in] lda Leading dimension of A.
    /// @param[in] beta Scalar multiplier for existing entries of C.
    /// @param[in,out] c Pointer to symmetric matrix C updated only in its lower triangle.
    /// @param[in] ldc Leading dimension of C.
    void blas_syrk_lower(
        const bool trans_a,
        const std::size_t n,
        const std::size_t k,
        const float alpha,
        const float* const a,
        const std::size_t lda,
        const float beta,
        float* const c,
        const std::size_t ldc
    ) noexcept
    {
        if (n == 0U)
        {
            return;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
#    ifdef CXXGPR_HAS_MKL
        const MklLocalThreadsGuard mkl_guard(n <= 256U && k <= 256U);
#    endif
        cblas_ssyrk(
            CblasColMajor, CblasLower, trans_a ? CblasTrans : CblasNoTrans, static_cast<BlasInt>(n),
            static_cast<BlasInt>(k), alpha, a, static_cast<BlasInt>(lda), beta, c, static_cast<BlasInt>(ldc)
        );
#else
        builtin_syrk_lower_impl(trans_a, n, k, alpha, a, lda, beta, c, ldc);
#endif
    }

    /// @brief Applies symmetric rank-1 update C += alpha * x * x^T on the lower triangle of double matrix C.
    /// @param[in] n Order of symmetric matrix C and length of x.
    /// @param[in] alpha Scalar multiplier.
    /// @param[in] x Contiguous input vector of length n.
    /// @param[in,out] c Column-major symmetric matrix C updated in its lower triangle.
    /// @param[in] ldc Leading dimension of C.
    void blas_syr_lower(
        const std::size_t n, const double alpha, const double* const x, double* const c, const std::size_t ldc
    ) noexcept
    {
        if (n == 0U || alpha == 0.0)
        {
            return;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
#    ifdef CXXGPR_HAS_MKL
        const MklLocalThreadsGuard mkl_guard(n <= 512U);
#    endif
        cblas_dsyr(CblasColMajor, CblasLower, static_cast<BlasInt>(n), alpha, x, 1, c, static_cast<BlasInt>(ldc));
#else
        for (std::size_t col = 0U; col < n; ++col)
        {
            const double xc = alpha * x[col];
            double* const c_col = c + (col * ldc);
            for (std::size_t row = col; row < n; ++row)
            {
                c_col[row] += x[row] * xc;
            }
        }
#endif
    }

    /// @brief Single-precision overload of `blas_syr_lower`.
    /// @param[in] n Order of symmetric matrix C and length of x.
    /// @param[in] alpha Scalar multiplier.
    /// @param[in] x Contiguous input vector of length n.
    /// @param[in,out] c Column-major symmetric matrix C updated in its lower triangle.
    /// @param[in] ldc Leading dimension of C.
    void blas_syr_lower(
        const std::size_t n, const float alpha, const float* const x, float* const c, const std::size_t ldc
    ) noexcept
    {
        if (n == 0U || alpha == 0.0F)
        {
            return;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
#    ifdef CXXGPR_HAS_MKL
        const MklLocalThreadsGuard mkl_guard(n <= 512U);
#    endif
        cblas_ssyr(CblasColMajor, CblasLower, static_cast<BlasInt>(n), alpha, x, 1, c, static_cast<BlasInt>(ldc));
#else
        for (std::size_t col = 0U; col < n; ++col)
        {
            const float xc = alpha * x[col];
            float* const c_col = c + (col * ldc);
            for (std::size_t row = col; row < n; ++row)
            {
                c_col[row] += x[row] * xc;
            }
        }
#endif
    }

    /// @brief Computes y = alpha * A * x + beta * y reading only the lower triangle of symmetric double matrix A.
    /// @param[in] n Order of symmetric matrix A and vectors x, y.
    /// @param[in] alpha Scalar multiplier for A * x.
    /// @param[in] a Column-major symmetric matrix A.
    /// @param[in] lda Leading dimension of A.
    /// @param[in] x Contiguous input vector of length n.
    /// @param[in] beta Scalar multiplier for y.
    /// @param[in,out] y Contiguous output vector of length n.
    void blas_symv_lower(
        const std::size_t n,
        const double alpha,
        const double* const a,
        const std::size_t lda,
        const double* const x,
        const double beta,
        double* const y
    ) noexcept
    {
        if (n == 0U)
        {
            return;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
#    ifdef CXXGPR_HAS_MKL
        const MklLocalThreadsGuard mkl_guard(n <= 512U);
#    endif
        cblas_dsymv(
            CblasColMajor, CblasLower, static_cast<BlasInt>(n), alpha, a, static_cast<BlasInt>(lda), x, 1, beta, y, 1
        );
#else
        eigen_symv_lower_impl(n, alpha, a, lda, x, beta, y);
#endif
    }

    /// @brief Single-precision overload of `blas_symv_lower`.
    /// @param[in] n Order of symmetric matrix A and vectors x, y.
    /// @param[in] alpha Scalar multiplier for A * x.
    /// @param[in] a Column-major symmetric matrix A.
    /// @param[in] lda Leading dimension of A.
    /// @param[in] x Contiguous input vector of length n.
    /// @param[in] beta Scalar multiplier for y.
    /// @param[in,out] y Contiguous output vector of length n.
    void blas_symv_lower(
        const std::size_t n,
        const float alpha,
        const float* const a,
        const std::size_t lda,
        const float* const x,
        const float beta,
        float* const y
    ) noexcept
    {
        if (n == 0U)
        {
            return;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
#    ifdef CXXGPR_HAS_MKL
        const MklLocalThreadsGuard mkl_guard(n <= 512U);
#    endif
        cblas_ssymv(
            CblasColMajor, CblasLower, static_cast<BlasInt>(n), alpha, a, static_cast<BlasInt>(lda), x, 1, beta, y, 1
        );
#else
        eigen_symv_lower_impl(n, alpha, a, lda, x, beta, y);
#endif
    }

    /// @brief Solves B <- op(L)^{-1} B or B <- B op(L)^{-T} in place for double lower-triangular L.
    /// @param[in] right_side If true solves X * op(L) = alpha * B; else op(L) * X = alpha * B.
    /// @param[in] trans_a Transpose flag for triangular matrix L.
    /// @param[in] unit_diag True if diagonal of L is assumed to be unit ones.
    /// @param[in] m Number of rows of B.
    /// @param[in] n Number of columns of B.
    /// @param[in] alpha Scalar multiplier applied to B prior to solve.
    /// @param[in] a Pointer to column-major lower-triangular matrix L.
    /// @param[in] lda Leading dimension of L.
    /// @param[in,out] b Column-major matrix B overwritten with the solution.
    /// @param[in] ldb Leading dimension of B.
    void blas_trsm_lower(
        const bool right_side,
        const bool trans_a,
        const bool unit_diag,
        const std::size_t m,
        const std::size_t n,
        const double alpha,
        const double* const a,
        const std::size_t lda,
        double* const b,
        const std::size_t ldb
    ) noexcept
    {
        if (m == 0U || n == 0U)
        {
            return;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
#    ifdef CXXGPR_HAS_MKL
        const std::size_t dim_a = right_side ? n : m;
        const MklLocalThreadsGuard mkl_guard(m <= 256U && n <= 256U && dim_a <= 256U);
#    endif
        cblas_dtrsm(
            CblasColMajor, right_side ? CblasRight : CblasLeft, CblasLower, trans_a ? CblasTrans : CblasNoTrans,
            unit_diag ? CblasUnit : CblasNonUnit, static_cast<BlasInt>(m), static_cast<BlasInt>(n), alpha, a,
            static_cast<BlasInt>(lda), b, static_cast<BlasInt>(ldb)
        );
#else
        builtin_trsm_lower_impl(right_side, trans_a, unit_diag, m, n, alpha, a, lda, b, ldb);
#endif
    }

    /// @brief Single-precision overload of `blas_trsm_lower`.
    /// @param[in] right_side If true solves X * op(L) = alpha * B; else op(L) * X = alpha * B.
    /// @param[in] trans_a Transpose flag for triangular matrix L.
    /// @param[in] unit_diag True if diagonal of L is assumed to be unit ones.
    /// @param[in] m Number of rows of B.
    /// @param[in] n Number of columns of B.
    /// @param[in] alpha Scalar multiplier applied to B prior to solve.
    /// @param[in] a Pointer to column-major lower-triangular matrix L.
    /// @param[in] lda Leading dimension of L.
    /// @param[in,out] b Column-major matrix B overwritten with the solution.
    /// @param[in] ldb Leading dimension of B.
    void blas_trsm_lower(
        const bool right_side,
        const bool trans_a,
        const bool unit_diag,
        const std::size_t m,
        const std::size_t n,
        const float alpha,
        const float* const a,
        const std::size_t lda,
        float* const b,
        const std::size_t ldb
    ) noexcept
    {
        if (m == 0U || n == 0U)
        {
            return;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
#    ifdef CXXGPR_HAS_MKL
        const std::size_t dim_a = right_side ? n : m;
        const MklLocalThreadsGuard mkl_guard(m <= 256U && n <= 256U && dim_a <= 256U);
#    endif
        cblas_strsm(
            CblasColMajor, right_side ? CblasRight : CblasLeft, CblasLower, trans_a ? CblasTrans : CblasNoTrans,
            unit_diag ? CblasUnit : CblasNonUnit, static_cast<BlasInt>(m), static_cast<BlasInt>(n), alpha, a,
            static_cast<BlasInt>(lda), b, static_cast<BlasInt>(ldb)
        );
#else
        builtin_trsm_lower_impl(right_side, trans_a, unit_diag, m, n, alpha, a, lda, b, ldb);
#endif
    }

    /// @brief Solves x <- op(L)^{-1} x in place for double lower-triangular L and contiguous vector x.
    /// @param[in] trans_a Transpose flag for lower-triangular matrix L.
    /// @param[in] unit_diag True if diagonal of L is assumed to be unit ones.
    /// @param[in] n Matrix order and vector length.
    /// @param[in] a Pointer to column-major lower-triangular matrix L.
    /// @param[in] lda Leading dimension of L.
    /// @param[in,out] x Contiguous vector overwritten with the triangular solve result.
    void blas_trsv_lower(
        const bool trans_a,
        const bool unit_diag,
        const std::size_t n,
        const double* const a,
        const std::size_t lda,
        double* const x
    ) noexcept
    {
        if (n == 0U)
        {
            return;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
#    ifdef CXXGPR_HAS_MKL
        const MklLocalThreadsGuard mkl_guard(n <= 512U);
#    endif
        cblas_dtrsv(
            CblasColMajor, CblasLower, trans_a ? CblasTrans : CblasNoTrans, unit_diag ? CblasUnit : CblasNonUnit,
            static_cast<BlasInt>(n), a, static_cast<BlasInt>(lda), x, 1
        );
#else
        builtin_trsm_lower_impl(false, trans_a, unit_diag, n, 1U, 1.0, a, lda, x, n);
#endif
    }

    /// @brief Single-precision overload of `blas_trsv_lower`.
    /// @param[in] trans_a Transpose flag for lower-triangular matrix L.
    /// @param[in] unit_diag True if diagonal of L is assumed to be unit ones.
    /// @param[in] n Matrix order and vector length.
    /// @param[in] a Pointer to column-major lower-triangular matrix L.
    /// @param[in] lda Leading dimension of L.
    /// @param[in,out] x Contiguous vector overwritten with the triangular solve result.
    void blas_trsv_lower(
        const bool trans_a,
        const bool unit_diag,
        const std::size_t n,
        const float* const a,
        const std::size_t lda,
        float* const x
    ) noexcept
    {
        if (n == 0U)
        {
            return;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
#    ifdef CXXGPR_HAS_MKL
        const MklLocalThreadsGuard mkl_guard(n <= 512U);
#    endif
        cblas_strsv(
            CblasColMajor, CblasLower, trans_a ? CblasTrans : CblasNoTrans, unit_diag ? CblasUnit : CblasNonUnit,
            static_cast<BlasInt>(n), a, static_cast<BlasInt>(lda), x, 1
        );
#else
        builtin_trsm_lower_impl(false, trans_a, unit_diag, n, 1U, 1.0F, a, lda, x, n);
#endif
    }

    /// @brief Computes y = alpha * op(A) * x + beta * y for column-major double matrix A.
    /// @param[in] trans_a Transpose flag for matrix A.
    /// @param[in] m Number of rows of matrix A.
    /// @param[in] n Number of columns of matrix A.
    /// @param[in] alpha Scalar multiplier for op(A) * x.
    /// @param[in] a Column-major matrix A.
    /// @param[in] lda Leading dimension of A.
    /// @param[in] x Contiguous input vector.
    /// @param[in] beta Scalar multiplier for y.
    /// @param[in,out] y Contiguous output vector.
    void blas_gemv(
        const bool trans_a,
        const std::size_t m,
        const std::size_t n,
        const double alpha,
        const double* const a,
        const std::size_t lda,
        const double* const x,
        const double beta,
        double* const y
    ) noexcept
    {
        if (m == 0U || n == 0U)
        {
            return;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
#    ifdef CXXGPR_HAS_MKL
        const MklLocalThreadsGuard mkl_guard(
            (m <= 512U && n <= 512U) || (m <= (262144U / std::max<std::size_t>(1U, n)))
        );
#    endif
        cblas_dgemv(
            CblasColMajor, trans_a ? CblasTrans : CblasNoTrans, static_cast<BlasInt>(m), static_cast<BlasInt>(n), alpha,
            a, static_cast<BlasInt>(lda), x, 1, beta, y, 1
        );
#else
        eigen_gemv_impl(trans_a, m, n, alpha, a, lda, x, beta, y);
#endif
    }

    /// @brief Single-precision overload of `blas_gemv`.
    /// @param[in] trans_a Transpose flag for matrix A.
    /// @param[in] m Number of rows of matrix A.
    /// @param[in] n Number of columns of matrix A.
    /// @param[in] alpha Scalar multiplier for op(A) * x.
    /// @param[in] a Column-major matrix A.
    /// @param[in] lda Leading dimension of A.
    /// @param[in] x Contiguous input vector.
    /// @param[in] beta Scalar multiplier for y.
    /// @param[in,out] y Contiguous output vector.
    void blas_gemv(
        const bool trans_a,
        const std::size_t m,
        const std::size_t n,
        const float alpha,
        const float* const a,
        const std::size_t lda,
        const float* const x,
        const float beta,
        float* const y
    ) noexcept
    {
        if (m == 0U || n == 0U)
        {
            return;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
#    ifdef CXXGPR_HAS_MKL
        const MklLocalThreadsGuard mkl_guard(
            (m <= 512U && n <= 512U) || (m <= (262144U / std::max<std::size_t>(1U, n)))
        );
#    endif
        cblas_sgemv(
            CblasColMajor, trans_a ? CblasTrans : CblasNoTrans, static_cast<BlasInt>(m), static_cast<BlasInt>(n), alpha,
            a, static_cast<BlasInt>(lda), x, 1, beta, y, 1
        );
#else
        eigen_gemv_impl(trans_a, m, n, alpha, a, lda, x, beta, y);
#endif
    }

    /// @brief Computes the Euclidean dot product of two double vectors.
    /// @param[in] n Vector length.
    /// @param[in] x First contiguous vector pointer.
    /// @param[in] y Second contiguous vector pointer.
    /// @return Inner product sum_k x[k] * y[k].
    double blas_dot(const std::size_t n, const double* const x, const double* const y) noexcept
    {
        if (n == 0U)
        {
            return 0.0;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
        return cblas_ddot(static_cast<BlasInt>(n), x, 1, y, 1);
#else
        return Eigen::Map<const Eigen::VectorXd>(x, static_cast<Eigen::Index>(n))
            .dot(Eigen::Map<const Eigen::VectorXd>(y, static_cast<Eigen::Index>(n)));
#endif
    }

    /// @brief Single-precision overload of `blas_dot`.
    /// @param[in] n Vector length.
    /// @param[in] x First contiguous vector pointer.
    /// @param[in] y Second contiguous vector pointer.
    /// @return Inner product sum_k x[k] * y[k].
    float blas_dot(const std::size_t n, const float* const x, const float* const y) noexcept
    {
        if (n == 0U)
        {
            return 0.0F;
        }
#if defined(CXXGPR_HAS_MKL) || defined(CXXGPR_HAS_OPENBLAS)
        return cblas_sdot(static_cast<BlasInt>(n), x, 1, y, 1);
#else
        return Eigen::Map<const Eigen::VectorXf>(x, static_cast<Eigen::Index>(n))
            .dot(Eigen::Map<const Eigen::VectorXf>(y, static_cast<Eigen::Index>(n)));
#endif
    }

    /// @brief Computes dense double lower-triangular Cholesky factorization A = L L^T preserving upper triangle.
    /// @param[in] n Order of symmetric matrix A.
    /// @param[in,out] a Column-major symmetric matrix overwritten in lower triangle with L.
    /// @param[in] lda Leading dimension of A.
    /// @return Zero on success, or the 1-based failing pivot index on non-SPD failure.
    std::size_t blas_cholesky_lower(const std::size_t n, double* const a, const std::size_t lda) noexcept
    {
        if (n == 0U)
        {
            return 0U;
        }
#ifdef CXXGPR_HAS_MKL
        if (n <= 256U && linalg::openmp_max_threads() > 1U)
        {
            return builtin_cholesky_lower_impl(n, a, lda);
        }
        const MklLocalThreadsGuard mkl_guard(n < 192U);
        const lapack_int info =
            LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', static_cast<lapack_int>(n), a, static_cast<lapack_int>(lda));
        return info > 0 ? static_cast<std::size_t>(info) : 0U;
#elifdef CXXGPR_HAS_BLAS_BUILTIN
        return builtin_cholesky_lower_impl(n, a, lda);
#else
        return blocked_cholesky_lower_impl(n, a, lda);
#endif
    }

    /// @brief Single-precision overload of `blas_cholesky_lower`.
    /// @param[in] n Order of symmetric matrix A.
    /// @param[in,out] a Column-major symmetric matrix overwritten in lower triangle with L.
    /// @param[in] lda Leading dimension of A.
    /// @return Zero on success, or the 1-based failing pivot index on non-SPD failure.
    std::size_t blas_cholesky_lower(const std::size_t n, float* const a, const std::size_t lda) noexcept
    {
        if (n == 0U)
        {
            return 0U;
        }
#ifdef CXXGPR_HAS_MKL
        const MklLocalThreadsGuard mkl_guard(n < 192U);
        const lapack_int info =
            LAPACKE_spotrf(LAPACK_COL_MAJOR, 'L', static_cast<lapack_int>(n), a, static_cast<lapack_int>(lda));
        return info > 0 ? static_cast<std::size_t>(info) : 0U;
#elifdef CXXGPR_HAS_BLAS_BUILTIN
        return builtin_cholesky_lower_impl(n, a, lda);
#else
        return blocked_cholesky_lower_impl(n, a, lda);
#endif
    }

    /// @brief Forms lower triangle of posterior covariance `kss + noise_var * I - V^T * V` in double precision.
    /// @param[in] n Inner contraction dimension (rows of `n x m` matrix `V`).
    /// @param[in] m Query covariance order (columns of `V` and order of `kss_ptr` and `cov_ptr`).
    /// @param[in] v_ptr Double-precision column-major matrix `V` of size `n x m` (stride `n`).
    /// @param[in] kss_ptr Double-precision column-major prior matrix `K_**` of size `m x m` (stride `m`).
    /// @param[in] noise_var Diagonal observation noise variance added when `row == col`.
    /// @param[out] cov_ptr Double-precision `m x m` matrix receiving lower-triangular posterior covariance.
    /// @param[in] ldc Leading dimension of `cov_ptr`.
    void blas_posterior_cov_lower(
        const std::size_t n,
        const std::size_t m,
        const double* const v_ptr,
        const double* const kss_ptr,
        const double noise_var,
        double* const cov_ptr,
        const std::size_t ldc
    ) noexcept
    {
        blas_posterior_cov_lower_same_precision_impl(n, m, v_ptr, kss_ptr, noise_var, cov_ptr, ldc);
    }

    /// @brief Single-precision overload of `blas_posterior_cov_lower`.
    /// @param[in] n Inner contraction dimension (rows of `n x m` matrix `V`).
    /// @param[in] m Query covariance order (columns of `V` and order of `kss_ptr` and `cov_ptr`).
    /// @param[in] v_ptr Single-precision column-major matrix `V` of size `n x m` (stride `n`).
    /// @param[in] kss_ptr Single-precision column-major prior matrix `K_**` of size `m x m` (stride `m`).
    /// @param[in] noise_var Diagonal observation noise variance added when `row == col`.
    /// @param[out] cov_ptr Single-precision `m x m` matrix receiving lower-triangular posterior covariance.
    /// @param[in] ldc Leading dimension of `cov_ptr`.
    void blas_posterior_cov_lower(
        const std::size_t n,
        const std::size_t m,
        const float* const v_ptr,
        const float* const kss_ptr,
        const float noise_var,
        float* const cov_ptr,
        const std::size_t ldc
    ) noexcept
    {
        blas_posterior_cov_lower_same_precision_impl(n, m, v_ptr, kss_ptr, noise_var, cov_ptr, ldc);
    }

    /// @brief Mixed-precision overload forming double posterior covariance `kss + noise_var * I - V^T * V` from float
    /// `V` and `kss_ptr` using fixed stack tiles.
    /// @param[in] n Inner contraction dimension (rows of `n x m` matrix `V`).
    /// @param[in] m Covariance order (columns of `V` and order of `kss_ptr` and `cov_ptr`).
    /// @param[in] v_ptr Single-precision column-major matrix `V` of size `n x m` (stride `n`).
    /// @param[in] kss_ptr Single-precision column-major prior matrix `K_**` of size `m x m` (stride `m`).
    /// @param[in] noise_var Double-precision diagonal observation noise variance added when `row == col`.
    /// @param[out] cov_ptr Double-precision `m x m` matrix receiving lower-triangular posterior covariance.
    /// @param[in] ldc Leading dimension of `cov_ptr`.
    void blas_posterior_cov_lower(
        const std::size_t n,
        const std::size_t m,
        const float* const v_ptr,
        const float* const kss_ptr,
        const double noise_var,
        double* const cov_ptr,
        const std::size_t ldc
    ) noexcept
    {
        if (m == 0U)
        {
            return;
        }
        if (n == 0U)
        {
            init_posterior_cov_lower<float, double>(m, kss_ptr, noise_var, cov_ptr, ldc);
            return;
        }
#ifdef CXXGPR_HAS_BUILTIN_SIMD_KERNELS
        if (m >= 64U && n >= 64U)
        {
            fused_builtin_posterior_cov_lower<float, double>(n, m, v_ptr, kss_ptr, noise_var, cov_ptr, ldc);
            return;
        }
#endif
        init_posterior_cov_lower<float, double>(m, kss_ptr, noise_var, cov_ptr, ldc);
        // Accumulate lower-triangular tiles of V^T * V in promoted double precision with zero heap allocation.
        constexpr std::size_t k_tile = 64U;
        std::array<double, k_tile * k_tile> a_tile{};
        std::array<double, k_tile * k_tile> b_tile{};
        std::array<double, k_tile * k_tile> c_tile{};
        for (std::size_t ib = 0U; ib < m; ib += k_tile)
        {
            const std::size_t rm = std::min(k_tile, m - ib);
            for (std::size_t jb = 0U; jb <= ib; jb += k_tile)
            {
                const std::size_t cn = std::min(k_tile, m - jb);
                c_tile.fill(0.0);
                for (std::size_t kb = 0U; kb < n; kb += k_tile)
                {
                    const std::size_t bs = std::min(k_tile, n - kb);
                    for (std::size_t c = 0U; c < rm; ++c)
                    {
                        const float* const src = v_ptr + ((ib + c) * n) + kb;
                        double* const dst = a_tile.data() + (c * bs);
                        for (std::size_t r = 0U; r < bs; ++r)
                        {
                            dst[r] = static_cast<double>(src[r]);
                        }
                    }
                    for (std::size_t c = 0U; c < cn; ++c)
                    {
                        const float* const src = v_ptr + ((jb + c) * n) + kb;
                        double* const dst = b_tile.data() + (c * bs);
                        for (std::size_t r = 0U; r < bs; ++r)
                        {
                            dst[r] = static_cast<double>(src[r]);
                        }
                    }
                    blas_gemm(
                        true, false, rm, cn, bs, 1.0, a_tile.data(), bs, b_tile.data(), bs, 1.0, c_tile.data(), rm
                    );
                }
                for (std::size_t c = 0U; c < cn; ++c)
                {
                    const std::size_t col = jb + c;
                    double* const dst_col = cov_ptr + (col * ldc);
                    const std::size_t r_start = (ib == jb) ? c : 0U;
                    for (std::size_t r = r_start; r < rm; ++r)
                    {
                        dst_col[ib + r] -= c_tile[(c * rm) + r];
                    }
                }
            }
        }
    }

    /// @brief Computes lower triangle of W = alpha * alpha^T - (L L^T)^-1 in double precision.
    /// @param[in] n Order of matrices L, Z, and W.
    /// @param[in] l Square n x n lower Cholesky factor column-major pointer.
    /// @param[in] ldl Leading dimension of L.
    /// @param[in] alpha Solved weight vector A^-1 y of length n.
    /// @param[out] z_scratch Preallocated n x n scratch matrix overwritten with Z = L^-1.
    /// @param[in] ldz Leading dimension of z_scratch.
    /// @param[out] w_out Preallocated n x n output matrix receiving lower-triangular W (may alias L).
    /// @param[in] ldw Leading dimension of w_out.
    void blas_compute_w_lower(
        const std::size_t n,
        const double* const l,
        const std::size_t ldl,
        const double* const alpha,
        double* const z_scratch,
        const std::size_t ldz,
        double* const w_out,
        const std::size_t ldw
    ) noexcept
    {
        if (n == 0U)
        {
            return;
        }
#ifdef CXXGPR_HAS_BLAS_BUILTIN
        builtin_compute_w_lower_impl(n, l, ldl, alpha, z_scratch, ldz, w_out, ldw);
#elifdef CXXGPR_HAS_MKL
        if (n <= 384U && linalg::openmp_max_threads() > 1U)
        {
            builtin_compute_w_lower_impl(n, l, ldl, alpha, z_scratch, ldz, w_out, ldw);
            return;
        }
        (void)z_scratch;
        (void)ldz;
        mkl_compute_w_lower_impl(n, l, ldl, alpha, w_out, ldw);
#else
        openblas_compute_w_lower_impl(n, l, ldl, alpha, z_scratch, ldz, w_out, ldw);
#endif
    }

    /// @brief Single-precision overload of `blas_compute_w_lower`.
    /// @param[in] n Order of matrices L, Z, and W.
    /// @param[in] l Square n x n lower Cholesky factor column-major pointer.
    /// @param[in] ldl Leading dimension of L.
    /// @param[in] alpha Solved weight vector A^-1 y of length n.
    /// @param[out] z_scratch Preallocated n x n scratch matrix overwritten with Z = L^-1.
    /// @param[in] ldz Leading dimension of z_scratch.
    /// @param[out] w_out Preallocated n x n output matrix receiving lower-triangular W (may alias L).
    /// @param[in] ldw Leading dimension of w_out.
    void blas_compute_w_lower(
        const std::size_t n,
        const float* const l,
        const std::size_t ldl,
        const float* const alpha,
        float* const z_scratch,
        const std::size_t ldz,
        float* const w_out,
        const std::size_t ldw
    ) noexcept
    {
        if (n == 0U)
        {
            return;
        }
#ifdef CXXGPR_HAS_BLAS_BUILTIN
        builtin_compute_w_lower_impl(n, l, ldl, alpha, z_scratch, ldz, w_out, ldw);
#elifdef CXXGPR_HAS_MKL
        (void)z_scratch;
        (void)ldz;
        mkl_compute_w_lower_impl(n, l, ldl, alpha, w_out, ldw);
#else
        openblas_compute_w_lower_impl(n, l, ldl, alpha, z_scratch, ldz, w_out, ldw);
#endif
    }

    /// @brief Mirrors the lower triangle into the strictly upper triangle of an n x n column-major double matrix.
    /// @param[in] n Matrix order.
    /// @param[in,out] a Column-major square matrix buffer.
    /// @param[in] lda Leading dimension of `a`.
    void symmetrize_lower_to_upper(const std::size_t n, double* const a, const std::size_t lda) noexcept
    {
        for (std::size_t j = 0U; j < n; ++j)
        {
            for (std::size_t i = j + 1U; i < n; ++i)
            {
                a[(i * lda) + j] = a[(j * lda) + i];
            }
        }
    }

    /// @brief Single-precision overload of `symmetrize_lower_to_upper`.
    /// @param[in] n Matrix order.
    /// @param[in,out] a Column-major square matrix buffer.
    /// @param[in] lda Leading dimension of `a`.
    void symmetrize_lower_to_upper(const std::size_t n, float* const a, const std::size_t lda) noexcept
    {
        for (std::size_t j = 0U; j < n; ++j)
        {
            for (std::size_t i = j + 1U; i < n; ++i)
            {
                a[(i * lda) + j] = a[(j * lda) + i];
            }
        }
    }

    /// @brief Reports whether Intel oneMKL is the active compiled BLAS/LAPACK backend.
    /// @return True when MKL is active, false otherwise.
    bool blas_backend_has_mkl() noexcept
    {
#ifdef CXXGPR_HAS_MKL
        return true;
#else
        return false;
#endif
    }

}  // namespace cxxgpr::detail
