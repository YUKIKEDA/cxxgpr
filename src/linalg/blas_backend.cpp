// SPDX-License-Identifier: Apache-2.0
/// @file blas_backend.cpp
/// @brief Implementation of internal CBLAS/LAPACKE dense linear algebra wrappers.

#include "linalg/blas_backend.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#ifdef CXXGPR_HAS_MKL
#    include <mkl.h>
#    include <mkl_cblas.h>
#    include <mkl_lapacke.h>
#elifdef CXXGPR_HAS_OPENBLAS
#    if __has_include(<cblas.h>)
#        include <cblas.h>
#    elif __has_include(<openblas/cblas.h>)
#        include <openblas/cblas.h>
#    else
#        error "OpenBLAS selected but <cblas.h> could not be located."
#    endif
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
                // Scale and update sub-diagonal column entries in place without touching upper triangle.
                for (std::size_t i = j + 1U; i < n; ++i)
                {
                    T sum = col_j[i];
                    for (std::size_t k = 0U; k < j; ++k)
                    {
                        sum -= a[(k * lda) + i] * a[(k * lda) + j];
                    }
                    col_j[i] = sum * inv_l_jj;
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

#ifdef CXXGPR_HAS_MKL
        /// @brief Vendor BLAS integer type matching Intel oneMKL LP64 CBLAS signatures.
        using BlasInt = MKL_INT;
#elifdef CXXGPR_HAS_OPENBLAS
        /// @brief Vendor BLAS integer type matching OpenBLAS CBLAS signatures.
        using BlasInt = blasint;
#endif

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
        cblas_dgemm(
            CblasColMajor, trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
            static_cast<BlasInt>(m), static_cast<BlasInt>(n), static_cast<BlasInt>(k), alpha, a,
            static_cast<BlasInt>(lda), b, static_cast<BlasInt>(ldb), beta, c, static_cast<BlasInt>(ldc)
        );
#else
        const auto rows_a = static_cast<Eigen::Index>(trans_a ? k : m);
        const auto cols_a = static_cast<Eigen::Index>(trans_a ? m : k);
        const auto rows_b = static_cast<Eigen::Index>(trans_b ? n : k);
        const auto cols_b = static_cast<Eigen::Index>(trans_b ? k : n);
        ConstColMap<double> a_mat(a, rows_a, cols_a, Eigen::OuterStride<>(static_cast<Eigen::Index>(lda)));
        ConstColMap<double> b_mat(b, rows_b, cols_b, Eigen::OuterStride<>(static_cast<Eigen::Index>(ldb)));
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
        cblas_sgemm(
            CblasColMajor, trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
            static_cast<BlasInt>(m), static_cast<BlasInt>(n), static_cast<BlasInt>(k), alpha, a,
            static_cast<BlasInt>(lda), b, static_cast<BlasInt>(ldb), beta, c, static_cast<BlasInt>(ldc)
        );
#else
        const auto rows_a = static_cast<Eigen::Index>(trans_a ? k : m);
        const auto cols_a = static_cast<Eigen::Index>(trans_a ? m : k);
        const auto rows_b = static_cast<Eigen::Index>(trans_b ? n : k);
        const auto cols_b = static_cast<Eigen::Index>(trans_b ? k : n);
        ConstColMap<float> a_mat(a, rows_a, cols_a, Eigen::OuterStride<>(static_cast<Eigen::Index>(lda)));
        ConstColMap<float> b_mat(b, rows_b, cols_b, Eigen::OuterStride<>(static_cast<Eigen::Index>(ldb)));
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
        cblas_dsyrk(
            CblasColMajor, CblasLower, trans_a ? CblasTrans : CblasNoTrans, static_cast<BlasInt>(n),
            static_cast<BlasInt>(k), alpha, a, static_cast<BlasInt>(lda), beta, c, static_cast<BlasInt>(ldc)
        );
#else
        ColMap<double> c_mat(
            c, static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n),
            Eigen::OuterStride<>(static_cast<Eigen::Index>(ldc))
        );
        if (beta == 0.0)
        {
            c_mat.triangularView<Eigen::Lower>().setZero();
        }
        else if (beta != 1.0)
        {
            for (std::size_t col = 0U; col < n; ++col)
            {
                for (std::size_t row = col; row < n; ++row)
                {
                    c_mat(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) *= beta;
                }
            }
        }
        if (k == 0U || alpha == 0.0)
        {
            return;
        }
        const auto rows_a = static_cast<Eigen::Index>(trans_a ? k : n);
        const auto cols_a = static_cast<Eigen::Index>(trans_a ? n : k);
        ConstColMap<double> a_mat(a, rows_a, cols_a, Eigen::OuterStride<>(static_cast<Eigen::Index>(lda)));
        if (trans_a)
        {
            c_mat.triangularView<Eigen::Lower>() += alpha * (a_mat.transpose() * a_mat);
        }
        else
        {
            c_mat.triangularView<Eigen::Lower>() += alpha * (a_mat * a_mat.transpose());
        }
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
        cblas_ssyrk(
            CblasColMajor, CblasLower, trans_a ? CblasTrans : CblasNoTrans, static_cast<BlasInt>(n),
            static_cast<BlasInt>(k), alpha, a, static_cast<BlasInt>(lda), beta, c, static_cast<BlasInt>(ldc)
        );
#else
        ColMap<float> c_mat(
            c, static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n),
            Eigen::OuterStride<>(static_cast<Eigen::Index>(ldc))
        );
        if (beta == 0.0F)
        {
            c_mat.triangularView<Eigen::Lower>().setZero();
        }
        else if (beta != 1.0F)
        {
            for (std::size_t col = 0U; col < n; ++col)
            {
                for (std::size_t row = col; row < n; ++row)
                {
                    c_mat(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) *= beta;
                }
            }
        }
        if (k == 0U || alpha == 0.0F)
        {
            return;
        }
        const auto rows_a = static_cast<Eigen::Index>(trans_a ? k : n);
        const auto cols_a = static_cast<Eigen::Index>(trans_a ? n : k);
        ConstColMap<float> a_mat(a, rows_a, cols_a, Eigen::OuterStride<>(static_cast<Eigen::Index>(lda)));
        if (trans_a)
        {
            c_mat.triangularView<Eigen::Lower>() += alpha * (a_mat.transpose() * a_mat);
        }
        else
        {
            c_mat.triangularView<Eigen::Lower>() += alpha * (a_mat * a_mat.transpose());
        }
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
        cblas_dsymv(
            CblasColMajor, CblasLower, static_cast<BlasInt>(n), alpha, a, static_cast<BlasInt>(lda), x, 1, beta, y, 1
        );
#else
        ConstColMap<double> a_mat(
            a, static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n),
            Eigen::OuterStride<>(static_cast<Eigen::Index>(lda))
        );
        const Eigen::Map<const Eigen::VectorXd> x_vec(x, static_cast<Eigen::Index>(n));
        Eigen::Map<Eigen::VectorXd> y_vec(y, static_cast<Eigen::Index>(n));
        if (beta == 0.0)
        {
            y_vec.noalias() = alpha * (a_mat.template selfadjointView<Eigen::Lower>() * x_vec);
        }
        else
        {
            y_vec = (beta * y_vec) + alpha * (a_mat.template selfadjointView<Eigen::Lower>() * x_vec);
        }
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
        cblas_ssymv(
            CblasColMajor, CblasLower, static_cast<BlasInt>(n), alpha, a, static_cast<BlasInt>(lda), x, 1, beta, y, 1
        );
#else
        ConstColMap<float> a_mat(
            a, static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n),
            Eigen::OuterStride<>(static_cast<Eigen::Index>(lda))
        );
        const Eigen::Map<const Eigen::VectorXf> x_vec(x, static_cast<Eigen::Index>(n));
        Eigen::Map<Eigen::VectorXf> y_vec(y, static_cast<Eigen::Index>(n));
        if (beta == 0.0F)
        {
            y_vec.noalias() = alpha * (a_mat.template selfadjointView<Eigen::Lower>() * x_vec);
        }
        else
        {
            y_vec = (beta * y_vec) + alpha * (a_mat.template selfadjointView<Eigen::Lower>() * x_vec);
        }
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
        cblas_dtrsm(
            CblasColMajor, right_side ? CblasRight : CblasLeft, CblasLower, trans_a ? CblasTrans : CblasNoTrans,
            unit_diag ? CblasUnit : CblasNonUnit, static_cast<BlasInt>(m), static_cast<BlasInt>(n), alpha, a,
            static_cast<BlasInt>(lda), b, static_cast<BlasInt>(ldb)
        );
#else
        const std::size_t dim_a = right_side ? n : m;
        ConstColMap<double> a_mat(
            a, static_cast<Eigen::Index>(dim_a), static_cast<Eigen::Index>(dim_a),
            Eigen::OuterStride<>(static_cast<Eigen::Index>(lda))
        );
        ColMap<double> b_mat(
            b, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n),
            Eigen::OuterStride<>(static_cast<Eigen::Index>(ldb))
        );
        if (alpha != 1.0)
        {
            b_mat *= alpha;
        }
        if (unit_diag)
        {
            if (right_side)
            {
                if (trans_a)
                {
                    a_mat.triangularView<Eigen::UnitLower>().transpose().template solveInPlace<Eigen::OnTheRight>(
                        b_mat
                    );
                }
                else
                {
                    a_mat.triangularView<Eigen::UnitLower>().template solveInPlace<Eigen::OnTheRight>(b_mat);
                }
            }
            else if (trans_a)
            {
                a_mat.triangularView<Eigen::UnitLower>().transpose().solveInPlace(b_mat);
            }
            else
            {
                a_mat.triangularView<Eigen::UnitLower>().solveInPlace(b_mat);
            }
        }
        else
        {
            if (right_side)
            {
                if (trans_a)
                {
                    a_mat.triangularView<Eigen::Lower>().transpose().template solveInPlace<Eigen::OnTheRight>(b_mat);
                }
                else
                {
                    a_mat.triangularView<Eigen::Lower>().template solveInPlace<Eigen::OnTheRight>(b_mat);
                }
            }
            else if (trans_a)
            {
                a_mat.triangularView<Eigen::Lower>().transpose().solveInPlace(b_mat);
            }
            else
            {
                a_mat.triangularView<Eigen::Lower>().solveInPlace(b_mat);
            }
        }
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
        cblas_strsm(
            CblasColMajor, right_side ? CblasRight : CblasLeft, CblasLower, trans_a ? CblasTrans : CblasNoTrans,
            unit_diag ? CblasUnit : CblasNonUnit, static_cast<BlasInt>(m), static_cast<BlasInt>(n), alpha, a,
            static_cast<BlasInt>(lda), b, static_cast<BlasInt>(ldb)
        );
#else
        const std::size_t dim_a = right_side ? n : m;
        ConstColMap<float> a_mat(
            a, static_cast<Eigen::Index>(dim_a), static_cast<Eigen::Index>(dim_a),
            Eigen::OuterStride<>(static_cast<Eigen::Index>(lda))
        );
        ColMap<float> b_mat(
            b, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n),
            Eigen::OuterStride<>(static_cast<Eigen::Index>(ldb))
        );
        if (alpha != 1.0F)
        {
            b_mat *= alpha;
        }
        if (unit_diag)
        {
            if (right_side)
            {
                if (trans_a)
                {
                    a_mat.triangularView<Eigen::UnitLower>().transpose().template solveInPlace<Eigen::OnTheRight>(
                        b_mat
                    );
                }
                else
                {
                    a_mat.triangularView<Eigen::UnitLower>().template solveInPlace<Eigen::OnTheRight>(b_mat);
                }
            }
            else if (trans_a)
            {
                a_mat.triangularView<Eigen::UnitLower>().transpose().solveInPlace(b_mat);
            }
            else
            {
                a_mat.triangularView<Eigen::UnitLower>().solveInPlace(b_mat);
            }
        }
        else
        {
            if (right_side)
            {
                if (trans_a)
                {
                    a_mat.triangularView<Eigen::Lower>().transpose().template solveInPlace<Eigen::OnTheRight>(b_mat);
                }
                else
                {
                    a_mat.triangularView<Eigen::Lower>().template solveInPlace<Eigen::OnTheRight>(b_mat);
                }
            }
            else if (trans_a)
            {
                a_mat.triangularView<Eigen::Lower>().transpose().solveInPlace(b_mat);
            }
            else
            {
                a_mat.triangularView<Eigen::Lower>().solveInPlace(b_mat);
            }
        }
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
        cblas_dtrsv(
            CblasColMajor, CblasLower, trans_a ? CblasTrans : CblasNoTrans, unit_diag ? CblasUnit : CblasNonUnit,
            static_cast<BlasInt>(n), a, static_cast<BlasInt>(lda), x, 1
        );
#else
        blas_trsm_lower(false, trans_a, unit_diag, n, 1U, 1.0, a, lda, x, n);
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
        cblas_strsv(
            CblasColMajor, CblasLower, trans_a ? CblasTrans : CblasNoTrans, unit_diag ? CblasUnit : CblasNonUnit,
            static_cast<BlasInt>(n), a, static_cast<BlasInt>(lda), x, 1
        );
#else
        blas_trsm_lower(false, trans_a, unit_diag, n, 1U, 1.0F, a, lda, x, n);
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
        cblas_dgemv(
            CblasColMajor, trans_a ? CblasTrans : CblasNoTrans, static_cast<BlasInt>(m), static_cast<BlasInt>(n), alpha,
            a, static_cast<BlasInt>(lda), x, 1, beta, y, 1
        );
#else
        ConstColMap<double> a_mat(
            a, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n),
            Eigen::OuterStride<>(static_cast<Eigen::Index>(lda))
        );
        const Eigen::Index len_x = static_cast<Eigen::Index>(trans_a ? m : n);
        const Eigen::Index len_y = static_cast<Eigen::Index>(trans_a ? n : m);
        const Eigen::Map<const Eigen::VectorXd> x_vec(x, len_x);
        Eigen::Map<Eigen::VectorXd> y_vec(y, len_y);
        if (beta == 0.0)
        {
            y_vec.setZero();
        }
        else if (beta != 1.0)
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
        cblas_sgemv(
            CblasColMajor, trans_a ? CblasTrans : CblasNoTrans, static_cast<BlasInt>(m), static_cast<BlasInt>(n), alpha,
            a, static_cast<BlasInt>(lda), x, 1, beta, y, 1
        );
#else
        ConstColMap<float> a_mat(
            a, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n),
            Eigen::OuterStride<>(static_cast<Eigen::Index>(lda))
        );
        const Eigen::Index len_x = static_cast<Eigen::Index>(trans_a ? m : n);
        const Eigen::Index len_y = static_cast<Eigen::Index>(trans_a ? n : m);
        const Eigen::Map<const Eigen::VectorXf> x_vec(x, len_x);
        Eigen::Map<Eigen::VectorXf> y_vec(y, len_y);
        if (beta == 0.0F)
        {
            y_vec.setZero();
        }
        else if (beta != 1.0F)
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
        const lapack_int info =
            LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', static_cast<lapack_int>(n), a, static_cast<lapack_int>(lda));
        return info > 0 ? static_cast<std::size_t>(info) : 0U;
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
        const lapack_int info =
            LAPACKE_spotrf(LAPACK_COL_MAJOR, 'L', static_cast<lapack_int>(n), a, static_cast<lapack_int>(lda));
        return info > 0 ? static_cast<std::size_t>(info) : 0U;
#else
        return blocked_cholesky_lower_impl(n, a, lda);
#endif
    }

    /// @brief Subtracts promoted V^T * V from lower triangle of double matrix C using fixed stack tiles.
    /// @param[in] n Inner contraction dimension (rows of n x m matrix V).
    /// @param[in] m Covariance order (columns of V and order of C).
    /// @param[in] v_ptr Single-precision column-major matrix V of size n x m (stride n).
    /// @param[in,out] cov_ptr Double-precision m x m matrix updated in lower triangle: C -= V^T * V.
    /// @param[in] ldc Leading dimension of `cov_ptr`.
    void subtract_promoted_gram_lower(
        const std::size_t n, const std::size_t m, const float* const v_ptr, double* const cov_ptr, const std::size_t ldc
    ) noexcept
    {
        if (m == 0U || n == 0U)
        {
            return;
        }
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
