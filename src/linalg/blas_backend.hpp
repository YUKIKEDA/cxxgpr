// SPDX-License-Identifier: Apache-2.0
/// @file blas_backend.hpp
/// @brief Internal CBLAS/LAPACKE facade isolating third-party BLAS/MKL headers from the codebase.
/// @details Declares zero-allocation dense linear algebra wrappers implemented against OpenBLAS,
/// Intel oneMKL, or BuiltIn SIMD+OpenMP kernels in blas_backend.cpp, plus a deterministic promoted dot helper.

#pragma once

#include <concepts>
#include <cstddef>

/// @brief Internal CBLAS/LAPACKE implementation details isolated from public headers.
namespace cxxgpr::detail
{

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
        bool trans_a,
        bool trans_b,
        std::size_t m,
        std::size_t n,
        std::size_t k,
        double alpha,
        const double* a,
        std::size_t lda,
        const double* b,
        std::size_t ldb,
        double beta,
        double* c,
        std::size_t ldc
    ) noexcept;

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
        bool trans_a,
        bool trans_b,
        std::size_t m,
        std::size_t n,
        std::size_t k,
        float alpha,
        const float* a,
        std::size_t lda,
        const float* b,
        std::size_t ldb,
        float beta,
        float* c,
        std::size_t ldc
    ) noexcept;

    /// @brief Updates lower triangle C = alpha * op(A) * op(A)^T + beta * C for column-major symmetric matrices.
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
        bool trans_a,
        std::size_t n,
        std::size_t k,
        double alpha,
        const double* a,
        std::size_t lda,
        double beta,
        double* c,
        std::size_t ldc
    ) noexcept;

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
        bool trans_a,
        std::size_t n,
        std::size_t k,
        float alpha,
        const float* a,
        std::size_t lda,
        float beta,
        float* c,
        std::size_t ldc
    ) noexcept;

    /// @brief Applies symmetric rank-1 update C += alpha * x * x^T on the lower triangle of C.
    /// @param[in] n Order of symmetric matrix C and length of x.
    /// @param[in] alpha Scalar multiplier.
    /// @param[in] x Contiguous input vector of length n.
    /// @param[in,out] c Column-major symmetric matrix C updated in its lower triangle.
    /// @param[in] ldc Leading dimension of C.
    void blas_syr_lower(std::size_t n, double alpha, const double* x, double* c, std::size_t ldc) noexcept;

    /// @brief Single-precision overload of `blas_syr_lower`.
    /// @param[in] n Order of symmetric matrix C and length of x.
    /// @param[in] alpha Scalar multiplier.
    /// @param[in] x Contiguous input vector of length n.
    /// @param[in,out] c Column-major symmetric matrix C updated in its lower triangle.
    /// @param[in] ldc Leading dimension of C.
    void blas_syr_lower(std::size_t n, float alpha, const float* x, float* c, std::size_t ldc) noexcept;

    /// @brief Computes y = alpha * A * x + beta * y reading only the lower triangle of symmetric matrix A.
    /// @param[in] n Order of symmetric matrix A and vectors x, y.
    /// @param[in] alpha Scalar multiplier for A * x.
    /// @param[in] a Column-major symmetric matrix A.
    /// @param[in] lda Leading dimension of A.
    /// @param[in] x Contiguous input vector of length n.
    /// @param[in] beta Scalar multiplier for y.
    /// @param[in,out] y Contiguous output vector of length n.
    void blas_symv_lower(
        std::size_t n, double alpha, const double* a, std::size_t lda, const double* x, double beta, double* y
    ) noexcept;

    /// @brief Single-precision overload of `blas_symv_lower`.
    /// @param[in] n Order of symmetric matrix A and vectors x, y.
    /// @param[in] alpha Scalar multiplier for A * x.
    /// @param[in] a Column-major symmetric matrix A.
    /// @param[in] lda Leading dimension of A.
    /// @param[in] x Contiguous input vector of length n.
    /// @param[in] beta Scalar multiplier for y.
    /// @param[in,out] y Contiguous output vector of length n.
    void blas_symv_lower(
        std::size_t n, float alpha, const float* a, std::size_t lda, const float* x, float beta, float* y
    ) noexcept;

    /// @brief Solves B <- op(L)^{-1} B or B <- B op(L)^{-T} in place for lower-triangular L.
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
        bool right_side,
        bool trans_a,
        bool unit_diag,
        std::size_t m,
        std::size_t n,
        double alpha,
        const double* a,
        std::size_t lda,
        double* b,
        std::size_t ldb
    ) noexcept;

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
        bool right_side,
        bool trans_a,
        bool unit_diag,
        std::size_t m,
        std::size_t n,
        float alpha,
        const float* a,
        std::size_t lda,
        float* b,
        std::size_t ldb
    ) noexcept;

    /// @brief Solves x <- op(L)^{-1} x in place for lower-triangular L and contiguous vector x.
    /// @param[in] trans_a Transpose flag for lower-triangular matrix L.
    /// @param[in] unit_diag True if diagonal of L is assumed to be unit ones.
    /// @param[in] n Matrix order and vector length.
    /// @param[in] a Pointer to column-major lower-triangular matrix L.
    /// @param[in] lda Leading dimension of L.
    /// @param[in,out] x Contiguous vector overwritten with the triangular solve result.
    void blas_trsv_lower(
        bool trans_a, bool unit_diag, std::size_t n, const double* a, std::size_t lda, double* x
    ) noexcept;

    /// @brief Single-precision overload of `blas_trsv_lower`.
    /// @param[in] trans_a Transpose flag for lower-triangular matrix L.
    /// @param[in] unit_diag True if diagonal of L is assumed to be unit ones.
    /// @param[in] n Matrix order and vector length.
    /// @param[in] a Pointer to column-major lower-triangular matrix L.
    /// @param[in] lda Leading dimension of L.
    /// @param[in,out] x Contiguous vector overwritten with the triangular solve result.
    void blas_trsv_lower(
        bool trans_a, bool unit_diag, std::size_t n, const float* a, std::size_t lda, float* x
    ) noexcept;

    /// @brief Computes y = alpha * op(A) * x + beta * y for a column-major matrix A.
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
        bool trans_a,
        std::size_t m,
        std::size_t n,
        double alpha,
        const double* a,
        std::size_t lda,
        const double* x,
        double beta,
        double* y
    ) noexcept;

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
        bool trans_a,
        std::size_t m,
        std::size_t n,
        float alpha,
        const float* a,
        std::size_t lda,
        const float* x,
        float beta,
        float* y
    ) noexcept;

    /// @brief Computes the Euclidean dot product of two double vectors.
    /// @param[in] n Vector length.
    /// @param[in] x First contiguous vector pointer.
    /// @param[in] y Second contiguous vector pointer.
    /// @return Inner product sum_k x[k] * y[k].
    [[nodiscard]] double blas_dot(std::size_t n, const double* x, const double* y) noexcept;

    /// @brief Single-precision overload of `blas_dot`.
    /// @param[in] n Vector length.
    /// @param[in] x First contiguous vector pointer.
    /// @param[in] y Second contiguous vector pointer.
    /// @return Inner product sum_k x[k] * y[k].
    [[nodiscard]] float blas_dot(std::size_t n, const float* x, const float* y) noexcept;

    /// @brief Computes dense lower-triangular Cholesky factorization A = L L^T preserving upper triangle.
    /// @param[in] n Order of symmetric matrix A.
    /// @param[in,out] a Column-major symmetric matrix overwritten in lower triangle with L.
    /// @param[in] lda Leading dimension of A.
    /// @return Zero on success, the 1-based failing pivot index on non-SPD failure, or
    /// std::numeric_limits<std::size_t>::max() on backend argument/size error.
    [[nodiscard]] std::size_t blas_cholesky_lower(std::size_t n, double* a, std::size_t lda) noexcept;

    /// @brief Single-precision overload of `blas_cholesky_lower`.
    /// @param[in] n Order of symmetric matrix A.
    /// @param[in,out] a Column-major symmetric matrix overwritten in lower triangle with L.
    /// @param[in] lda Leading dimension of A.
    /// @return Zero on success, the 1-based failing pivot index on non-SPD failure, or
    /// std::numeric_limits<std::size_t>::max() on backend argument/size error.
    [[nodiscard]] std::size_t blas_cholesky_lower(std::size_t n, float* a, std::size_t lda) noexcept;

    /// @brief Forms lower triangle of posterior covariance `kss + noise_var * I - V^T * V` in double precision.
    /// @details Executes a fused parallel tile pass on `BuiltIn` (`m >= 64 && n >= 64`) or initializes `cov_ptr` from
    /// `kss_ptr` plus diagonal `noise_var` followed by `blas_syrk_lower`.
    /// @param[in] n Training point count (number of rows of `n x m` matrix `V`).
    /// @param[in] m Query point count (order of square matrices `kss_ptr` and `cov_ptr`).
    /// @param[in] v_ptr Double-precision column-major matrix `V` of size `n x m` (stride `n`).
    /// @param[in] kss_ptr Double-precision column-major prior matrix `K_**` of size `m x m` (stride `m`).
    /// @param[in] noise_var Non-negative diagonal observation noise variance added when `row == col`.
    /// @param[out] cov_ptr Double-precision `m x m` output matrix receiving lower-triangular posterior covariance.
    /// @param[in] ldc Column stride of `cov_ptr`.
    void blas_posterior_cov_lower(
        std::size_t n,
        std::size_t m,
        const double* v_ptr,
        const double* kss_ptr,
        double noise_var,
        double* cov_ptr,
        std::size_t ldc
    ) noexcept;

    /// @brief Single-precision overload of `blas_posterior_cov_lower`.
    /// @param[in] n Training point count (number of rows of `n x m` matrix `V`).
    /// @param[in] m Query point count (order of square matrices `kss_ptr` and `cov_ptr`).
    /// @param[in] v_ptr Single-precision column-major matrix `V` of size `n x m` (stride `n`).
    /// @param[in] kss_ptr Single-precision column-major prior matrix `K_**` of size `m x m` (stride `m`).
    /// @param[in] noise_var Non-negative diagonal observation noise variance added when `row == col`.
    /// @param[out] cov_ptr Single-precision `m x m` output matrix receiving lower-triangular posterior covariance.
    /// @param[in] ldc Column stride of `cov_ptr`.
    void blas_posterior_cov_lower(
        std::size_t n,
        std::size_t m,
        const float* v_ptr,
        const float* kss_ptr,
        float noise_var,
        float* cov_ptr,
        std::size_t ldc
    ) noexcept;

    /// @brief Mixed-precision overload forming double posterior covariance `kss + noise_var * I - V^T * V` from float
    /// `V` and `kss_ptr` without heap allocation.
    /// @param[in] n Training point count (number of rows of `n x m` matrix `V`).
    /// @param[in] m Query point count (order of square matrices `kss_ptr` and `cov_ptr`).
    /// @param[in] v_ptr Single-precision column-major matrix `V` of size `n x m` (stride `n`).
    /// @param[in] kss_ptr Single-precision column-major prior matrix `K_**` of size `m x m` (stride `m`).
    /// @param[in] noise_var Non-negative double-precision diagonal observation noise variance added when `row == col`.
    /// @param[out] cov_ptr Double-precision `m x m` output matrix receiving lower-triangular posterior covariance.
    /// @param[in] ldc Column stride of `cov_ptr`.
    void blas_posterior_cov_lower(
        std::size_t n,
        std::size_t m,
        const float* v_ptr,
        const float* kss_ptr,
        double noise_var,
        double* cov_ptr,
        std::size_t ldc
    ) noexcept;

    /// @brief Computes lower triangle of W = alpha * alpha^T - (L L^T)^-1 from lower Cholesky factor L.
    /// @details Uses preallocated n x n scratch matrix z_scratch for Z = L^-1; w_out may safely alias l.
    /// @param[in] n Order of matrices L, Z, and W.
    /// @param[in] l Square n x n lower Cholesky factor column-major pointer.
    /// @param[in] ldl Leading dimension of L.
    /// @param[in] alpha Solved weight vector A^-1 y of length n.
    /// @param[out] z_scratch Preallocated n x n scratch matrix overwritten with Z = L^-1.
    /// @param[in] ldz Leading dimension of z_scratch.
    /// @param[out] w_out Preallocated n x n matrix receiving lower-triangular W (may alias l).
    /// @param[in] ldw Leading dimension of w_out.
    /// @return True on success, or false if the backend inversion fails or extents overflow.
    [[nodiscard]] bool blas_compute_w_lower(
        std::size_t n,
        const double* l,
        std::size_t ldl,
        const double* alpha,
        double* z_scratch,
        std::size_t ldz,
        double* w_out,
        std::size_t ldw
    ) noexcept;

    /// @brief Single-precision overload of `blas_compute_w_lower`.
    /// @param[in] n Order of matrices L, Z, and W.
    /// @param[in] l Square n x n lower Cholesky factor column-major pointer.
    /// @param[in] ldl Leading dimension of L.
    /// @param[in] alpha Solved weight vector A^-1 y of length n.
    /// @param[out] z_scratch Preallocated n x n scratch matrix overwritten with Z = L^-1.
    /// @param[in] ldz Leading dimension of z_scratch.
    /// @param[out] w_out Preallocated n x n matrix receiving lower-triangular W (may alias l).
    /// @param[in] ldw Leading dimension of w_out.
    /// @return True on success, or false if the backend inversion fails or extents overflow.
    [[nodiscard]] bool blas_compute_w_lower(
        std::size_t n,
        const float* l,
        std::size_t ldl,
        const float* alpha,
        float* z_scratch,
        std::size_t ldz,
        float* w_out,
        std::size_t ldw
    ) noexcept;

    /// @brief Accumulates the inner product sum_k static_cast<Accum>(a[k]) * static_cast<Accum>(b[k]).
    /// @details Safe inside OpenMP regions and bitwise deterministic across thread counts via fixed index order.
    /// @tparam Accum Accumulator scalar type.
    /// @tparam A First input element type.
    /// @tparam B Second input element type.
    /// @param[in] a Pointer to first contiguous input array.
    /// @param[in] b Pointer to second contiguous input array.
    /// @param[in] len Number of elements in both arrays.
    /// @return Accumulated inner product in `Accum`.
    template <typename Accum, typename A, typename B>
    [[nodiscard]] inline Accum dot_accum(const A* a, const B* b, std::size_t len) noexcept
    {
        Accum sum0{0};
        Accum sum1{0};
        Accum sum2{0};
        Accum sum3{0};
        std::size_t k = 0U;
        for (; k + 4U <= len; k += 4U)
        {
            sum0 += static_cast<Accum>(a[k]) * static_cast<Accum>(b[k]);
            sum1 += static_cast<Accum>(a[k + 1U]) * static_cast<Accum>(b[k + 1U]);
            sum2 += static_cast<Accum>(a[k + 2U]) * static_cast<Accum>(b[k + 2U]);
            sum3 += static_cast<Accum>(a[k + 3U]) * static_cast<Accum>(b[k + 3U]);
        }
        Accum tail{0};
        for (; k < len; ++k)
        {
            tail += static_cast<Accum>(a[k]) * static_cast<Accum>(b[k]);
        }
        return (sum0 + sum1) + (sum2 + sum3) + tail;
    }

    /// @brief Accumulates inner product in double precision for matching scalar elements.
    /// @tparam T Storage element type.
    /// @param[in] a Pointer to first contiguous input array.
    /// @param[in] b Pointer to second contiguous input array.
    /// @param[in] len Number of elements in both arrays.
    /// @return Double-precision inner product.
    template <typename T> [[nodiscard]] inline double dot_promoted_f64(const T* a, const T* b, std::size_t len) noexcept
    {
        return dot_accum<double>(a, b, len);
    }

    /// @brief Mirrors the lower triangle into the strictly upper triangle of an n x n column-major double matrix.
    /// @param[in] n Order of square matrix A.
    /// @param[in,out] a Pointer to column-major matrix A.
    /// @param[in] lda Leading dimension of A.
    void symmetrize_lower_to_upper(std::size_t n, double* a, std::size_t lda) noexcept;

    /// @brief Single-precision overload of `symmetrize_lower_to_upper`.
    /// @param[in] n Order of square matrix A.
    /// @param[in,out] a Pointer to column-major matrix A.
    /// @param[in] lda Leading dimension of A.
    void symmetrize_lower_to_upper(std::size_t n, float* a, std::size_t lda) noexcept;

    /// @brief Returns true when Intel oneMKL is active (for LAPACKE availability queries).
    /// @return True when MKL is compiled and active, false otherwise.
    [[nodiscard]] bool blas_backend_has_mkl() noexcept;

}  // namespace cxxgpr::detail
