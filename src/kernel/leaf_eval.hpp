// SPDX-License-Identifier: Apache-2.0
/// @file leaf_eval.hpp
/// @brief Internal kernel leaf evaluation, gradient, Hessian, and packed ARD dispatch declarations.
/// @details Declares low-level leaf operations shared by KernelSpec, CompiledKernel, and
/// CompiledDistanceKernel across squared-distance, coordinate, and packed lower-triangular ARD inputs.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/kernel/exp.hpp"
#include "cxxgpr/kernel/spec.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <cstddef>
#include <expected>
#include <span>

/// @brief Internal implementation helpers for kernel leaf evaluation and parameter dispatch.
namespace cxxgpr::detail
{

    /// @brief Returns the coordinate representation mode required by a single kernel leaf.
    /// @details Maps isotropic leaves and CustomKernel to Dist, ARD and Linear leaves to Points,
    /// and ConstantKernel/WhiteKernel to Either.
    /// @param leaf Kernel leaf variant to inspect.
    /// @return Required CoordMode for leaf.
    [[nodiscard]] CoordMode leaf_coord_mode(const KernelLeaf& leaf) noexcept;

    /// @brief Returns the number of log-space hyperparameters owned by a single kernel leaf.
    /// @details Delegates to the active variant member's num_params().
    /// @param leaf Kernel leaf variant to inspect.
    /// @return Number of log-space hyperparameters.
    [[nodiscard]] std::size_t leaf_num_params(const KernelLeaf& leaf) noexcept;

    /// @brief Writes a kernel leaf's unconstrained log-space hyperparameters into out.
    /// @details Delegates to the active variant member's get_params(out).
    /// @param leaf Kernel leaf variant to query.
    /// @param out Destination span sized to leaf_num_params(leaf).
    /// @return Empty expected on success or GprError on span length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != leaf_num_params(leaf).
    [[nodiscard]] std::expected<void, GprError> leaf_get_params(const KernelLeaf& leaf, std::span<double> out);

    /// @brief Atomically updates a kernel leaf's parameters from unconstrained log-space values.
    /// @details Delegates to the active variant member's set_params(params).
    /// @param leaf Kernel leaf variant to update.
    /// @param params Input span of log-space parameters sized to leaf_num_params(leaf).
    /// @return Empty expected on success or GprError on count mismatch, non-finite value, or bound error.
    /// @retval GprErrorCode::LengthMismatch When params.size() != leaf_num_params(leaf).
    /// @retval GprErrorCode::NonFiniteInput When any parameter is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When any exponentiated value exits its leaf interval.
    [[nodiscard]] std::expected<void, GprError> leaf_set_params(KernelLeaf& leaf, std::span<const double> params);

    /// @brief Writes a kernel leaf's open parameter intervals in user units into out.
    /// @details Delegates to the active variant member's bounds_into(out).
    /// @param leaf Kernel leaf variant to query.
    /// @param out Destination span sized to leaf_num_params(leaf).
    /// @return Empty expected on success or GprError on span length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != leaf_num_params(leaf).
    [[nodiscard]] std::expected<void, GprError> leaf_bounds_into(const KernelLeaf& leaf, std::span<Interval> out);

    /// @brief Evaluates a leaf training Gram matrix from precomputed squared distances.
    /// @details Supports isotropic distance leaves, ConstantKernel, WhiteKernel, and CustomKernel;
    /// rejects coordinate-only leaves (ARD and Linear).
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param leaf Kernel leaf variant to evaluate.
    /// @param dist Column-major n x n squared-distance matrix.
    /// @param out Column-major n x n destination Gram matrix.
    /// @param uplo Matrix triangle or full region to write.
    /// @param exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @return Empty expected on success or GprError on invalid inputs or unsupported leaf type.
    /// @retval GprErrorCode::EmptyInput When dist is empty.
    /// @retval GprErrorCode::LengthMismatch When dist is non-square or out shape differs from dist.
    /// @retval GprErrorCode::NonFiniteInput When any active entry of dist is non-finite.
    /// @retval GprErrorCode::UnsupportedKernelOperation When leaf requires point coordinates.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_apply_dist(
        const KernelLeaf& leaf,
        MatrixRef<T> dist,
        MatrixMut<T> out,
        Triangle uplo,
        KernelExp exp_mode = KernelExp::Accurate
    );

    /// @brief Evaluates a leaf training Gram matrix directly from input point coordinates x.
    /// @details Supports all built-in leaves (isotropic, ARD, Linear, Constant, White); rejects CustomKernel.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param leaf Kernel leaf variant to evaluate.
    /// @param x Column-major n x d training coordinate matrix.
    /// @param out Column-major n x n destination Gram matrix.
    /// @param uplo Matrix triangle or full region to write.
    /// @param exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @param cached_ls Pre-cached ARD lengthscales in double, or empty to read from leaf.
    /// @return Empty expected on success or GprError on invalid inputs, mismatched dims, or CustomKernel.
    /// @retval GprErrorCode::EmptyInput When x is empty.
    /// @retval GprErrorCode::LengthMismatch When out shape is not n x n.
    /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD leaf's dimensions.
    /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
    /// @retval GprErrorCode::UnsupportedKernelOperation When leaf is CustomKernel.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_apply_points(
        const KernelLeaf& leaf,
        MatrixRef<T> x,
        MatrixMut<T> out,
        Triangle uplo,
        KernelExp exp_mode = KernelExp::Accurate,
        std::span<const double> cached_ls = {}
    );

    /// @brief Evaluates a rectangular cross-covariance matrix for a leaf from squared distances.
    /// @details Supports isotropic distance leaves, ConstantKernel, WhiteKernel (writes 0), and CustomKernel.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param leaf Kernel leaf variant to evaluate.
    /// @param dist Column-major n1 x n2 squared-distance matrix.
    /// @param out Column-major n1 x n2 destination cross-covariance matrix.
    /// @param exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @return Empty expected on success or GprError on invalid inputs or coordinate-only leaf.
    /// @retval GprErrorCode::EmptyInput When dist is empty.
    /// @retval GprErrorCode::LengthMismatch When out shape differs from dist.
    /// @retval GprErrorCode::NonFiniteInput When any entry of dist is non-finite.
    /// @retval GprErrorCode::UnsupportedKernelOperation When leaf requires point coordinates.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_apply_cross_dist(
        const KernelLeaf& leaf, MatrixRef<T> dist, MatrixMut<T> out, KernelExp exp_mode = KernelExp::Accurate
    );

    /// @brief Evaluates a rectangular cross-covariance matrix for a leaf from coordinate matrices x1, x2.
    /// @details Computes k(x1_i, x2_j) across all pairs for any built-in leaf; rejects CustomKernel.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param leaf Kernel leaf variant to evaluate.
    /// @param x1 First point matrix of shape n1 x d in column-major order.
    /// @param x2 Second point matrix of shape n2 x d in column-major order.
    /// @param out Column-major n1 x n2 destination cross-covariance matrix.
    /// @param exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @param cached_ls Pre-cached ARD lengthscales in double, or empty to read from leaf.
    /// @return Empty expected on success or GprError on invalid inputs, shape/dim mismatch, or CustomKernel.
    /// @retval GprErrorCode::EmptyInput When x1 or x2 is empty.
    /// @retval GprErrorCode::LengthMismatch When out shape is not n1 x n2.
    /// @retval GprErrorCode::DimensionMismatch When x1.cols() != x2.cols() or disagrees with ARD dims.
    /// @retval GprErrorCode::NonFiniteInput When any entry of x1 or x2 is non-finite.
    /// @retval GprErrorCode::UnsupportedKernelOperation When leaf is CustomKernel.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_apply_cross_points(
        const KernelLeaf& leaf,
        MatrixRef<T> x1,
        MatrixRef<T> x2,
        MatrixMut<T> out,
        KernelExp exp_mode = KernelExp::Accurate,
        std::span<const double> cached_ls = {}
    );

    /// @brief Fills the training diagonal k(x_i, x_i) for a single kernel leaf over points x.
    /// @details Evaluates stationary leaves to 1, ConstantKernel to constant(), WhiteKernel to variance(),
    /// LinearKernel to variance * ||x_i||_2^2, and delegates CustomKernel to KernelTerm::fill_diag().
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param leaf Kernel leaf variant to evaluate.
    /// @param x Column-major n x d input coordinate matrix.
    /// @param out Destination span of length n receiving diagonal entries.
    /// @param cached_ls Pre-cached ARD lengthscales in double, or empty to read from leaf.
    /// @return Empty expected on success or GprError on invalid input, length mismatch, or ARD mismatch.
    /// @retval GprErrorCode::EmptyInput When x is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != x.rows().
    /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD leaf's dimensions.
    /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_fill_diag(
        const KernelLeaf& leaf, MatrixRef<T> x, std::span<T> out, std::span<const double> cached_ls = {}
    );

    /// @brief Evaluates the first derivative dK / d(log theta_{param_idx}) from squared distances.
    /// @details Supports isotropic distance leaves, ConstantKernel, WhiteKernel, and CustomKernel;
    /// rejects coordinate-only leaves (ARD and Linear).
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param leaf Kernel leaf variant to differentiate.
    /// @param dist Column-major n x n squared-distance matrix.
    /// @param d_k Column-major n x n destination derivative matrix.
    /// @param param_idx Leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param uplo Matrix triangle or full region to write.
    /// @param exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @return Empty expected on success or GprError on invalid inputs, out-of-range index, or unsupported leaf.
    /// @retval GprErrorCode::EmptyInput When dist is empty.
    /// @retval GprErrorCode::LengthMismatch When dist is non-square or d_k shape differs from dist.
    /// @retval GprErrorCode::NonFiniteInput When any active entry of dist is non-finite.
    /// @retval GprErrorCode::IndexOutOfRange When param_idx >= leaf_num_params(leaf).
    /// @retval GprErrorCode::UnsupportedKernelOperation When leaf requires point coordinates.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_grad_dist(
        const KernelLeaf& leaf,
        MatrixRef<T> dist,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        KernelExp exp_mode = KernelExp::Accurate
    );

    /// @brief Evaluates the first derivative dK / d(log theta_{param_idx}) from point coordinates x.
    /// @details Supports all built-in leaves (including ARD and Linear); rejects CustomKernel.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param leaf Kernel leaf variant to differentiate.
    /// @param x Column-major n x d training coordinate matrix.
    /// @param d_k Column-major n x n destination derivative matrix.
    /// @param param_idx Leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param uplo Matrix triangle or full region to write.
    /// @param exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @param cached_ls Pre-cached ARD lengthscales in double, or empty to read from leaf.
    /// @return Empty expected on success or GprError on validation, index, or CustomKernel error.
    /// @retval GprErrorCode::EmptyInput When x is empty.
    /// @retval GprErrorCode::LengthMismatch When d_k shape is not n x n.
    /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD leaf's dimensions.
    /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
    /// @retval GprErrorCode::IndexOutOfRange When param_idx >= leaf_num_params(leaf).
    /// @retval GprErrorCode::UnsupportedKernelOperation When leaf is CustomKernel.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_grad_points(
        const KernelLeaf& leaf,
        MatrixRef<T> x,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        KernelExp exp_mode = KernelExp::Accurate,
        std::span<const double> cached_ls = {}
    );

    /// @brief Evaluates the second derivative d^2 K / (d(log theta_i) d(log theta_j)) from squared distances.
    /// @details Supports isotropic distance leaves, ConstantKernel, WhiteKernel, and CustomKernel;
    /// rejects coordinate-only leaves (ARD and Linear).
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param leaf Kernel leaf variant to differentiate.
    /// @param dist Column-major n x n squared-distance matrix.
    /// @param d2_k Column-major n x n destination second-derivative matrix.
    /// @param i First leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param j Second leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param uplo Matrix triangle or full region to write.
    /// @param exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @return Empty expected on success or GprError on invalid inputs, out-of-range indices, or unsupported leaf.
    /// @retval GprErrorCode::EmptyInput When dist is empty.
    /// @retval GprErrorCode::LengthMismatch When dist is non-square or d2_k shape differs from dist.
    /// @retval GprErrorCode::NonFiniteInput When any active entry of dist is non-finite.
    /// @retval GprErrorCode::IndexOutOfRange When i or j >= leaf_num_params(leaf).
    /// @retval GprErrorCode::UnsupportedKernelOperation When leaf requires point coordinates.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_hess_dist(
        const KernelLeaf& leaf,
        MatrixRef<T> dist,
        MatrixMut<T> d2_k,
        std::size_t i,
        std::size_t j,
        Triangle uplo,
        KernelExp exp_mode = KernelExp::Accurate
    );

    /// @brief Evaluates the second derivative d^2 K / (d(log theta_i) d(log theta_j)) from point coordinates.
    /// @details Supports all built-in leaves (including ARD and Linear); rejects CustomKernel.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param leaf Kernel leaf variant to differentiate.
    /// @param x Column-major n x d training coordinate matrix.
    /// @param d2_k Column-major n x n destination second-derivative matrix.
    /// @param i First leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param j Second leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param uplo Matrix triangle or full region to write.
    /// @param exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @param cached_ls Pre-cached ARD lengthscales in double, or empty to read from leaf.
    /// @return Empty expected on success or GprError on validation, index, or CustomKernel error.
    /// @retval GprErrorCode::EmptyInput When x is empty.
    /// @retval GprErrorCode::LengthMismatch When d2_k shape is not n x n.
    /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD leaf's dimensions.
    /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
    /// @retval GprErrorCode::IndexOutOfRange When i or j >= leaf_num_params(leaf).
    /// @retval GprErrorCode::UnsupportedKernelOperation When leaf is CustomKernel.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_hess_points(
        const KernelLeaf& leaf,
        MatrixRef<T> x,
        MatrixMut<T> d2_k,
        std::size_t i,
        std::size_t j,
        Triangle uplo,
        KernelExp exp_mode = KernelExp::Accurate,
        std::span<const double> cached_ls = {}
    );

    /// @brief Evaluates an ARD kernel leaf from packed lower-triangular per-dimension squared differences.
    /// @details Reads dimension-major lower-triangular blocks of size n * (n + 1) / 2 from ard_packed
    /// for RbfArdKernel, MaternArdKernel, and RationalQuadraticArdKernel, mirroring entries when needed.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param leaf ARD kernel leaf variant to evaluate.
    /// @param ls Lengthscale vector in user units of length d.
    /// @param ard_packed Packed lower-triangular (dx_d)^2 array of length d * n * (n + 1) / 2.
    /// @param n Number of training points.
    /// @param out Column-major n x n destination Gram matrix.
    /// @param uplo Matrix triangle or full region to write.
    /// @param exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @return Empty expected on success or GprError on shape mismatch or non-ARD leaf.
    /// @retval GprErrorCode::EmptyInput When n == 0 or ls is empty.
    /// @retval GprErrorCode::LengthMismatch When out shape is not n x n or ard_packed size disagrees.
    /// @retval GprErrorCode::UnsupportedKernelOperation When leaf does not consume packed ARD distances.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_apply_ard_packed(
        const KernelLeaf& leaf,
        std::span<const double> ls,
        std::span<const T> ard_packed,
        std::size_t n,
        MatrixMut<T> out,
        Triangle uplo,
        KernelExp exp_mode = KernelExp::Accurate
    );

    /// @brief Evaluates the first derivative of an ARD leaf from packed lower-triangular ARD differences.
    /// @details Supports RbfArdKernel, MaternArdKernel, and RationalQuadraticArdKernel (including alpha).
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param leaf ARD kernel leaf variant to differentiate.
    /// @param ls Lengthscale vector in user units of length d.
    /// @param ard_packed Packed lower-triangular (dx_d)^2 array of length d * n * (n + 1) / 2.
    /// @param n Number of training points.
    /// @param d_k Column-major n x n destination derivative matrix.
    /// @param param_idx Leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param uplo Matrix triangle or full region to write.
    /// @param exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @return Empty expected on success or GprError on shape mismatch, bad index, or non-ARD leaf.
    /// @retval GprErrorCode::EmptyInput When n == 0 or ls is empty.
    /// @retval GprErrorCode::LengthMismatch When d_k shape is not n x n or ard_packed size disagrees.
    /// @retval GprErrorCode::IndexOutOfRange When param_idx >= leaf_num_params(leaf).
    /// @retval GprErrorCode::UnsupportedKernelOperation When leaf does not consume packed ARD distances.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_grad_ard_packed(
        const KernelLeaf& leaf,
        std::span<const double> ls,
        std::span<const T> ard_packed,
        std::size_t n,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        KernelExp exp_mode = KernelExp::Accurate
    );

    /// @brief Evaluates the second derivative of an ARD leaf from packed lower-triangular ARD differences.
    /// @details Supports RbfArdKernel, MaternArdKernel, and RationalQuadraticArdKernel across both
    /// lengthscale-lengthscale and alpha-coupled second partials.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param leaf ARD kernel leaf variant to differentiate.
    /// @param ls Lengthscale vector in user units of length d.
    /// @param ard_packed Packed lower-triangular (dx_d)^2 array of length d * n * (n + 1) / 2.
    /// @param n Number of training points.
    /// @param d2_k Column-major n x n destination second-derivative matrix.
    /// @param i First leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param j Second leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param uplo Matrix triangle or full region to write.
    /// @param exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @return Empty expected on success or GprError on shape mismatch, bad index, or non-ARD leaf.
    /// @retval GprErrorCode::EmptyInput When n == 0 or ls is empty.
    /// @retval GprErrorCode::LengthMismatch When d2_k shape is not n x n or ard_packed size disagrees.
    /// @retval GprErrorCode::IndexOutOfRange When i or j >= leaf_num_params(leaf).
    /// @retval GprErrorCode::UnsupportedKernelOperation When leaf does not consume packed ARD distances.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_hess_ard_packed(
        const KernelLeaf& leaf,
        std::span<const double> ls,
        std::span<const T> ard_packed,
        std::size_t n,
        MatrixMut<T> d2_k,
        std::size_t i,
        std::size_t j,
        Triangle uplo,
        KernelExp exp_mode = KernelExp::Accurate
    );

}  // namespace cxxgpr::detail
