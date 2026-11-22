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
#include <vector>

/// @brief Internal implementation helpers for kernel leaf evaluation and parameter dispatch.
namespace cxxgpr::detail
{

    /// @brief Populates per-leaf cached ARD lengthscale vectors in place without reallocating existing capacities.
    /// @details Writes each ARD leaf's lengthscales into cached_ls[idx] and clears non-ARD entries while retaining
    /// vector capacity.
    /// @param[in] leaves Compiled kernel leaf vector.
    /// @param[in,out] cached_ls Per-leaf lengthscale snapshot vectors resized to leaves.size().
    void refresh_cached_ls_entries(std::span<const KernelLeaf> leaves, std::vector<std::vector<double>>& cached_ls);

    /// @brief Atomically updates all leaf hyperparameters in leaves from log-space params without hot-path allocations
    /// for built-in leaves.
    /// @details When leaves contains only built-in leaf types, validates every parameter against its leaf bounds before
    /// updating leaves in place so neither validation nor update allocates. When a CustomKernel is present, stages a
    /// copy of leaves so any custom set_params() error rolls back all leaves atomically.
    /// @param[in,out] leaves Compiled kernel leaf vector to update.
    /// @param[in] params Log-space parameter span of length expected_params.
    /// @param[in] expected_params Expected flattened parameter count across leaves.
    /// @return Empty expected on success, or GprError leaving leaves unchanged.
    /// @retval GprErrorCode::LengthMismatch When params.size() != expected_params.
    /// @retval GprErrorCode::NonFiniteInput When any parameter is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When any exponentiated value exits its leaf interval.
    [[nodiscard]] std::expected<void, GprError> set_leaves_params_atomic(
        std::vector<KernelLeaf>& leaves, std::span<const double> params, std::size_t expected_params
    );

    /// @brief Returns the coordinate representation mode required by a single kernel leaf.
    /// @details Maps isotropic leaves and CustomKernel to Dist, ARD and Linear leaves to Points,
    /// and ConstantKernel/WhiteKernel to Either.
    /// @param[in] leaf Kernel leaf variant to inspect.
    /// @return Required CoordMode for leaf.
    [[nodiscard]] CoordMode leaf_coord_mode(const KernelLeaf& leaf) noexcept;

    /// @brief Returns the number of log-space hyperparameters owned by a single kernel leaf.
    /// @details Delegates to the active variant member's num_params().
    /// @param[in] leaf Kernel leaf variant to inspect.
    /// @return Number of log-space hyperparameters.
    [[nodiscard]] std::size_t leaf_num_params(const KernelLeaf& leaf) noexcept;

    /// @brief Writes a kernel leaf's unconstrained log-space hyperparameters into out.
    /// @details Delegates to the active variant member's get_params(out).
    /// @param[in] leaf Kernel leaf variant to query.
    /// @param[out] out Destination span sized to leaf_num_params(leaf).
    /// @return Empty expected on success or GprError on span length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != leaf_num_params(leaf).
    [[nodiscard]] std::expected<void, GprError> leaf_get_params(const KernelLeaf& leaf, std::span<double> out);

    /// @brief Atomically updates a kernel leaf's parameters from unconstrained log-space values.
    /// @details Delegates to the active variant member's set_params(params).
    /// @param[in,out] leaf Kernel leaf variant to update.
    /// @param[in] params Input span of log-space parameters sized to leaf_num_params(leaf).
    /// @return Empty expected on success or GprError on count mismatch, non-finite value, or bound error.
    /// @retval GprErrorCode::LengthMismatch When params.size() != leaf_num_params(leaf).
    /// @retval GprErrorCode::NonFiniteInput When any parameter is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When any exponentiated value exits its leaf interval.
    [[nodiscard]] std::expected<void, GprError> leaf_set_params(KernelLeaf& leaf, std::span<const double> params);

    /// @brief Writes a kernel leaf's open parameter intervals in user units into out.
    /// @details Delegates to the active variant member's bounds_into(out).
    /// @param[in] leaf Kernel leaf variant to query.
    /// @param[out] out Destination span sized to leaf_num_params(leaf).
    /// @return Empty expected on success or GprError on span length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != leaf_num_params(leaf).
    [[nodiscard]] std::expected<void, GprError> leaf_bounds_into(const KernelLeaf& leaf, std::span<Interval> out);

    /// @brief Evaluates a leaf training Gram matrix from precomputed squared distances.
    /// @details Supports isotropic distance leaves, ConstantKernel, WhiteKernel, and CustomKernel;
    /// rejects coordinate-only leaves (ARD and Linear).
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param[in] leaf Kernel leaf variant to evaluate.
    /// @param[in] dist Column-major n x n squared-distance matrix.
    /// @param[out] out Column-major n x n destination Gram matrix.
    /// @param[in] uplo Matrix triangle or full region to write.
    /// @param[in] exp_mode Exponential evaluation backend (Accurate or FastApprox).
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
    /// @param[in] leaf Kernel leaf variant to evaluate.
    /// @param[in] x Column-major n x d training coordinate matrix.
    /// @param[out] out Column-major n x n destination Gram matrix.
    /// @param[in] uplo Matrix triangle or full region to write.
    /// @param[in] exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @param[in] cached_ls Pre-cached ARD lengthscales in double, or empty to read from leaf.
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
    /// @param[in] leaf Kernel leaf variant to evaluate.
    /// @param[in] dist Column-major n1 x n2 squared-distance matrix.
    /// @param[out] out Column-major n1 x n2 destination cross-covariance matrix.
    /// @param[in] exp_mode Exponential evaluation backend (Accurate or FastApprox).
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
    /// @param[in] leaf Kernel leaf variant to evaluate.
    /// @param[in] x1 First point matrix of shape n1 x d in column-major order.
    /// @param[in] x2 Second point matrix of shape n2 x d in column-major order.
    /// @param[out] out Column-major n1 x n2 destination cross-covariance matrix.
    /// @param[in] exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @param[in] cached_ls Pre-cached ARD lengthscales in double, or empty to read from leaf.
    /// @param[in] coordinates_validated True when the caller already checked all coordinates for finiteness.
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
        std::span<const double> cached_ls = {},
        bool coordinates_validated = false
    );

    /// @brief Fills the training diagonal k(x_i, x_i) for a single kernel leaf over points x.
    /// @details Evaluates stationary leaves to 1, ConstantKernel to constant(), WhiteKernel to variance(),
    /// LinearKernel to variance * ||x_i||_2^2, and delegates CustomKernel to KernelTerm::fill_diag().
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param[in] leaf Kernel leaf variant to evaluate.
    /// @param[in] x Column-major n x d input coordinate matrix.
    /// @param[out] out Destination span of length n receiving diagonal entries.
    /// @param[in] cached_ls Pre-cached ARD lengthscales in double, or empty to read from leaf.
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
    /// @param[in] leaf Kernel leaf variant to differentiate.
    /// @param[in] dist Column-major n x n squared-distance matrix.
    /// @param[out] d_k Column-major n x n destination derivative matrix.
    /// @param[in] param_idx Leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param[in] uplo Matrix triangle or full region to write.
    /// @param[in] exp_mode Exponential evaluation backend (Accurate or FastApprox).
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
    /// @param[in] leaf Kernel leaf variant to differentiate.
    /// @param[in] x Column-major n x d training coordinate matrix.
    /// @param[out] d_k Column-major n x n destination derivative matrix.
    /// @param[in] param_idx Leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param[in] uplo Matrix triangle or full region to write.
    /// @param[in] exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @param[in] cached_ls Pre-cached ARD lengthscales in double, or empty to read from leaf.
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
    /// @param[in] leaf Kernel leaf variant to differentiate.
    /// @param[in] dist Column-major n x n squared-distance matrix.
    /// @param[out] d2_k Column-major n x n destination second-derivative matrix.
    /// @param[in] i First leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param[in] j Second leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param[in] uplo Matrix triangle or full region to write.
    /// @param[in] exp_mode Exponential evaluation backend (Accurate or FastApprox).
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
    /// @param[in] leaf Kernel leaf variant to differentiate.
    /// @param[in] x Column-major n x d training coordinate matrix.
    /// @param[out] d2_k Column-major n x n destination second-derivative matrix.
    /// @param[in] i First leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param[in] j Second leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param[in] uplo Matrix triangle or full region to write.
    /// @param[in] exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @param[in] cached_ls Pre-cached ARD lengthscales in double, or empty to read from leaf.
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
    /// @param[in] leaf ARD kernel leaf variant to evaluate.
    /// @param[in] ls Lengthscale vector in user units of length d.
    /// @param[in] ard_packed Packed lower-triangular (dx_d)^2 array of length d * n * (n + 1) / 2.
    /// @param[in] n Number of training points.
    /// @param[out] out Column-major n x n destination Gram matrix.
    /// @param[in] uplo Matrix triangle or full region to write.
    /// @param[in] exp_mode Exponential evaluation backend (Accurate or FastApprox).
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

    /// @brief Evaluates an ARD kernel leaf using an incrementally updated lower-triangular $s^2$ cache.
    /// @details When `s2_valid` is true and only a small subset of lengthscales changed moderately since `prev_ls`,
    /// updates `ard_s2_cache` in $O(n^2)$ via AXPY over changed dimensions instead of rebuilding $s^2$ across all $d$
    /// dimensions. Falls back to full rebuild when invalid, step count saturates, or large jumps risk cancellation.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param[in] leaf ARD kernel leaf variant to evaluate.
    /// @param[in] ls Current lengthscale vector in user units of length d.
    /// @param[in] ard_packed Packed lower-triangular (dx_d)^2 array of length d * n * (n + 1) / 2.
    /// @param[in] n Number of training points.
    /// @param[out] out Column-major n x n destination Gram matrix.
    /// @param[in] uplo Matrix triangle or full region to write.
    /// @param[in] exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @param[in,out] ard_s2_cache Pre-allocated per-leaf packed $s^2$ buffer of length `n * (n + 1) / 2`.
    /// @param[in,out] prev_ls Pre-allocated lengthscale snapshot buffer of length at least `d`.
    /// @param[in,out] s2_valid True while `ard_s2_cache` corresponds bit-exactly to `prev_ls[0..d-1]`.
    /// @param[in,out] inc_steps Number of consecutive incremental delta updates applied since the last rebuild.
    /// @return Empty expected on success or GprError on shape mismatch or non-ARD leaf.
    /// @retval GprErrorCode::EmptyInput When n == 0 or ls is empty.
    /// @retval GprErrorCode::LengthMismatch When out shape is not n x n or ard_packed size disagrees.
    /// @retval GprErrorCode::UnsupportedKernelOperation When leaf does not consume packed ARD distances.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_apply_ard_packed_incremental(
        const KernelLeaf& leaf,
        std::span<const double> ls,
        std::span<const T> ard_packed,
        std::size_t n,
        MatrixMut<T> out,
        Triangle uplo,
        KernelExp exp_mode,
        std::span<T> ard_s2_cache,
        std::span<double> prev_ls,
        bool& s2_valid,
        std::size_t& inc_steps
    );

    /// @brief Evaluates the first derivative of an ARD leaf from packed lower-triangular ARD differences.
    /// @details Supports RbfArdKernel, MaternArdKernel, and RationalQuadraticArdKernel (including alpha).
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param[in] leaf ARD kernel leaf variant to differentiate.
    /// @param[in] ls Lengthscale vector in user units of length d.
    /// @param[in] ard_packed Packed lower-triangular (dx_d)^2 array of length d * n * (n + 1) / 2.
    /// @param[in] n Number of training points.
    /// @param[out] d_k Column-major n x n destination derivative matrix.
    /// @param[in] param_idx Leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param[in] uplo Matrix triangle or full region to write.
    /// @param[in] exp_mode Exponential evaluation backend (Accurate or FastApprox).
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
    /// @param[in] leaf ARD kernel leaf variant to differentiate.
    /// @param[in] ls Lengthscale vector in user units of length d.
    /// @param[in] ard_packed Packed lower-triangular (dx_d)^2 array of length d * n * (n + 1) / 2.
    /// @param[in] n Number of training points.
    /// @param[out] d2_k Column-major n x n destination second-derivative matrix.
    /// @param[in] i First leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param[in] j Second leaf-local parameter index in [0, leaf_num_params(leaf)).
    /// @param[in] uplo Matrix triangle or full region to write.
    /// @param[in] exp_mode Exponential evaluation backend (Accurate or FastApprox).
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

    /// @brief Contracts all hyperparameters of an ARD leaf against symmetric weight matrix `weight` in one radial pass.
    /// @details Computes scaled squared distances and radial derivative weights once across the lower triangle before
    /// contracting each ARD dimension (and alpha for RationalQuadraticArdKernel) deterministically via `col_partials`.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param[in] leaf ARD kernel leaf variant to contract.
    /// @param[in] ls Lengthscale vector in user units of length d.
    /// @param[in] x Column-major n x d training coordinate matrix (used when `ard_packed` is empty).
    /// @param[in] ard_packed Precomputed packed lower-triangular (dx_d)^2 buffer, or empty to read `x`.
    /// @param[in] weight Symmetric n x n adjoint weight matrix (lower triangle read).
    /// @param[in,out] scratch Reusable n x n square scratch matrix.
    /// @param[in,out] col_partials Pre-allocated per-column partial sum buffer of length at least n.
    /// @param[out] out_grads Destination span of length leaf_num_params(leaf) receiving <weight, dK/dtheta_p>_F.
    /// @param[in] need_value True when <weight, K_leaf>_F must also be evaluated and returned.
    /// @param[in] exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @return <weight, K_leaf>_F when `need_value` is true (otherwise 0.0), or GprError on invalid inputs/buffers.
    /// @retval GprErrorCode::UnsupportedKernelOperation When leaf is not an ARD leaf.
    /// @retval GprErrorCode::EmptyInput When `ls`, `weight`, or uncached `x` is empty.
    /// @retval GprErrorCode::LengthMismatch When matrix/cache shapes or `out_grads.size()` mismatch.
    /// @retval GprErrorCode::DimensionMismatch When `x.cols()` disagrees with `ls.size()`.
    /// @retval GprErrorCode::WorkspaceTooSmall When `scratch` or `col_partials` is undersized.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<double, GprError> leaf_ard_weighted_grads(
        const KernelLeaf& leaf,
        std::span<const double> ls,
        MatrixRef<T> x,
        std::span<const T> ard_packed,
        MatrixRef<T> weight,
        MatrixMut<T> scratch,
        std::span<double> col_partials,
        std::span<double> out_grads,
        bool need_value,
        KernelExp exp_mode = KernelExp::Accurate
    );

    /// @brief Contracts all hyperparameters of an isotropic or linear leaf against `weight` in one lower-triangle pass.
    /// @details Evaluates value-and-gradient jets once per pair `(row, col)` across OpenMP workers for
    /// `RbfKernel`, `MaternKernel`, `PeriodicKernel`, `RationalQuadraticKernel`, and `LinearKernel`, accumulating both
    /// `out_grads` and optional $\langle \text{weight}, K_{\text{leaf}} \rangle_F$ deterministically via
    /// `col_partials`.
    /// @tparam T Kernel evaluation scalar type (double or float).
    /// @param[in] leaf Isotropic or linear kernel leaf variant to contract.
    /// @param[in] x Column-major n x d training coordinate matrix (used for `LinearKernel` or when `dist` is empty).
    /// @param[in] dist Precomputed n x n squared-distance matrix, or empty view to compute slices from `x`.
    /// @param[in] weight Symmetric n x n adjoint weight matrix (lower triangle read).
    /// @param[in,out] scratch Reusable n x n square scratch matrix used for uncached/linear column slices.
    /// @param[in,out] col_partials Per-column partial sum buffer (length >= n; >= 3*n enables parallel multi-term).
    /// @param[out] out_grads Destination span of length leaf_num_params(leaf) receiving <weight, dK/dtheta_p>_F.
    /// @param[in] need_value True when <weight, K_leaf>_F must also be evaluated and returned.
    /// @param[in] exp_mode Exponential evaluation backend (Accurate or FastApprox).
    /// @return <weight, K_leaf>_F when `need_value` is true (otherwise 0.0), or GprError on invalid inputs/buffers.
    /// @retval GprErrorCode::UnsupportedKernelOperation When `leaf` is not isotropic-distance or `LinearKernel`.
    /// @retval GprErrorCode::EmptyInput When `weight`, required `x`, or supplied `dist` is empty/null.
    /// @retval GprErrorCode::LengthMismatch When matrix shapes or `out_grads.size()` disagree with n/leaf.
    /// @retval GprErrorCode::WorkspaceTooSmall When `scratch` or `col_partials` is undersized.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<double, GprError> leaf_iso_weighted_grads(
        const KernelLeaf& leaf,
        MatrixRef<T> x,
        MatrixRef<T> dist,
        MatrixRef<T> weight,
        MatrixMut<T> scratch,
        std::span<double> col_partials,
        std::span<double> out_grads,
        bool need_value,
        KernelExp exp_mode = KernelExp::Accurate
    );

}  // namespace cxxgpr::detail
