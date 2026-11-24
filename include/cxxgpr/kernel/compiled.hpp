// SPDX-License-Identifier: Apache-2.0
/// @file compiled.hpp
/// @brief Execution-layer compiled kernel evaluator and pre-allocated intermediate buffers.
/// @details Declares NestedBuffers<T> for zero-allocation tree evaluation and CompiledKernel<T>,
/// which associatively flattens KernelSpec trees, caches ARD lengthscales, supports Accurate and
/// FastApprox exponential backends, consumes DistanceCache<T>, and computes single-pass weighted
/// gradients for marginal likelihood optimization.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/interval.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/kernel/distance_cache.hpp"
#include "cxxgpr/kernel/exp.hpp"
#include "cxxgpr/kernel/spec.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <vector>

/// @brief Root namespace of the cxxgpr Gaussian Process Regression library.
namespace cxxgpr
{

    /// @brief Pre-allocated nested intermediate matrix buffers for zero-allocation tree evaluation.
    /// @details Allocates one contiguous column-major block holding depth matrices of shape rows x cols
    /// so recursive sum/product evaluations obtain non-aliasing scratch matrices via level().
    /// @tparam T Kernel evaluation scalar type (double or float).
    template <KernelScalarConcept T> class NestedBuffers
    {
    public:
        /// @brief Constructs empty nested buffers with zero capacity.
        /// @details Depth, rows, and cols are initialized to 0 until resize() is called.
        NestedBuffers() noexcept = default;

        /// @brief Constructs depth contiguous column-major buffers of shape rows x cols.
        /// @details Allocates depth * rows * cols elements upfront in a single contiguous vector.
        /// @param[in] depth Number of nested intermediate matrix levels.
        /// @param[in] rows Row capacity of each level matrix.
        /// @param[in] cols Column capacity of each level matrix.
        NestedBuffers(std::size_t depth, std::size_t rows, std::size_t cols);

        /// @brief Resizes buffers to hold depth matrices of shape rows x cols.
        /// @details Grows or reuses existing contiguous storage without shrinking allocation if capacity suffices.
        /// @param[in] depth Required number of nested buffer levels.
        /// @param[in] rows Required row capacity per level matrix.
        /// @param[in] cols Required column capacity per level matrix.
        void resize(std::size_t depth, std::size_t rows, std::size_t cols);

        /// @brief Returns the number of nested buffer levels.
        /// @details Indicates the maximum recursion depth index + 1 available via level().
        /// @return Allocated buffer level count.
        [[nodiscard]] std::size_t depth() const noexcept { return depth_; }

        /// @brief Returns the row capacity of each buffer.
        /// @details Every matrix view returned by level() must satisfy rows <= this capacity.
        /// @return Maximum row count per buffer level.
        [[nodiscard]] std::size_t rows() const noexcept { return rows_; }

        /// @brief Returns the column capacity of each buffer.
        /// @details Every matrix view returned by level() must satisfy cols <= this capacity.
        /// @return Maximum column count per buffer level.
        [[nodiscard]] std::size_t cols() const noexcept { return cols_; }

        /// @brief Returns a mutable view into buffer level index with shape rows x cols.
        /// @details Produces a tightly packed column-major MatrixMut<T> over storage slice index.
        /// @param[in] index Zero-based buffer level index in [0, depth()).
        /// @param[in] rows Logical row count for the returned matrix view.
        /// @param[in] cols Logical column count for the returned matrix view.
        /// @return Mutable column-major view of shape rows x cols backed by level index.
        [[nodiscard]] MatrixMut<T> level(std::size_t index, std::size_t rows, std::size_t cols);

    private:
        /// @brief Number of pre-allocated matrix levels.
        std::size_t depth_ = 0;
        /// @brief Row stride and capacity of each buffer level.
        std::size_t rows_ = 0;
        /// @brief Column capacity of each buffer level.
        std::size_t cols_ = 0;
        /// @brief Contiguous backing storage of length depth_ * rows_ * cols_.
        std::vector<T> storage_;
    };

    /// @brief Flattened execution-layer kernel tree dispatching statically over built-in leaves.
    /// @details Compiles a double-precision KernelSpec into an associatively flattened sum/product
    /// execution tree specialized on scalar type T, providing zero-allocation hot-path Gram,
    /// cross-covariance, derivative, Hessian, and weighted-gradient operations.
    /// @tparam T Kernel evaluation scalar type (double or float).
    template <KernelScalarConcept T> class CompiledKernel
    {
    public:
        /// @brief Compiles a KernelSpec into a flattened CompiledKernel<T>.
        /// @details Flattens nested sums and products associatively, caches ARD lengthscales, and
        /// precomputes parameter-to-leaf bindings and required nested buffer depth.
        /// @param[in] spec Source declaration-layer kernel expression tree.
        /// @return Compiled execution-layer kernel ready for Gram and derivative evaluation.
        [[nodiscard]] static CompiledKernel<T> from_spec(const KernelSpec& spec);

        /// @brief Copy-constructs a compiled kernel with independent mutable leaf state.
        /// @details Deep-copies the internal execution tree and cloned CustomKernel leaves.
        /// @param[in] other CompiledKernel instance to copy.
        CompiledKernel(const CompiledKernel& other);

        /// @brief Copy-assigns a compiled kernel with independent mutable leaf state.
        /// @details Deep-copies the internal execution tree when self-assignment does not occur.
        /// @param[in] other CompiledKernel instance to copy.
        /// @return Reference to this CompiledKernel.
        CompiledKernel& operator=(const CompiledKernel& other);

        /// @brief Move-constructs a compiled kernel.
        /// @details Transfers ownership of the internal compiled implementation pointer.
        /// @param[in] other CompiledKernel instance to move from.
        CompiledKernel(CompiledKernel&& other) noexcept;

        /// @brief Move-assigns a compiled kernel.
        /// @details Transfers ownership of the internal compiled implementation pointer.
        /// @param[in] other CompiledKernel instance to move from.
        /// @return Reference to this CompiledKernel.
        CompiledKernel& operator=(CompiledKernel&& other) noexcept;

        /// @brief Destroys the compiled kernel and releases its internal implementation.
        /// @details Frees the Pimpl implementation out of line where Impl is complete.
        ~CompiledKernel();

        /// @brief Returns the total number of log-space hyperparameters.
        /// @details Matches the size of parameter_bindings() and expected spans for get_params/set_params.
        /// @return Total number of hyperparameters across all leaves.
        [[nodiscard]] std::size_t num_params() const noexcept;

        /// @brief Returns the number of leaf nodes in depth-first left-to-right order.
        /// @details Equals the required length of leaf_grams passed to combine_from_leaf_grams().
        /// @return Number of leaves in the flattened tree.
        [[nodiscard]] std::size_t num_leaves() const noexcept;

        /// @brief Returns the input coordinate mode required by the compiled tree.
        /// @details Indicates whether squared distances, point coordinates, or both are supported.
        /// @return Aggregate CoordMode of the compiled kernel tree.
        [[nodiscard]] CoordMode coord_mode() const noexcept;

        /// @brief Returns the number of nested intermediate buffers required for tree evaluation.
        /// @details Matches the depth parameter passed to NestedBuffers<T> via make_nested_buffers().
        /// @return Required NestedBuffers depth.
        [[nodiscard]] std::size_t nested_depth() const noexcept;

        /// @brief Returns true if any leaf in the compiled tree reads squared-distance caches.
        /// @details Guides DistanceCache construction for isotropic distance and CustomKernel leaves.
        /// @return True when isotropic or custom distance leaves are present.
        [[nodiscard]] bool reads_distances() const noexcept;

        /// @brief Returns true if any leaf in the compiled tree reads packed ARD squared-difference caches.
        /// @details Guides DistanceCache construction for RbfArdKernel, MaternArdKernel, and
        /// RationalQuadraticArdKernel leaves.
        /// @return True when ARD leaves are present.
        [[nodiscard]] bool needs_ard_sq_diff() const noexcept;

        /// @brief Returns the number of ARD kernel leaves in the compiled tree.
        /// @return Count of RbfArdKernel, MaternArdKernel, and RationalQuadraticArdKernel leaves.
        [[nodiscard]] std::size_t ard_leaf_count() const noexcept;

        /// @brief Returns per-leaf ARD lengthscale counts in depth-first leaf order.
        /// @details Entry i equals the feature dimension count when leaf i is an ARD leaf, or 0 otherwise.
        /// @return Span of length num_leaves() giving ARD dimension counts per leaf.
        [[nodiscard]] std::span<const std::size_t> ard_dims_per_leaf() const noexcept;

        /// @brief Returns the exponential evaluation backend used during kernel evaluation.
        /// @details Defaults to KernelExp::Accurate upon compilation from KernelSpec.
        /// @return Active KernelExp selector.
        [[nodiscard]] KernelExp kernel_exp() const noexcept;

        /// @brief Sets the exponential evaluation backend used during kernel evaluation.
        /// @details Selects KernelExp::Accurate or KernelExp::FastApprox for built-in exponential leaves.
        /// @param[in] mode New exponential evaluation backend.
        void set_kernel_exp(KernelExp mode) noexcept;

        /// @brief Allocates NestedBuffers sized for this kernel and an output matrix of shape rows x cols.
        /// @details Allocates nested_depth() intermediate matrices of dimensions rows x cols.
        /// @param[in] rows Target matrix row count.
        /// @param[in] cols Target matrix column count.
        /// @return Pre-sized NestedBuffers<T> ready for zero-allocation kernel calls.
        [[nodiscard]] NestedBuffers<T> make_nested_buffers(std::size_t rows, std::size_t cols) const;

        /// @brief Writes flattened log-space hyperparameters into out.
        /// @details Traverses leaves in depth-first left-to-right order.
        /// @param[out] out Destination span of length num_params().
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Atomically updates all leaf parameters from log-space params.
        /// @details Validates every leaf slice into a staging copy and refreshes cached ARD lengthscales
        /// before committing; failed updates leave parameters unchanged.
        /// @param[in] params Input span of unconstrained log-space parameters of length num_params().
        /// @return Empty expected on success or GprError on count mismatch, non-finite input, or bound error.
        /// @retval GprErrorCode::LengthMismatch When params.size() != num_params().
        /// @retval GprErrorCode::NonFiniteInput When any parameter is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When any exponentiated value exits its leaf interval.
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes flattened open intervals into out.
        /// @details Populates intervals in depth-first left-to-right parameter order.
        /// @param[out] out Destination span of length num_params().
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Returns flattened parameter-to-leaf bindings.
        /// @details Ordered identically to get_params() and set_params().
        /// @return Const reference to the vector of ParameterBinding descriptors.
        [[nodiscard]] const std::vector<ParameterBinding>& parameter_bindings() const noexcept;

        /// @brief Evaluates the training Gram matrix from squared Euclidean distances using caller buffers.
        /// @details Requires coord_mode() to support distance matrices (CoordMode::Dist or Either).
        /// @param[in] dist Column-major n x n squared-distance matrix.
        /// @param[out] out Column-major n x n destination Gram matrix.
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch Non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @return Empty expected on success or GprError on invalid inputs, buffers, or mode.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::UnsupportedKernelOperation When the tree requires point coordinates.
        /// @retval GprErrorCode::EmptyInput When dist is empty.
        /// @retval GprErrorCode::LengthMismatch When dist is non-square or out shape disagrees.
        /// @retval GprErrorCode::NonFiniteInput When active dist entries are non-finite.
        [[nodiscard]] std::expected<void, GprError> apply(
            MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo, MatrixMut<T> scratch, NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates the training Gram matrix from squared Euclidean distances.
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param[in] dist Column-major n x n squared-distance matrix.
        /// @param[out] out Column-major n x n destination Gram matrix.
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch Non-aliasing n x n scratch matrix buffer.
        /// @return Empty expected on success or GprError on invalid inputs, buffers, or mode.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch is undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::UnsupportedKernelOperation When the tree requires point coordinates.
        /// @retval GprErrorCode::EmptyInput When dist is empty.
        /// @retval GprErrorCode::LengthMismatch When dist is non-square or out shape disagrees.
        /// @retval GprErrorCode::NonFiniteInput When active dist entries are non-finite.
        [[nodiscard]] std::expected<void, GprError> apply(
            MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo, MatrixMut<T> scratch
        ) const;

        /// @brief Evaluates the training Gram matrix from input point coordinates using caller buffers.
        /// @details Supports all built-in leaves (isotropic, ARD, Linear, Constant, White); CustomKernel
        /// computes squared distances internally into scratch/nested buffers.
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[out] out Column-major n x n destination Gram matrix.
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch Non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @return Empty expected on success or GprError on invalid inputs, dimensions, or buffers.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When out shape is not n x n.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        [[nodiscard]] std::expected<void, GprError> apply_points(
            MatrixRef<T> x, MatrixMut<T> out, Triangle uplo, MatrixMut<T> scratch, NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates the training Gram matrix from input point coordinates.
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[out] out Column-major n x n destination Gram matrix.
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch Non-aliasing n x n scratch matrix buffer.
        /// @return Empty expected on success or GprError on invalid inputs, dimensions, or buffers.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch is undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When out shape is not n x n.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        [[nodiscard]] std::expected<void, GprError> apply_points(
            MatrixRef<T> x, MatrixMut<T> out, Triangle uplo, MatrixMut<T> scratch
        ) const;

        /// @brief Evaluates the cross-covariance matrix from rectangular squared distances using caller buffers.
        /// @details Requires coord_mode() to support distance matrices (CoordMode::Dist or Either).
        /// @param[in] dist Column-major n1 x n2 rectangular squared-distance matrix.
        /// @param[out] out Column-major n1 x n2 destination cross-covariance matrix.
        /// @param[in,out] scratch Non-aliasing n1 x n2 scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n1 x n2.
        /// @return Empty expected on success or GprError on invalid inputs, buffers, or mode.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::UnsupportedKernelOperation When the tree requires point coordinates.
        /// @retval GprErrorCode::EmptyInput When dist is empty.
        /// @retval GprErrorCode::LengthMismatch When out shape differs from dist.
        /// @retval GprErrorCode::NonFiniteInput When any entry of dist is non-finite.
        [[nodiscard]] std::expected<void, GprError> apply_cross(
            MatrixRef<T> dist, MatrixMut<T> out, MatrixMut<T> scratch, NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates the cross-covariance matrix from rectangular squared distances.
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param[in] dist Column-major n1 x n2 rectangular squared-distance matrix.
        /// @param[out] out Column-major n1 x n2 destination cross-covariance matrix.
        /// @param[in,out] scratch Non-aliasing n1 x n2 scratch matrix buffer.
        /// @return Empty expected on success or GprError on invalid inputs, buffers, or mode.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch is undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::UnsupportedKernelOperation When the tree requires point coordinates.
        /// @retval GprErrorCode::EmptyInput When dist is empty.
        /// @retval GprErrorCode::LengthMismatch When out shape differs from dist.
        /// @retval GprErrorCode::NonFiniteInput When any entry of dist is non-finite.
        [[nodiscard]] std::expected<void, GprError> apply_cross(
            MatrixRef<T> dist, MatrixMut<T> out, MatrixMut<T> scratch
        ) const;

        /// @brief Evaluates the cross-covariance matrix from point coordinates x1 and x2 using caller buffers.
        /// @details Computes k(x1_i, x2_j) across all pairs for any built-in or CustomKernel tree.
        /// @param[in] x1 First input coordinate matrix of shape n1 x d in column-major order.
        /// @param[in] x2 Second input coordinate matrix of shape n2 x d in column-major order.
        /// @param[out] out Column-major n1 x n2 destination cross-covariance matrix.
        /// @param[in,out] scratch Non-aliasing n1 x n2 scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n1 x n2.
        /// @return Empty expected on success or GprError on invalid inputs, dimensions, or buffers.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::EmptyInput When x1 or x2 is empty.
        /// @retval GprErrorCode::LengthMismatch When out shape is not n1 x n2.
        /// @retval GprErrorCode::DimensionMismatch When x1.cols() != x2.cols() or disagrees with ARD dims.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x1 or x2 is non-finite.
        [[nodiscard]] std::expected<void, GprError> apply_cross_points(
            MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out, MatrixMut<T> scratch, NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates the cross-covariance matrix from point coordinates x1 and x2.
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param[in] x1 First input coordinate matrix of shape n1 x d in column-major order.
        /// @param[in] x2 Second input coordinate matrix of shape n2 x d in column-major order.
        /// @param[out] out Column-major n1 x n2 destination cross-covariance matrix.
        /// @param[in,out] scratch Non-aliasing n1 x n2 scratch matrix buffer.
        /// @return Empty expected on success or GprError on invalid inputs, dimensions, or buffers.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch is undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::EmptyInput When x1 or x2 is empty.
        /// @retval GprErrorCode::LengthMismatch When out shape is not n1 x n2.
        /// @retval GprErrorCode::DimensionMismatch When x1.cols() != x2.cols() or disagrees with ARD dims.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x1 or x2 is non-finite.
        [[nodiscard]] std::expected<void, GprError> apply_cross_points(
            MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out, MatrixMut<T> scratch
        ) const;

        /// @brief Fills the diagonal k(x_i, x_i) for points x using caller-provided scratch (zero alloc).
        /// @details Requires scratch length at least (nested_depth() + 1) * x.rows() for zero-allocation execution.
        /// @param[in] x Column-major n x d input coordinate matrix.
        /// @param[out] out Destination span of length n receiving k(x_i, x_i).
        /// @param[in,out] scratch Caller-supplied scratch span of length >= (nested_depth() + 1) * x.rows().
        /// @return Empty expected on success or GprError on invalid inputs, dimensions, or undersized scratch.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != x.rows() or scratch is too small.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        [[nodiscard]] std::expected<void, GprError> fill_diag(
            MatrixRef<T> x, std::span<T> out, std::span<T> scratch
        ) const;

        /// @brief Fills the diagonal k(x_i, x_i) for points x.
        /// @details Convenience overload allocating a temporary scratch vector before delegating to fill_diag().
        /// @param[in] x Column-major n x d input coordinate matrix.
        /// @param[out] out Destination span of length n receiving k(x_i, x_i).
        /// @return Empty expected on success or GprError on invalid inputs or dimensions.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != x.rows().
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        [[nodiscard]] std::expected<void, GprError> fill_diag(MatrixRef<T> x, std::span<T> out) const;

        /// @brief Evaluates the first derivative wrt parameter param_idx from squared distances.
        /// @details Applies product-rule differentiation using two non-aliasing n x n scratch buffers.
        /// @param[in] dist Column-major n x n squared-distance matrix.
        /// @param[out] d_k Column-major n x n destination derivative matrix.
        /// @param[in] param_idx Zero-based flattened hyperparameter index in [0, num_params()).
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @return Empty expected on success or GprError on invalid buffers, inputs, or index.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch buffers alias d_k or each other.
        /// @retval GprErrorCode::UnsupportedKernelOperation When the tree requires point coordinates.
        /// @retval GprErrorCode::EmptyInput When dist is empty.
        /// @retval GprErrorCode::LengthMismatch When dist is non-square or d_k shape disagrees.
        /// @retval GprErrorCode::NonFiniteInput When active dist entries are non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When param_idx >= num_params().
        [[nodiscard]] std::expected<void, GprError> grad(
            MatrixRef<T> dist,
            MatrixMut<T> d_k,
            std::size_t param_idx,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates the first derivative wrt parameter param_idx from squared distances.
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param[in] dist Column-major n x n squared-distance matrix.
        /// @param[out] d_k Column-major n x n destination derivative matrix.
        /// @param[in] param_idx Zero-based flattened hyperparameter index in [0, num_params()).
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @return Empty expected on success or GprError on invalid buffers, inputs, or index.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch buffers alias d_k or each other.
        /// @retval GprErrorCode::UnsupportedKernelOperation When the tree requires point coordinates.
        /// @retval GprErrorCode::EmptyInput When dist is empty.
        /// @retval GprErrorCode::LengthMismatch When dist is non-square or d_k shape disagrees.
        /// @retval GprErrorCode::NonFiniteInput When active dist entries are non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When param_idx >= num_params().
        [[nodiscard]] std::expected<void, GprError> grad(
            MatrixRef<T> dist,
            MatrixMut<T> d_k,
            std::size_t param_idx,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b
        ) const;

        /// @brief Evaluates the first derivative wrt parameter param_idx from point coordinates.
        /// @details Applies product-rule differentiation over coordinate inputs using caller buffers.
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[out] d_k Column-major n x n destination derivative matrix.
        /// @param[in] param_idx Zero-based flattened hyperparameter index in [0, num_params()).
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @return Empty expected on success or GprError on invalid buffers, inputs, or index.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch buffers alias d_k or each other.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When d_k shape is not n x n.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When param_idx >= num_params().
        [[nodiscard]] std::expected<void, GprError> grad_points(
            MatrixRef<T> x,
            MatrixMut<T> d_k,
            std::size_t param_idx,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates the first derivative wrt parameter param_idx from point coordinates.
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[out] d_k Column-major n x n destination derivative matrix.
        /// @param[in] param_idx Zero-based flattened hyperparameter index in [0, num_params()).
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @return Empty expected on success or GprError on invalid buffers, inputs, or index.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch buffers alias d_k or each other.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When d_k shape is not n x n.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When param_idx >= num_params().
        [[nodiscard]] std::expected<void, GprError> grad_points(
            MatrixRef<T> x,
            MatrixMut<T> d_k,
            std::size_t param_idx,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b
        ) const;

        /// @brief Evaluates the second derivative wrt parameters (i, j) from squared distances.
        /// @details Applies second-order product rule using three non-aliasing n x n scratch buffers.
        /// @param[in] dist Column-major n x n squared-distance matrix.
        /// @param[out] d2_k Column-major n x n destination second-derivative matrix.
        /// @param[in] i First flattened hyperparameter index in [0, num_params()).
        /// @param[in] j Second flattened hyperparameter index in [0, num_params()).
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_c Third non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @return Empty expected on success or GprError on invalid buffers, inputs, or indices.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When any scratch buffers alias d2_k or each other.
        /// @retval GprErrorCode::UnsupportedKernelOperation When the tree requires point coordinates.
        /// @retval GprErrorCode::EmptyInput When dist is empty.
        /// @retval GprErrorCode::LengthMismatch When dist is non-square or d2_k shape disagrees.
        /// @retval GprErrorCode::NonFiniteInput When active dist entries are non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When i >= num_params() or j >= num_params().
        [[nodiscard]] std::expected<void, GprError> hess(
            MatrixRef<T> dist,
            MatrixMut<T> d2_k,
            std::size_t i,
            std::size_t j,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            MatrixMut<T> scratch_c,
            NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates the second derivative wrt parameters (i, j) from squared distances.
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param[in] dist Column-major n x n squared-distance matrix.
        /// @param[out] d2_k Column-major n x n destination second-derivative matrix.
        /// @param[in] i First flattened hyperparameter index in [0, num_params()).
        /// @param[in] j Second flattened hyperparameter index in [0, num_params()).
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_c Third non-aliasing n x n scratch matrix buffer.
        /// @return Empty expected on success or GprError on invalid buffers, inputs, or indices.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When any scratch buffers alias d2_k or each other.
        /// @retval GprErrorCode::UnsupportedKernelOperation When the tree requires point coordinates.
        /// @retval GprErrorCode::EmptyInput When dist is empty.
        /// @retval GprErrorCode::LengthMismatch When dist is non-square or d2_k shape disagrees.
        /// @retval GprErrorCode::NonFiniteInput When active dist entries are non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When i >= num_params() or j >= num_params().
        [[nodiscard]] std::expected<void, GprError> hess(
            MatrixRef<T> dist,
            MatrixMut<T> d2_k,
            std::size_t i,
            std::size_t j,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            MatrixMut<T> scratch_c
        ) const;

        /// @brief Evaluates the second derivative wrt parameters (i, j) from point coordinates.
        /// @details Applies second-order product rule over coordinate inputs using caller buffers.
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[out] d2_k Column-major n x n destination second-derivative matrix.
        /// @param[in] i First flattened hyperparameter index in [0, num_params()).
        /// @param[in] j Second flattened hyperparameter index in [0, num_params()).
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_c Third non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @return Empty expected on success or GprError on invalid buffers, inputs, or indices.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When any scratch buffers alias d2_k or each other.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When d2_k shape is not n x n.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When i >= num_params() or j >= num_params().
        [[nodiscard]] std::expected<void, GprError> hess_points(
            MatrixRef<T> x,
            MatrixMut<T> d2_k,
            std::size_t i,
            std::size_t j,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            MatrixMut<T> scratch_c,
            NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates the second derivative wrt parameters (i, j) from point coordinates.
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[out] d2_k Column-major n x n destination second-derivative matrix.
        /// @param[in] i First flattened hyperparameter index in [0, num_params()).
        /// @param[in] j Second flattened hyperparameter index in [0, num_params()).
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_c Third non-aliasing n x n scratch matrix buffer.
        /// @return Empty expected on success or GprError on invalid buffers, inputs, or indices.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When any scratch buffers alias d2_k or each other.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When d2_k shape is not n x n.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When i >= num_params() or j >= num_params().
        [[nodiscard]] std::expected<void, GprError> hess_points(
            MatrixRef<T> x,
            MatrixMut<T> d2_k,
            std::size_t i,
            std::size_t j,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            MatrixMut<T> scratch_c
        ) const;

        /// @brief Evaluates the training Gram matrix using precomputed distance and ARD caches when populated.
        /// @details Falls back to coordinate evaluation from x for leaves whose required cache is empty.
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] cache Precomputed DistanceCache matching x in shape and dimension.
        /// @param[out] out Column-major n x n destination Gram matrix.
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch Non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @return Empty expected on success or GprError on invalid inputs, cache mismatch, or buffers.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When out or cache row count/length mismatches x.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with ARD leaf or cache dims.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        [[nodiscard]] std::expected<void, GprError> apply_cached(
            MatrixRef<T> x,
            const DistanceCache<T>& cache,
            MatrixMut<T> out,
            Triangle uplo,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates the training Gram matrix using precomputed distance and ARD caches when populated.
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] cache Precomputed DistanceCache matching x in shape and dimension.
        /// @param[out] out Column-major n x n destination Gram matrix.
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch Non-aliasing n x n scratch matrix buffer.
        /// @return Empty expected on success or GprError on invalid inputs, cache mismatch, or buffers.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch is undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When out or cache row count/length mismatches x.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with ARD leaf or cache dims.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        [[nodiscard]] std::expected<void, GprError> apply_cached(
            MatrixRef<T> x, const DistanceCache<T>& cache, MatrixMut<T> out, Triangle uplo, MatrixMut<T> scratch
        ) const;

        /// @brief Evaluates the first derivative wrt param_idx using precomputed caches when populated.
        /// @details Uses cached squared distances and packed ARD differences whenever available in cache.
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] cache Precomputed DistanceCache matching x in shape and dimension.
        /// @param[out] d_k Column-major n x n destination derivative matrix.
        /// @param[in] param_idx Zero-based flattened hyperparameter index in [0, num_params()).
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @return Empty expected on success or GprError on invalid buffers, cache mismatch, or index.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch buffers alias d_k or each other.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When d_k or cache shape mismatches x.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with ARD leaf or cache dims.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When param_idx >= num_params().
        [[nodiscard]] std::expected<void, GprError> grad_cached(
            MatrixRef<T> x,
            const DistanceCache<T>& cache,
            MatrixMut<T> d_k,
            std::size_t param_idx,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates the first derivative wrt param_idx using precomputed caches when populated.
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] cache Precomputed DistanceCache matching x in shape and dimension.
        /// @param[out] d_k Column-major n x n destination derivative matrix.
        /// @param[in] param_idx Zero-based flattened hyperparameter index in [0, num_params()).
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @return Empty expected on success or GprError on invalid buffers, cache mismatch, or index.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch buffers alias d_k or each other.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When d_k or cache shape mismatches x.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with ARD leaf or cache dims.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When param_idx >= num_params().
        [[nodiscard]] std::expected<void, GprError> grad_cached(
            MatrixRef<T> x,
            const DistanceCache<T>& cache,
            MatrixMut<T> d_k,
            std::size_t param_idx,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b
        ) const;

        /// @brief Evaluates the second derivative wrt (i, j) using precomputed caches when populated.
        /// @details Uses cached squared distances and packed ARD differences whenever available in cache.
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] cache Precomputed DistanceCache matching x in shape and dimension.
        /// @param[out] d2_k Column-major n x n destination second-derivative matrix.
        /// @param[in] i First flattened hyperparameter index in [0, num_params()).
        /// @param[in] j Second flattened hyperparameter index in [0, num_params()).
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_c Third non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @return Empty expected on success or GprError on invalid buffers, cache mismatch, or indices.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When any scratch buffers alias d2_k or each other.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When d2_k or cache shape mismatches x.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with ARD leaf or cache dims.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When i >= num_params() or j >= num_params().
        [[nodiscard]] std::expected<void, GprError> hess_cached(
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
        ) const;

        /// @brief Evaluates the second derivative wrt (i, j) using precomputed caches when populated.
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] cache Precomputed DistanceCache matching x in shape and dimension.
        /// @param[out] d2_k Column-major n x n destination second-derivative matrix.
        /// @param[in] i First flattened hyperparameter index in [0, num_params()).
        /// @param[in] j Second flattened hyperparameter index in [0, num_params()).
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] scratch_c Third non-aliasing n x n scratch matrix buffer.
        /// @return Empty expected on success or GprError on invalid buffers, cache mismatch, or indices.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When any scratch buffers alias d2_k or each other.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When d2_k or cache shape mismatches x.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with ARD leaf or cache dims.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When i >= num_params() or j >= num_params().
        [[nodiscard]] std::expected<void, GprError> hess_cached(
            MatrixRef<T> x,
            const DistanceCache<T>& cache,
            MatrixMut<T> d2_k,
            std::size_t i,
            std::size_t j,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            MatrixMut<T> scratch_c
        ) const;

        /// @brief Returns the leaf index in [0, num_leaves()) owning flattened parameter param_idx.
        /// @details Looks up bindings[param_idx].leaf_id after checking bounds.
        /// @param[in] param_idx Zero-based flattened hyperparameter index.
        /// @return Owning leaf index or GprError if param_idx is out of range.
        /// @retval GprErrorCode::IndexOutOfRange When param_idx >= num_params().
        [[nodiscard]] std::expected<std::size_t, GprError> leaf_index_for_param(std::size_t param_idx) const;

        /// @brief Evaluates a single leaf kernel from point coordinates x into out.
        /// @details Evaluates leaf leaf_idx in isolation without computing sibling leaves.
        /// @param[in] leaf_idx Zero-based leaf index in [0, num_leaves()).
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[out] out Column-major n x n destination leaf Gram matrix.
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] nested Pre-allocated NestedBuffers used if leaf_idx is a CustomKernel.
        /// @return Empty expected on success or GprError on bad index, inputs, or buffers.
        /// @retval GprErrorCode::IndexOutOfRange When leaf_idx >= num_leaves().
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When out shape is not n x n.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::WorkspaceTooSmall When nested is too small for CustomKernel distance scratch.
        [[nodiscard]] std::expected<void, GprError> apply_leaf_points(
            std::size_t leaf_idx, MatrixRef<T> x, MatrixMut<T> out, Triangle uplo, NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates a single leaf kernel using precomputed distance and ARD caches when populated.
        /// @details Evaluates leaf leaf_idx using cache when populated or falling back to x.
        /// @param[in] leaf_idx Zero-based leaf index in [0, num_leaves()).
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] cache Precomputed DistanceCache matching x in shape and dimension.
        /// @param[out] out Column-major n x n destination leaf Gram matrix.
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] nested Pre-allocated NestedBuffers used when falling back for CustomKernel.
        /// @return Empty expected on success or GprError on bad index, inputs, cache mismatch, or buffers.
        /// @retval GprErrorCode::IndexOutOfRange When leaf_idx >= num_leaves().
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When out or cache shape mismatches x.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with ARD leaf or cache dims.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::WorkspaceTooSmall When nested is too small for CustomKernel fallback.
        [[nodiscard]] std::expected<void, GprError> apply_leaf_cached(
            std::size_t leaf_idx,
            MatrixRef<T> x,
            const DistanceCache<T>& cache,
            MatrixMut<T> out,
            Triangle uplo,
            NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates a single leaf kernel using DistanceCache<T> and an incremental ARD squared-distance cache.
        /// @details For ARD leaves when cache.has_ard() holds, updates ard_s2_cache incrementally over changed
        /// lengthscale dimensions instead of summing across all dimensions; falls back to apply_leaf_cached()
        /// for non-ARD leaves or when cache has no packed ARD table.
        /// @param[in] leaf_idx Zero-based leaf index in [0, num_leaves()).
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] cache Precomputed DistanceCache matching x in shape and dimension.
        /// @param[out] out Column-major n x n destination leaf Gram matrix.
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] nested Pre-allocated NestedBuffers used when falling back for CustomKernel.
        /// @param[in,out] ard_s2_cache Caller-owned packed s^2 buffer of length n * (n + 1) / 2 for this leaf.
        /// @param[in,out] prev_ls Caller-owned lengthscale snapshot buffer for this leaf.
        /// @param[in,out] s2_valid True while ard_s2_cache matches prev_ls.
        /// @param[in,out] inc_steps Consecutive incremental updates applied since the last full rebuild.
        /// @param[in] commit_s2 True to update `ard_s2_cache`/`prev_ls`; false for non-destructive trials.
        /// @return Empty expected on success or GprError on bad index, inputs, cache mismatch, or buffers.
        /// @retval GprErrorCode::IndexOutOfRange When leaf_idx >= num_leaves().
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When out or cache shape mismatches x.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with ARD leaf or cache dims.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::WorkspaceTooSmall When nested is too small for CustomKernel fallback.
        [[nodiscard]] std::expected<void, GprError> apply_leaf_cached_incremental(
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
            bool commit_s2 = true
        ) const;

        /// @brief Evaluates a single leaf kernel incrementally using an optional DistanceCache<T>.
        /// @details For ARD leaves in either cached (`cache != nullptr && cache->has_ard()`) or uncached coordinate
        /// mode, fuses $O(n^2)$ delta updates over changed dimensions with radial evaluation in one pass; leaves
        /// `ard_s2_cache` and `prev_ls` unmodified when `commit_s2` is false for non-destructive proposal trials.
        /// @param[in] leaf_idx Zero-based leaf index in [0, num_leaves()).
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] cache Optional precomputed DistanceCache pointer, or nullptr for uncached evaluation.
        /// @param[out] out Column-major n x n destination leaf Gram matrix.
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] nested Pre-allocated NestedBuffers used when falling back for CustomKernel.
        /// @param[in,out] ard_s2_cache Caller-owned packed s^2 buffer of length n * (n + 1) / 2 for this leaf.
        /// @param[in,out] prev_ls Caller-owned lengthscale snapshot buffer for this leaf.
        /// @param[in,out] s2_valid True while ard_s2_cache matches prev_ls.
        /// @param[in,out] inc_steps Consecutive incremental updates applied since the last full rebuild.
        /// @param[in] commit_s2 True to commit baseline cache state; false for non-destructive trial evaluation.
        /// @return Empty expected on success or GprError on bad index, inputs, cache mismatch, or buffers.
        /// @retval GprErrorCode::IndexOutOfRange When leaf_idx >= num_leaves().
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When out or cache shape mismatches x.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with ARD leaf or cache dims.
        /// @retval GprErrorCode::NonFiniteInput When any touched entry of x is non-finite.
        /// @retval GprErrorCode::WorkspaceTooSmall When nested is too small for CustomKernel fallback.
        [[nodiscard]] std::expected<void, GprError> apply_leaf_incremental(
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
            bool commit_s2 = true
        ) const;

        /// @brief Commits an ARD leaf's current compiled lengthscales into its packed $s^2$ baseline cache.
        /// @details Applies $O(n^2)$ delta updates while numerically safe, or rebuilds cleanly on step saturation or
        /// wide scale changes; no-op for non-ARD leaves.
        /// @param[in] leaf_idx Zero-based leaf index in [0, num_leaves()).
        /// @param[in] x Column-major n x d training coordinate matrix (used when uncached).
        /// @param[in] cache Optional precomputed DistanceCache pointer, or nullptr for uncached evaluation.
        /// @param[in,out] ard_s2_cache Caller-owned packed s^2 buffer of length n * (n + 1) / 2 for this leaf.
        /// @param[in,out] prev_ls Caller-owned lengthscale snapshot buffer for this leaf.
        /// @param[in,out] s2_valid True while ard_s2_cache matches prev_ls.
        /// @param[in,out] inc_steps Consecutive incremental updates applied since the last full rebuild.
        /// @return Empty expected on success or GprError on bad index, inputs, or cache shape mismatch.
        /// @retval GprErrorCode::IndexOutOfRange When leaf_idx >= num_leaves().
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When baseline buffers or cache shape mismatch.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with ARD leaf or cache dims.
        /// @retval GprErrorCode::NonFiniteInput When any touched entry of x is non-finite.
        [[nodiscard]] std::expected<void, GprError> commit_leaf_ard_s2(
            std::size_t leaf_idx,
            MatrixRef<T> x,
            const DistanceCache<T>* cache,
            std::span<T> ard_s2_cache,
            std::span<double> prev_ls,
            bool& s2_valid,
            std::size_t& inc_steps
        ) const;

        /// @brief Reconstructs the composite Gram matrix from precomputed per-leaf Gram matrices.
        /// @details Combines leaf_grams[0..num_leaves()-1] according to the flattened sum/product AST
        /// without recomputing untouched leaves.
        /// @param[in] leaf_grams Span of num_leaves() square n x n Gram matrix views in depth-first leaf order.
        /// @param[out] out Column-major n x n destination composite Gram matrix.
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch Non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @return Empty expected on success or GprError on invalid count, shapes, or buffers.
        /// @retval GprErrorCode::EmptyInput When out is empty or null.
        /// @retval GprErrorCode::LengthMismatch When out is non-square, leaf_grams count disagrees, or shapes differ.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        [[nodiscard]] std::expected<void, GprError> combine_from_leaf_grams(
            std::span<const MatrixRef<T>> leaf_grams,
            MatrixMut<T> out,
            Triangle uplo,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested
        ) const;

        /// @brief Returns the maximum number of top-level products whose factor Grams can be kept.
        /// @details Counts top-level product terms with at least two non-constant factors whose factor
        /// Grams fit within the unkept contraction buffer budget for single-pass weighted_grads().
        /// @return Maximum number of kept top-level product terms.
        [[nodiscard]] std::size_t kept_products() const noexcept;

        /// @brief Returns the number of n x n buffers kept for the first products top-level products.
        /// @details Sums the number of non-constant factors across the first products qualifying top-level
        /// product terms.
        /// @param[in] products Number of top-level product terms to keep.
        /// @return Number of n x n buffers required in kept_bufs.
        [[nodiscard]] std::size_t kept_buffers(std::size_t products) const noexcept;

        /// @brief Returns the total n x n buffers needed by weighted_grads when keeping products products.
        /// @details Equals kept_buffers(products) plus the peak walk scratch buffers needed for remaining terms.
        /// @param[in] products Number of top-level product terms whose factor Grams are kept.
        /// @return Minimum depth required for the weighted_bufs argument to weighted_grads().
        [[nodiscard]] std::size_t weighted_buffers(std::size_t products) const noexcept;

        /// @brief Evaluates the training Gram matrix from points x while saving kept product factor Grams.
        /// @details Populates kept_bufs[0..kept_buffers(products)-1] for reuse in weighted_grads().
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[out] out Column-major n x n destination Gram matrix.
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch Non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @param[in,out] kept_bufs Pre-allocated NestedBuffers with depth >= kept_buffers(products) and shape >= n x
        /// n.
        /// @param[in] products Number of top-level product terms whose factor Grams to retain.
        /// @param[out] ard_s2_buf Optional pre-allocated buffer capturing forward packed ARD $s^2$ per ARD leaf.
        /// @param[in,out] ard_s2_ready Optional per-ARD-leaf readiness bytes set to `1` when captured.
        /// @return Empty expected on success or GprError on invalid inputs, dimensions, or undersized buffers.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch, nested, or kept_bufs is undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When out shape is not n x n.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with an ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        [[nodiscard]] std::expected<void, GprError> apply_keeping(
            MatrixRef<T> x,
            MatrixMut<T> out,
            Triangle uplo,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested,
            NestedBuffers<T>& kept_bufs,
            std::size_t products,
            std::span<T> ard_s2_buf = {},
            std::span<std::uint8_t> ard_s2_ready = {}
        ) const;

        /// @brief Evaluates the training Gram matrix using cached distances while saving kept factor Grams.
        /// @details Combines DistanceCache reuse with kept factor Gram retention for single-pass gradients.
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] cache Precomputed DistanceCache matching x in shape and dimension.
        /// @param[out] out Column-major n x n destination Gram matrix.
        /// @param[in] uplo Matrix triangle or full region to populate.
        /// @param[in,out] scratch Non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @param[in,out] kept_bufs Pre-allocated NestedBuffers with depth >= kept_buffers(products) and shape >= n x
        /// n.
        /// @param[in] products Number of top-level product terms whose factor Grams to retain.
        /// @param[out] ard_s2_buf Optional pre-allocated buffer capturing forward packed ARD $s^2$ per ARD leaf.
        /// @param[in,out] ard_s2_ready Optional per-ARD-leaf readiness bytes set to `1` when captured.
        /// @return Empty expected on success or GprError on invalid inputs, cache mismatch, or buffers.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch, nested, or kept_bufs is undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::LengthMismatch When out or cache shape mismatches x.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with ARD leaf or cache dims.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        [[nodiscard]] std::expected<void, GprError> apply_cached_keeping(
            MatrixRef<T> x,
            const DistanceCache<T>& cache,
            MatrixMut<T> out,
            Triangle uplo,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested,
            NestedBuffers<T>& kept_bufs,
            std::size_t products,
            std::span<T> ard_s2_buf = {},
            std::span<std::uint8_t> ard_s2_ready = {}
        ) const;

        /// @brief Evaluates <weight_lower, dK/dtheta_p>_F for all kernel parameters p in one tree walk.
        /// @details Contracts each leaf derivative directly against the effective lower-triangular weight
        /// matrix (incorporating kept sibling factors for products) and sums column partials in ascending
        /// column order for deterministic bitwise-identical parallel reduction.
        /// @param[in] x Column-major n x d training coordinate matrix.
        /// @param[in] cache Optional precomputed DistanceCache pointer, or nullptr for uncached evaluation.
        /// @param[in] weight_lower Lower-triangular symmetric weight matrix W of shape n x n. Only the lower triangle
        /// is read and it is mirrored: each strictly-lower entry counts twice (once per symmetric position) and each
        /// diagonal entry once, so out_grads equals the Frobenius product with the full symmetric matrix.
        /// @param[out] out_grads Destination span of length num_params() receiving Frobenius inner products.
        /// @param[in,out] kernel_scratch Non-aliasing n x n scratch matrix buffer.
        /// @param[in,out] nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @param[in,out] weighted_bufs Pre-allocated NestedBuffers holding kept Grams plus walk scratch
        /// (depth >= weighted_buffers(products) and shape >= n x n).
        /// @param[in] products Number of top-level product terms pre-populated in weighted_bufs[0..kept-1].
        /// @param[in,out] col_partials Caller-allocated per-column reduction scratch of length >= x.rows().
        /// @param[in] ard_s2_buf Optional forward-captured ARD $s^2$ storage passed from apply_keeping.
        /// @param[in] ard_s2_ready Optional readiness bytes identifying valid slices inside ard_s2_buf.
        /// @return Empty expected on success or GprError on length mismatch, cache mismatch, or undersized buffers.
        /// @retval GprErrorCode::LengthMismatch When out_grads.size() != num_params() or matrix shapes mismatch.
        /// @retval GprErrorCode::WorkspaceTooSmall When kernel_scratch, col_partials, or weighted_bufs is undersized.
        /// @retval GprErrorCode::InvalidConfig When kernel_scratch aliases weight_lower.
        /// @retval GprErrorCode::EmptyInput When x is empty.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with ARD leaf or cache dims.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        [[nodiscard]] std::expected<void, GprError> weighted_grads(
            MatrixRef<T> x,
            const DistanceCache<T>* cache,
            MatrixRef<T> weight_lower,
            std::span<double> out_grads,
            MatrixMut<T> kernel_scratch,
            NestedBuffers<T>& nested,
            NestedBuffers<T>& weighted_bufs,
            std::size_t products,
            std::span<double> col_partials,
            std::span<const T> ard_s2_buf = {},
            std::span<const std::uint8_t> ard_s2_ready = {}
        ) const;

    private:
        /// @brief Pimpl state storing flattened sum/product execution nodes, leaves, and bindings.
        struct Impl;

        /// @brief Private constructor taking ownership of a compiled Impl state.
        /// @details Invoked by CompiledKernel<T>::from_spec().
        /// @param[in] impl Heap-allocated compiled execution implementation.
        explicit CompiledKernel(std::unique_ptr<Impl> impl) noexcept;

        /// @brief Owning pointer to the compiled kernel implementation.
        std::unique_ptr<Impl> impl_;
    };

    extern template class NestedBuffers<double>;
    extern template class NestedBuffers<float>;
    extern template class CompiledKernel<double>;
    extern template class CompiledKernel<float>;

}  // namespace cxxgpr
