// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/interval.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/kernel/spec.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <vector>

namespace cxxgpr {

/// Pre-allocated nested intermediate matrix buffers for zero-allocation tree evaluation.
template <KernelScalarConcept T> class NestedBuffers {
public:
    /// Constructs empty nested buffers.
    NestedBuffers() noexcept = default;

    /// Constructs depth contiguous column-major buffers of shape rows x cols.
    NestedBuffers(std::size_t depth, std::size_t rows, std::size_t cols);

    /// Resizes buffers to hold depth matrices of shape rows x cols.
    void resize(std::size_t depth, std::size_t rows, std::size_t cols);

    /// Returns the number of nested buffer levels.
    [[nodiscard]] std::size_t depth() const noexcept {
        return depth_;
    }

    /// Returns the row capacity of each buffer.
    [[nodiscard]] std::size_t rows() const noexcept {
        return rows_;
    }

    /// Returns the column capacity of each buffer.
    [[nodiscard]] std::size_t cols() const noexcept {
        return cols_;
    }

    /// Returns a mutable view into buffer level index with shape rows x cols.
    [[nodiscard]] MatrixMut<T> level(std::size_t index, std::size_t rows, std::size_t cols);

private:
    std::size_t depth_ = 0;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<T> storage_;
};

/// Flattened execution-layer kernel tree dispatching statically over built-in leaves.
template <KernelScalarConcept T> class CompiledKernel {
public:
    /// Compiles a KernelSpec into a flattened CompiledKernel<T>.
    [[nodiscard]] static CompiledKernel<T> from_spec(const KernelSpec& spec);

    /// Copy-constructs a compiled kernel with independent mutable leaf state.
    CompiledKernel(const CompiledKernel& other);
    /// Copy-assigns a compiled kernel with independent mutable leaf state.
    CompiledKernel& operator=(const CompiledKernel& other);
    /// Move-constructs a compiled kernel.
    CompiledKernel(CompiledKernel&& other) noexcept = default;
    /// Move-assigns a compiled kernel.
    CompiledKernel& operator=(CompiledKernel&& other) noexcept = default;
    /// Destroys the compiled kernel.
    ~CompiledKernel();

    /// Returns the total number of log-space hyperparameters.
    [[nodiscard]] std::size_t num_params() const noexcept;

    /// Returns the number of leaf nodes in depth-first left-to-right order.
    [[nodiscard]] std::size_t num_leaves() const noexcept;

    /// Returns the input coordinate mode required by the compiled tree.
    [[nodiscard]] CoordMode coord_mode() const noexcept;

    /// Returns the number of nested intermediate buffers required for tree evaluation.
    [[nodiscard]] std::size_t nested_depth() const noexcept;

    /// Allocates NestedBuffers sized for this kernel and an output matrix of shape rows x cols.
    [[nodiscard]] NestedBuffers<T> make_nested_buffers(std::size_t rows, std::size_t cols) const;

    /// Writes flattened log-space hyperparameters into out.
    [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

    /// Atomically updates all leaf parameters from log-space params.
    [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

    /// Writes flattened open intervals into out.
    [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

    /// Returns flattened parameter-to-leaf bindings.
    [[nodiscard]] const std::vector<ParameterBinding>& parameter_bindings() const noexcept;

    /// Evaluates the training Gram matrix from squared Euclidean distances using caller buffers.
    [[nodiscard]] std::expected<void, GprError> apply(MatrixRef<T> dist, MatrixMut<T> out,
                                                      Triangle uplo, MatrixMut<T> scratch,
                                                      NestedBuffers<T>& nested) const;

    /// Evaluates the training Gram matrix from squared Euclidean distances.
    [[nodiscard]] std::expected<void, GprError> apply(MatrixRef<T> dist, MatrixMut<T> out,
                                                      Triangle uplo, MatrixMut<T> scratch) const;

    /// Evaluates the training Gram matrix from input point coordinates using caller buffers.
    [[nodiscard]] std::expected<void, GprError> apply_points(MatrixRef<T> x, MatrixMut<T> out,
                                                             Triangle uplo, MatrixMut<T> scratch,
                                                             NestedBuffers<T>& nested) const;

    /// Evaluates the training Gram matrix from input point coordinates.
    [[nodiscard]] std::expected<void, GprError>
    apply_points(MatrixRef<T> x, MatrixMut<T> out, Triangle uplo, MatrixMut<T> scratch) const;

    /// Evaluates the cross-covariance matrix from rectangular squared distances using caller
    /// buffers.
    [[nodiscard]] std::expected<void, GprError> apply_cross(MatrixRef<T> dist, MatrixMut<T> out,
                                                            MatrixMut<T> scratch,
                                                            NestedBuffers<T>& nested) const;

    /// Evaluates the cross-covariance matrix from rectangular squared distances.
    [[nodiscard]] std::expected<void, GprError> apply_cross(MatrixRef<T> dist, MatrixMut<T> out,
                                                            MatrixMut<T> scratch) const;

    /// Evaluates the cross-covariance matrix from point coordinates x1 and x2 using caller buffers.
    [[nodiscard]] std::expected<void, GprError> apply_cross_points(MatrixRef<T> x1, MatrixRef<T> x2,
                                                                   MatrixMut<T> out,
                                                                   MatrixMut<T> scratch,
                                                                   NestedBuffers<T>& nested) const;

    /// Evaluates the cross-covariance matrix from point coordinates x1 and x2.
    [[nodiscard]] std::expected<void, GprError> apply_cross_points(MatrixRef<T> x1, MatrixRef<T> x2,
                                                                   MatrixMut<T> out,
                                                                   MatrixMut<T> scratch) const;

    /// Fills the diagonal k(x_i, x_i) for points x (or constant/stationary kernels of length n).
    [[nodiscard]] std::expected<void, GprError> fill_diag(MatrixRef<T> x, std::span<T> out) const;

    /// Evaluates the first derivative wrt parameter param_idx from squared distances.
    [[nodiscard]] std::expected<void, GprError> grad(MatrixRef<T> dist, MatrixMut<T> d_k,
                                                     std::size_t param_idx, Triangle uplo,
                                                     MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                                                     NestedBuffers<T>& nested) const;

    /// Evaluates the first derivative wrt parameter param_idx from squared distances.
    [[nodiscard]] std::expected<void, GprError> grad(MatrixRef<T> dist, MatrixMut<T> d_k,
                                                     std::size_t param_idx, Triangle uplo,
                                                     MatrixMut<T> scratch_a,
                                                     MatrixMut<T> scratch_b) const;

    /// Evaluates the first derivative wrt parameter param_idx from point coordinates.
    [[nodiscard]] std::expected<void, GprError>
    grad_points(MatrixRef<T> x, MatrixMut<T> d_k, std::size_t param_idx, Triangle uplo,
                MatrixMut<T> scratch_a, MatrixMut<T> scratch_b, NestedBuffers<T>& nested) const;

    /// Evaluates the first derivative wrt parameter param_idx from point coordinates.
    [[nodiscard]] std::expected<void, GprError> grad_points(MatrixRef<T> x, MatrixMut<T> d_k,
                                                            std::size_t param_idx, Triangle uplo,
                                                            MatrixMut<T> scratch_a,
                                                            MatrixMut<T> scratch_b) const;

    /// Evaluates the second derivative wrt parameters (i, j) from squared distances.
    [[nodiscard]] std::expected<void, GprError> hess(MatrixRef<T> dist, MatrixMut<T> d2_k,
                                                     std::size_t i, std::size_t j, Triangle uplo,
                                                     MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                                                     MatrixMut<T> scratch_c,
                                                     NestedBuffers<T>& nested) const;

    /// Evaluates the second derivative wrt parameters (i, j) from squared distances.
    [[nodiscard]] std::expected<void, GprError> hess(MatrixRef<T> dist, MatrixMut<T> d2_k,
                                                     std::size_t i, std::size_t j, Triangle uplo,
                                                     MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
                                                     MatrixMut<T> scratch_c) const;

    /// Evaluates the second derivative wrt parameters (i, j) from point coordinates.
    [[nodiscard]] std::expected<void, GprError>
    hess_points(MatrixRef<T> x, MatrixMut<T> d2_k, std::size_t i, std::size_t j, Triangle uplo,
                MatrixMut<T> scratch_a, MatrixMut<T> scratch_b, MatrixMut<T> scratch_c,
                NestedBuffers<T>& nested) const;

    /// Evaluates the second derivative wrt parameters (i, j) from point coordinates.
    [[nodiscard]] std::expected<void, GprError>
    hess_points(MatrixRef<T> x, MatrixMut<T> d2_k, std::size_t i, std::size_t j, Triangle uplo,
                MatrixMut<T> scratch_a, MatrixMut<T> scratch_b, MatrixMut<T> scratch_c) const;

private:
    struct Impl;

    explicit CompiledKernel(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

extern template class NestedBuffers<double>;
extern template class NestedBuffers<float>;
extern template class CompiledKernel<double>;
extern template class CompiledKernel<float>;

} // namespace cxxgpr
