// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/interval.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/kernel/compiled.hpp"
#include "cxxgpr/kernel/leaves.hpp"
#include "cxxgpr/kernel/spec.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace cxxgpr {

/// Tag marking a distance kernel that reads only supplied squared distances (plus Constant/White).
struct DistanceOnly {};

/// Tag marking a distance kernel that may mix supplied squared distances with coordinate leaves.
struct WithPoints {};

/// Constrains mode tags valid for DistanceKernel.
template <typename C>
concept DistanceModeConcept = std::same_as<C, DistanceOnly> || std::same_as<C, WithPoints>;

/// Process-unique identifier of a supplied squared-distance slot.
class SlotId {
public:
    /// Returns the numeric identifier value.
    [[nodiscard]] constexpr std::uint64_t value() const noexcept {
        return value_;
    }

    /// Compares two slot identifiers for equality.
    [[nodiscard]] friend constexpr bool operator==(const SlotId& lhs,
                                                   const SlotId& rhs) noexcept = default;

    /// Three-way comparison between two slot identifiers.
    [[nodiscard]] friend constexpr auto operator<=>(const SlotId& lhs,
                                                    const SlotId& rhs) noexcept = default;

private:
    friend class ScalarDistance;
    friend class ArdDistance;

    explicit constexpr SlotId(std::uint64_t value) noexcept : value_(value) {}

    std::uint64_t value_ = 0;
};

/// Describes one distance slot registered in a DistanceKernel in depth-first first-appearance
/// order.
struct SlotDescriptor {
    /// Unique slot identifier.
    SlotId id;
    /// Feature dimensionality for ARD slots, or std::nullopt for scalar slots.
    std::optional<std::size_t> ard_dims;
    /// Zero-based index among slots of the same shape (scalar or ARD).
    std::size_t at = 0;

    /// Checks value equality.
    [[nodiscard]] friend constexpr bool operator==(const SlotDescriptor& lhs,
                                                   const SlotDescriptor& rhs) noexcept = default;
};

/// Declaration-layer distance kernel expression tree parameterized by Mode.
template <DistanceModeConcept Mode> class DistanceKernel;

/// Execution-layer distance kernel parameterized by scalar type T.
template <KernelScalarConcept T> class CompiledDistanceKernel;

/// Supplied isotropic squared-distance slot binding isotropic built-in kernel leaves.
class ScalarDistance {
public:
    /// Creates a new scalar distance slot with a process-unique SlotId.
    ScalarDistance();

    /// Returns the slot identifier.
    [[nodiscard]] constexpr SlotId slot() const noexcept {
        return id_;
    }

    /// Binds an isotropic RBF leaf to this scalar distance slot.
    [[nodiscard]] DistanceKernel<DistanceOnly> rbf(RbfKernel leaf) const;

    /// Binds an isotropic Matern leaf to this scalar distance slot.
    [[nodiscard]] DistanceKernel<DistanceOnly> matern(MaternKernel leaf) const;

    /// Binds an isotropic Periodic leaf to this scalar distance slot.
    [[nodiscard]] DistanceKernel<DistanceOnly> periodic(PeriodicKernel leaf) const;

    /// Binds an isotropic RationalQuadratic leaf to this scalar distance slot.
    [[nodiscard]] DistanceKernel<DistanceOnly>
    rational_quadratic(RationalQuadraticKernel leaf) const;

    /// Binds a user-defined CustomKernel leaf to this scalar distance slot.
    [[nodiscard]] DistanceKernel<DistanceOnly> custom(CustomKernel leaf) const;

private:
    SlotId id_;
};

/// Supplied ARD per-dimension squared-distance slot binding ARD built-in kernel leaves.
class ArdDistance {
public:
    /// Creates an ARD distance slot for dims > 0 dimensions.
    [[nodiscard]] static std::expected<ArdDistance, GprError> create(std::size_t dims);

    /// Returns the slot identifier.
    [[nodiscard]] constexpr SlotId slot() const noexcept {
        return id_;
    }

    /// Returns the expected feature dimensionality of this slot.
    [[nodiscard]] constexpr std::size_t dims() const noexcept {
        return dims_;
    }

    /// Binds an ARD RBF leaf whose dimensionality matches dims().
    [[nodiscard]] std::expected<DistanceKernel<DistanceOnly>, GprError>
    rbf_ard(RbfArdKernel leaf) const;

    /// Binds an ARD Matern leaf whose dimensionality matches dims().
    [[nodiscard]] std::expected<DistanceKernel<DistanceOnly>, GprError>
    matern_ard(MaternArdKernel leaf) const;

    /// Binds an ARD RationalQuadratic leaf whose dimensionality matches dims().
    [[nodiscard]] std::expected<DistanceKernel<DistanceOnly>, GprError>
    rational_quadratic_ard(RationalQuadraticArdKernel leaf) const;

private:
    ArdDistance(SlotId id, std::size_t dims) noexcept : id_(id), dims_(dims) {}

    SlotId id_;
    std::size_t dims_ = 0;
};

/// Declaration-layer kernel expression supporting supplied squared-distance slots.
template <DistanceModeConcept Mode = DistanceOnly> class DistanceKernel {
public:
    /// Copy-constructs a DistanceKernel.
    DistanceKernel(const DistanceKernel&) = default;
    /// Copy-assigns a DistanceKernel.
    DistanceKernel& operator=(const DistanceKernel&) = default;
    /// Move-constructs a DistanceKernel.
    DistanceKernel(DistanceKernel&&) noexcept = default;
    /// Move-assigns a DistanceKernel.
    DistanceKernel& operator=(DistanceKernel&&) noexcept = default;
    /// Destroys a DistanceKernel.
    ~DistanceKernel() = default;

    /// Constructs a DistanceKernel from a ConstantKernel leaf.
    /* implicit */ DistanceKernel(ConstantKernel leaf);

    /// Constructs a DistanceKernel from a WhiteKernel leaf.
    /* implicit */ DistanceKernel(WhiteKernel leaf);

    /// Promotes a DistanceKernel<DistanceOnly> into DistanceKernel<WithPoints>.
    /* implicit */ DistanceKernel(const DistanceKernel<DistanceOnly>& other)
        requires std::same_as<Mode, WithPoints>;

    /// Constructs a DistanceKernel<WithPoints> from a coordinate KernelSpec.
    /* implicit */ DistanceKernel(const KernelSpec& coord_spec)
        requires std::same_as<Mode, WithPoints>;

    /// Promotes this kernel to WithPoints mode.
    [[nodiscard]] DistanceKernel<WithPoints> with_points() const;

    /// Returns the total number of log-space hyperparameters in the tree.
    [[nodiscard]] std::size_t num_params() const noexcept;

    /// Writes flattened log-space hyperparameters in depth-first left-to-right order.
    [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

    /// Atomically updates all leaf parameters from log-space params; leaves state unchanged on
    /// error.
    [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

    /// Writes flattened open intervals in depth-first left-to-right order.
    [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

    /// Returns flattened parameter-to-leaf bindings in depth-first left-to-right order.
    [[nodiscard]] std::vector<ParameterBinding> parameter_bindings() const;

    /// Returns unique distance slots referenced by this kernel in depth-first first-appearance
    /// order.
    [[nodiscard]] std::vector<SlotDescriptor> slots() const;

    /// Returns the number of scalar distance slots referenced by this kernel.
    [[nodiscard]] std::size_t num_scalar_slots() const;

    /// Returns the number of ARD distance slots referenced by this kernel.
    [[nodiscard]] std::size_t num_ard_slots() const;

    /// Compiles this specification into an execution-layer CompiledDistanceKernel<T>.
    template <KernelScalarConcept T> [[nodiscard]] CompiledDistanceKernel<T> compile() const;

private:
    struct InternalNode;

    explicit DistanceKernel(std::shared_ptr<const InternalNode> root) noexcept
        : root_(std::move(root)) {}

    std::shared_ptr<const InternalNode> root_;

    friend class ScalarDistance;
    friend class ArdDistance;
    template <DistanceModeConcept OtherMode> friend class DistanceKernel;
    template <KernelScalarConcept T> friend class CompiledDistanceKernel;
    friend struct DistanceKernelAccess;
};

/// Combines two DistanceOnly kernels by addition.
[[nodiscard]] DistanceKernel<DistanceOnly> operator+(DistanceKernel<DistanceOnly> lhs,
                                                     DistanceKernel<DistanceOnly> rhs);

/// Combines two DistanceOnly kernels by multiplication.
[[nodiscard]] DistanceKernel<DistanceOnly> operator*(DistanceKernel<DistanceOnly> lhs,
                                                     DistanceKernel<DistanceOnly> rhs);

/// Combines two WithPoints kernels by addition.
[[nodiscard]] DistanceKernel<WithPoints> operator+(DistanceKernel<WithPoints> lhs,
                                                   DistanceKernel<WithPoints> rhs);

/// Combines two WithPoints kernels by multiplication.
[[nodiscard]] DistanceKernel<WithPoints> operator*(DistanceKernel<WithPoints> lhs,
                                                   DistanceKernel<WithPoints> rhs);

/// Combines a DistanceKernel and a coordinate KernelSpec by addition in WithPoints mode.
[[nodiscard]] DistanceKernel<WithPoints> operator+(const DistanceKernel<DistanceOnly>& lhs,
                                                   const KernelSpec& rhs);

/// Combines a coordinate KernelSpec and a DistanceKernel by addition in WithPoints mode.
[[nodiscard]] DistanceKernel<WithPoints> operator+(const KernelSpec& lhs,
                                                   const DistanceKernel<DistanceOnly>& rhs);

/// Combines a DistanceKernel and a coordinate KernelSpec by multiplication in WithPoints mode.
[[nodiscard]] DistanceKernel<WithPoints> operator*(const DistanceKernel<DistanceOnly>& lhs,
                                                   const KernelSpec& rhs);

/// Combines a coordinate KernelSpec and a DistanceKernel by multiplication in WithPoints mode.
[[nodiscard]] DistanceKernel<WithPoints> operator*(const KernelSpec& lhs,
                                                   const DistanceKernel<DistanceOnly>& rhs);

/// Binds caller-supplied squared-distance tables to slots for a single preparation call.
class DistanceSource {
public:
    /// Binds a single scalar squared-distance table (n x n for square, n x m for cross) to slot.
    DistanceSource& bind_scalar(SlotId slot, MatrixRef<double> d2);

    /// Binds per-dimension ARD squared-distance tables (length dims) to slot.
    DistanceSource& bind_ard(SlotId slot, std::span<const MatrixRef<double>> d2_per_dim);

    /// Enables repair of minor negative/diagonal/asymmetry violations within relative tolerance.
    DistanceSource& tidy(double rel_tol);

    /// Returns the configured tidy relative tolerance, if enabled.
    [[nodiscard]] std::optional<double> tidy_tolerance() const noexcept {
        return tidy_rel_tol_;
    }

private:
    struct ScalarEntry {
        SlotId slot;
        MatrixRef<double> d2;
    };

    struct ArdEntry {
        SlotId slot;
        std::vector<MatrixRef<double>> d2_per_dim;
    };

    std::vector<ScalarEntry> scalars_;
    std::vector<ArdEntry> ards_;
    std::optional<double> tidy_rel_tol_;

    friend struct DistanceStoreBuilder;
};

/// Validated (and optionally repaired) supplied squared-distance store ready for zero-alloc eval.
template <KernelScalarConcept T> class CheckedDistanceStore {
public:
    /// Copy-constructs a checked distance store.
    CheckedDistanceStore(const CheckedDistanceStore& other);
    /// Copy-assigns a checked distance store.
    CheckedDistanceStore& operator=(const CheckedDistanceStore& other);
    /// Move-constructs a checked distance store.
    CheckedDistanceStore(CheckedDistanceStore&& other) noexcept = default;
    /// Move-assigns a checked distance store.
    CheckedDistanceStore& operator=(CheckedDistanceStore&& other) noexcept = default;
    /// Destroys the checked distance store.
    ~CheckedDistanceStore();

    /// Returns the row count of each distance block (number of training or query rows).
    [[nodiscard]] std::size_t rows() const noexcept;

    /// Returns the column count of each distance block (number of training or target points).
    [[nodiscard]] std::size_t cols() const noexcept;

    /// Returns true if this store holds square training blocks (as opposed to cross blocks).
    [[nodiscard]] bool is_square() const noexcept;

private:
    struct Impl;

    explicit CheckedDistanceStore(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;

    template <KernelScalarConcept U> friend class CompiledDistanceKernel;
    friend struct DistanceStoreBuilder;
};

/// Flattened execution-layer kernel reading validated DistanceSource slots (and optional points).
template <KernelScalarConcept T> class CompiledDistanceKernel {
public:
    /// Copy-constructs a compiled distance kernel with independent mutable leaf state.
    CompiledDistanceKernel(const CompiledDistanceKernel& other);
    /// Copy-assigns a compiled distance kernel with independent mutable leaf state.
    CompiledDistanceKernel& operator=(const CompiledDistanceKernel& other);
    /// Move-constructs a compiled distance kernel.
    CompiledDistanceKernel(CompiledDistanceKernel&& other) noexcept = default;
    /// Move-assigns a compiled distance kernel.
    CompiledDistanceKernel& operator=(CompiledDistanceKernel&& other) noexcept = default;
    /// Destroys the compiled distance kernel.
    ~CompiledDistanceKernel();

    /// Returns the total number of log-space hyperparameters.
    [[nodiscard]] std::size_t num_params() const noexcept;

    /// Returns the number of leaf nodes in depth-first left-to-right order.
    [[nodiscard]] std::size_t num_leaves() const noexcept;

    /// Returns true if the kernel contains coordinate-dependent (non-supplied) leaves.
    [[nodiscard]] bool requires_points() const noexcept;

    /// Returns the number of nested intermediate buffers required for tree evaluation.
    [[nodiscard]] std::size_t nested_depth() const noexcept;

    /// Returns the exponential evaluation backend used during kernel evaluation (default
    /// Accurate).
    [[nodiscard]] KernelExp kernel_exp() const noexcept;

    /// Sets the exponential evaluation backend used during kernel evaluation.
    void set_kernel_exp(KernelExp mode) noexcept;

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

    /// Returns unique distance slots referenced by this kernel in depth-first order.
    [[nodiscard]] const std::vector<SlotDescriptor>& slots() const noexcept;

    /// Validates (and optionally repairs via tidy) square training distance supplies and packs ARD
    /// tables in a single pass.
    [[nodiscard]] std::expected<CheckedDistanceStore<T>, GprError>
    prepare_square(const DistanceSource& source) const;

    /// Validates (and optionally repairs via tidy) rectangular cross-distance supplies.
    [[nodiscard]] std::expected<CheckedDistanceStore<T>, GprError>
    prepare_cross(const DistanceSource& source) const;

    /// Evaluates the training Gram matrix from a validated store (and optional points).
    [[nodiscard]] std::expected<void, GprError>
    apply(const CheckedDistanceStore<T>& store, std::optional<MatrixRef<T>> x, MatrixMut<T> out,
          Triangle uplo, MatrixMut<T> scratch, NestedBuffers<T>& nested) const;

    /// Evaluates the training Gram matrix from a validated store (and optional points).
    [[nodiscard]] std::expected<void, GprError> apply(const CheckedDistanceStore<T>& store,
                                                      std::optional<MatrixRef<T>> x,
                                                      MatrixMut<T> out, Triangle uplo,
                                                      MatrixMut<T> scratch) const;

    /// Evaluates the cross-covariance matrix from a validated rectangular store.
    [[nodiscard]] std::expected<void, GprError> apply_cross(const CheckedDistanceStore<T>& store,
                                                            std::optional<MatrixRef<T>> x1,
                                                            std::optional<MatrixRef<T>> x2,
                                                            MatrixMut<T> out, MatrixMut<T> scratch,
                                                            NestedBuffers<T>& nested) const;

    /// Evaluates the cross-covariance matrix from a validated rectangular store.
    [[nodiscard]] std::expected<void, GprError>
    apply_cross(const CheckedDistanceStore<T>& store, std::optional<MatrixRef<T>> x1,
                std::optional<MatrixRef<T>> x2, MatrixMut<T> out, MatrixMut<T> scratch) const;

    /// Fills the diagonal k(x_i, x_i) for n points (passing x when coordinate leaves are present).
    [[nodiscard]] std::expected<void, GprError>
    fill_diag(std::size_t n, std::optional<MatrixRef<T>> x, std::span<T> out) const;

    /// Evaluates the first derivative wrt parameter param_idx.
    [[nodiscard]] std::expected<void, GprError>
    grad(const CheckedDistanceStore<T>& store, std::optional<MatrixRef<T>> x, MatrixMut<T> d_k,
         std::size_t param_idx, Triangle uplo, MatrixMut<T> scratch_a, MatrixMut<T> scratch_b,
         NestedBuffers<T>& nested) const;

    /// Evaluates the first derivative wrt parameter param_idx.
    [[nodiscard]] std::expected<void, GprError> grad(const CheckedDistanceStore<T>& store,
                                                     std::optional<MatrixRef<T>> x,
                                                     MatrixMut<T> d_k, std::size_t param_idx,
                                                     Triangle uplo, MatrixMut<T> scratch_a,
                                                     MatrixMut<T> scratch_b) const;

    /// Evaluates the second derivative wrt parameters (i, j).
    [[nodiscard]] std::expected<void, GprError>
    hess(const CheckedDistanceStore<T>& store, std::optional<MatrixRef<T>> x, MatrixMut<T> d2_k,
         std::size_t i, std::size_t j, Triangle uplo, MatrixMut<T> scratch_a,
         MatrixMut<T> scratch_b, MatrixMut<T> scratch_c, NestedBuffers<T>& nested) const;

    /// Evaluates the second derivative wrt parameters (i, j).
    [[nodiscard]] std::expected<void, GprError>
    hess(const CheckedDistanceStore<T>& store, std::optional<MatrixRef<T>> x, MatrixMut<T> d2_k,
         std::size_t i, std::size_t j, Triangle uplo, MatrixMut<T> scratch_a,
         MatrixMut<T> scratch_b, MatrixMut<T> scratch_c) const;

private:
    struct Impl;

    explicit CompiledDistanceKernel(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;

    template <DistanceModeConcept Mode> friend class DistanceKernel;
    friend struct DistanceStoreBuilder;
};

extern template class DistanceKernel<DistanceOnly>;
extern template class DistanceKernel<WithPoints>;
extern template class CheckedDistanceStore<double>;
extern template class CheckedDistanceStore<float>;
extern template class CompiledDistanceKernel<double>;
extern template class CompiledDistanceKernel<float>;

} // namespace cxxgpr
