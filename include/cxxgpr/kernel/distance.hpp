// SPDX-License-Identifier: Apache-2.0
/// @file distance.hpp
/// @brief Supplied squared-distance kernel expression trees, slot descriptors, and validated stores.
/// @details Declares DistanceOnly and WithPoints mode tags, SlotId and SlotDescriptor, ScalarDistance
/// and ArdDistance slot factories, DistanceKernel<Mode>, DistanceSource, CheckedDistanceStore<T>,
/// and CompiledDistanceKernel<T> for zero-allocation evaluation over caller-supplied squared-distance
/// tables with optional tidy validation/repair.
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

/// @brief Root namespace of the cxxgpr Gaussian Process Regression library.
namespace cxxgpr
{

    /// @brief Tag marking a distance kernel that reads only supplied squared distances (plus Constant/White).
    /// @details DistanceKernel<DistanceOnly> expressions require no point coordinate matrix x at
    /// evaluation time because all distance-dependent leaves read DistanceSource slots.
    struct DistanceOnly
    {
    };

    /// @brief Tag marking a distance kernel that may mix supplied squared distances with coordinate leaves.
    /// @details DistanceKernel<WithPoints> allows mixing supplied distance leaves with KernelSpec
    /// subtrees evaluated directly from point coordinates x.
    struct WithPoints
    {
    };

    /// @brief Constrains mode tags valid for DistanceKernel.
    /// @details Satisfied exclusively by DistanceOnly and WithPoints.
    /// @tparam C Candidate distance mode tag type.
    template <typename C>
    concept DistanceModeConcept = std::same_as<C, DistanceOnly> || std::same_as<C, WithPoints>;

    /// @brief Process-unique identifier of a supplied squared-distance slot.
    /// @details Generated monotonically via atomic counter by ScalarDistance and ArdDistance so
    /// supplied distance tables bind unambiguously across composed kernel trees.
    class SlotId
    {
    public:
        /// @brief Returns the numeric identifier value.
        /// @details Useful for diagnostics or associative lookup keys.
        /// @return Process-unique 64-bit slot identifier.
        [[nodiscard]] constexpr std::uint64_t value() const noexcept { return value_; }

        /// @brief Compares two slot identifiers for equality.
        /// @details Compares underlying numeric slot identifiers.
        /// @param lhs Left-hand SlotId operand.
        /// @param rhs Right-hand SlotId operand.
        /// @return True if both SlotId instances refer to the same slot.
        [[nodiscard]] friend constexpr bool operator==(const SlotId& lhs, const SlotId& rhs) noexcept = default;

        /// @brief Three-way comparison between two slot identifiers.
        /// @details Orders SlotId instances deterministically by numeric identifier value.
        /// @param lhs Left-hand SlotId operand.
        /// @param rhs Right-hand SlotId operand.
        /// @return Strong ordering result comparing numeric values.
        [[nodiscard]] friend constexpr auto operator<=>(const SlotId& lhs, const SlotId& rhs) noexcept = default;

    private:
        friend class ScalarDistance;
        friend class ArdDistance;

        /// @brief Constructs a SlotId wrapping an assigned unique integer value.
        /// @details Private so callers can obtain SlotId values only through ScalarDistance or ArdDistance.
        /// @param value Monotonically assigned process-unique identifier.
        explicit constexpr SlotId(std::uint64_t value) noexcept : value_(value) {}

        /// @brief Monotonic process-unique numeric identifier.
        std::uint64_t value_ = 0;
    };

    /// @brief Describes one distance slot registered in a DistanceKernel in depth-first first-appearance order.
    /// @details Records the unique SlotId, optional ARD feature count, and zero-based table index `at`
    /// partitioned separately across scalar slots (ard_dims == std::nullopt) and ARD slots.
    struct SlotDescriptor
    {
        /// @brief Unique slot identifier matching the ScalarDistance or ArdDistance handle.
        SlotId id;
        /// @brief Feature dimensionality for ARD slots, or std::nullopt for scalar slots.
        std::optional<std::size_t> ard_dims;
        /// @brief Zero-based index among slots of the same shape (scalar or ARD).
        std::size_t at = 0;

        /// @brief Checks value equality between two SlotDescriptor instances.
        /// @details Compares id, ard_dims, and shape-local index at.
        /// @param lhs Left-hand SlotDescriptor operand.
        /// @param rhs Right-hand SlotDescriptor operand.
        /// @return True if all fields match.
        [[nodiscard]] friend constexpr bool operator==(
            const SlotDescriptor& lhs, const SlotDescriptor& rhs
        ) noexcept = default;
    };

    /// @brief Forward declaration of the declaration-layer distance kernel expression tree.
    /// @details Parameterized by DistanceModeConcept (DistanceOnly or WithPoints).
    /// @tparam Mode Distance kernel mode tag.
    template <DistanceModeConcept Mode> class DistanceKernel;

    /// @brief Forward declaration of the execution-layer compiled distance kernel.
    /// @details Evaluates validated CheckedDistanceStore<T> tables (and optional coordinate leaves).
    /// @tparam T Kernel evaluation scalar type (double or float).
    template <KernelScalarConcept T> class CompiledDistanceKernel;

    /// @brief Supplied isotropic squared-distance slot binding isotropic built-in or custom kernel leaves.
    /// @details Each ScalarDistance instance owns a distinct SlotId and acts as a factory for
    /// single-leaf DistanceKernel<DistanceOnly> expressions reading that scalar distance table.
    class ScalarDistance
    {
    public:
        /// @brief Creates a new scalar distance slot with a process-unique SlotId.
        /// @details Atomically allocates a fresh identifier from the global slot counter.
        ScalarDistance();

        /// @brief Returns the slot identifier.
        /// @details Used with DistanceSource::bind_scalar() to supply the n x n or n x m matrix for this slot.
        /// @return Process-unique SlotId of this scalar distance slot.
        [[nodiscard]] constexpr SlotId slot() const noexcept { return id_; }

        /// @brief Binds an isotropic RBF leaf to this scalar distance slot.
        /// @details Evaluates exp(-d^2 / (2 * l^2)) over the supplied squared distances d^2.
        /// @param leaf Configured RbfKernel leaf.
        /// @return Single-leaf DistanceKernel<DistanceOnly> reading this slot.
        [[nodiscard]] DistanceKernel<DistanceOnly> rbf(RbfKernel leaf) const;

        /// @brief Binds an isotropic Matern leaf to this scalar distance slot.
        /// @details Evaluates Matern(nu) over sqrt(d^2) from the supplied squared distances.
        /// @param leaf Configured MaternKernel leaf.
        /// @return Single-leaf DistanceKernel<DistanceOnly> reading this slot.
        [[nodiscard]] DistanceKernel<DistanceOnly> matern(MaternKernel leaf) const;

        /// @brief Binds an isotropic Periodic leaf to this scalar distance slot.
        /// @details Evaluates exp(-2 * sin^2(pi * sqrt(d^2) / p) / l^2) over the supplied squared distances.
        /// @param leaf Configured PeriodicKernel leaf.
        /// @return Single-leaf DistanceKernel<DistanceOnly> reading this slot.
        [[nodiscard]] DistanceKernel<DistanceOnly> periodic(PeriodicKernel leaf) const;

        /// @brief Binds an isotropic RationalQuadratic leaf to this scalar distance slot.
        /// @details Evaluates (1 + d^2 / (2 * alpha * l^2))^(-alpha) over the supplied squared distances.
        /// @param leaf Configured RationalQuadraticKernel leaf.
        /// @return Single-leaf DistanceKernel<DistanceOnly> reading this slot.
        [[nodiscard]] DistanceKernel<DistanceOnly> rational_quadratic(RationalQuadraticKernel leaf) const;

        /// @brief Binds a user-defined CustomKernel leaf to this scalar distance slot.
        /// @details Feeds the supplied squared-distance block directly into CustomKernel::term<T>().
        /// @param leaf Validated CustomKernel wrapper.
        /// @return Single-leaf DistanceKernel<DistanceOnly> reading this slot.
        [[nodiscard]] DistanceKernel<DistanceOnly> custom(CustomKernel leaf) const;

    private:
        /// @brief Assigned process-unique slot identifier.
        SlotId id_;
    };

    /// @brief Supplied ARD per-dimension squared-distance slot binding ARD built-in kernel leaves.
    /// @details Enforces that any bound ARD leaf has dimensionality matching dims().
    class ArdDistance
    {
    public:
        /// @brief Creates an ARD distance slot for dims > 0 dimensions.
        /// @details Rejects dims == 0 with EmptyInput and assigns a fresh process-unique SlotId.
        /// @param dims Expected number of per-dimension squared-distance tables.
        /// @return Validated ArdDistance handle or GprError if dims == 0.
        /// @retval GprErrorCode::EmptyInput When dims == 0.
        [[nodiscard]] static std::expected<ArdDistance, GprError> create(std::size_t dims);

        /// @brief Returns the slot identifier.
        /// @details Used with DistanceSource::bind_ard() to supply dims per-dimension distance tables.
        /// @return Process-unique SlotId of this ARD distance slot.
        [[nodiscard]] constexpr SlotId slot() const noexcept { return id_; }

        /// @brief Returns the expected feature dimensionality of this slot.
        /// @details All bound ARD leaves and supplied DistanceSource arrays must match this count.
        /// @return Positive feature dimension count dims_.
        [[nodiscard]] constexpr std::size_t dims() const noexcept { return dims_; }

        /// @brief Binds an ARD RBF leaf whose dimensionality matches dims().
        /// @details Rejects leaves whose lengthscale count differs from dims().
        /// @param leaf Configured RbfArdKernel leaf.
        /// @return Single-leaf DistanceKernel<DistanceOnly> or GprError on dimension mismatch.
        /// @retval GprErrorCode::DimensionMismatch When leaf.num_params() != dims().
        [[nodiscard]] std::expected<DistanceKernel<DistanceOnly>, GprError> rbf_ard(RbfArdKernel leaf) const;

        /// @brief Binds an ARD Matern leaf whose dimensionality matches dims().
        /// @details Rejects leaves whose lengthscale count differs from dims().
        /// @param leaf Configured MaternArdKernel leaf.
        /// @return Single-leaf DistanceKernel<DistanceOnly> or GprError on dimension mismatch.
        /// @retval GprErrorCode::DimensionMismatch When leaf.num_params() != dims().
        [[nodiscard]] std::expected<DistanceKernel<DistanceOnly>, GprError> matern_ard(MaternArdKernel leaf) const;

        /// @brief Binds an ARD RationalQuadratic leaf whose dimensionality matches dims().
        /// @details Rejects leaves whose lengthscale count differs from dims().
        /// @param leaf Configured RationalQuadraticArdKernel leaf.
        /// @return Single-leaf DistanceKernel<DistanceOnly> or GprError on dimension mismatch.
        /// @retval GprErrorCode::DimensionMismatch When leaf.lengthscales().size() != dims().
        [[nodiscard]] std::expected<DistanceKernel<DistanceOnly>, GprError> rational_quadratic_ard(
            RationalQuadraticArdKernel leaf
        ) const;

    private:
        /// @brief Constructs an ArdDistance handle from a validated SlotId and dimension count.
        /// @details Invoked by ArdDistance::create() after verifying dims > 0.
        /// @param id Assigned process-unique SlotId.
        /// @param dims Positive feature dimensionality.
        ArdDistance(SlotId id, std::size_t dims) noexcept : id_(id), dims_(dims) {}

        /// @brief Assigned process-unique slot identifier.
        SlotId id_;
        /// @brief Required number of per-dimension squared-distance blocks.
        std::size_t dims_ = 0;
    };

    /// @brief Declaration-layer kernel expression supporting supplied squared-distance slots.
    /// @details Composes supplied-distance leaves, ConstantKernel, WhiteKernel, and (in WithPoints mode)
    /// coordinate-evaluated KernelSpec subtrees with atomic log-space updates and compilation into
    /// CompiledDistanceKernel<T>.
    /// @tparam Mode Distance mode tag (DistanceOnly or WithPoints).
    template <DistanceModeConcept Mode = DistanceOnly> class DistanceKernel
    {
    public:
        /// @brief Copy-constructs a DistanceKernel.
        /// @details Shares the underlying immutable AST root node.
        DistanceKernel(const DistanceKernel&) = default;

        /// @brief Copy-assigns a DistanceKernel.
        /// @details Shares the underlying immutable AST root node.
        /// @return Reference to this DistanceKernel.
        DistanceKernel& operator=(const DistanceKernel&) = default;

        /// @brief Move-constructs a DistanceKernel.
        /// @details Transfers ownership of the root node pointer.
        DistanceKernel(DistanceKernel&&) noexcept = default;

        /// @brief Move-assigns a DistanceKernel.
        /// @details Transfers ownership of the root node pointer.
        /// @return Reference to this DistanceKernel.
        DistanceKernel& operator=(DistanceKernel&&) noexcept = default;

        /// @brief Destroys a DistanceKernel.
        /// @details Releases the shared AST node reference.
        ~DistanceKernel() = default;

        /// @brief Constructs a DistanceKernel from a ConstantKernel leaf.
        /// @details Available in both DistanceOnly and WithPoints modes because ConstantKernel requires
        /// neither supplied distances nor coordinates.
        /// @param leaf ConstantKernel amplitude leaf.
        /* implicit */ DistanceKernel(ConstantKernel leaf);

        /// @brief Constructs a DistanceKernel from a WhiteKernel leaf.
        /// @details Available in both DistanceOnly and WithPoints modes because WhiteKernel applies
        /// only to the square training diagonal.
        /// @param leaf WhiteKernel noise leaf.
        /* implicit */ DistanceKernel(WhiteKernel leaf);

        /// @brief Promotes a DistanceKernel<DistanceOnly> into DistanceKernel<WithPoints>.
        /// @details Deeply converts internal AST nodes to the WithPoints specialization.
        /// @param other Source DistanceOnly distance kernel expression.
        /* implicit */ DistanceKernel(const DistanceKernel<DistanceOnly>& other)
            requires std::same_as<Mode, WithPoints>;

        /// @brief Constructs a DistanceKernel<WithPoints> from a coordinate KernelSpec.
        /// @details Wraps all leaves in coord_spec as coordinate-evaluated subtrees.
        /// @param coord_spec Coordinate-evaluated kernel expression tree.
        /* implicit */ DistanceKernel(const KernelSpec& coord_spec)
            requires std::same_as<Mode, WithPoints>;

        /// @brief Promotes this kernel to WithPoints mode.
        /// @details Returns a copy if already in WithPoints mode, or promotes from DistanceOnly.
        /// @return Equivalent kernel expression typed as DistanceKernel<WithPoints>.
        [[nodiscard]] DistanceKernel<WithPoints> with_points() const;

        /// @brief Returns the total number of log-space hyperparameters in the tree.
        /// @details Sums parameter counts across all supplied-distance and coordinate leaves.
        /// @return Total number of hyperparameters.
        [[nodiscard]] std::size_t num_params() const noexcept;

        /// @brief Writes flattened log-space hyperparameters in depth-first left-to-right order.
        /// @details Validates out.size() == num_params() before traversing the tree.
        /// @param out Destination span of length num_params().
        /// @return Empty expected on success or GprError on length mismatch or leaf error.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Atomically updates all leaf parameters from log-space params.
        /// @details Rebuilds a validated immutable AST and swaps root_ only when every leaf succeeds,
        /// leaving *this unchanged on failure.
        /// @param params Input span of unconstrained log-space parameters of length num_params().
        /// @return Empty expected on success or GprError on count mismatch, non-finite value, or bound error.
        /// @retval GprErrorCode::LengthMismatch When params.size() != num_params().
        /// @retval GprErrorCode::NonFiniteInput When any log-space parameter is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When any exponentiated value falls outside leaf bounds.
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes flattened open intervals in depth-first left-to-right order.
        /// @details Order matches get_params() and set_params().
        /// @param out Destination span of length num_params().
        /// @return Empty expected on success or GprError on length mismatch or leaf error.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Returns flattened parameter-to-leaf bindings in depth-first left-to-right order.
        /// @details Maps each flat parameter index to its depth-first leaf ID and leaf-local slot.
        /// @return Vector of ParameterBinding descriptors of length num_params().
        [[nodiscard]] std::vector<ParameterBinding> parameter_bindings() const;

        /// @brief Returns unique distance slots referenced by this kernel in depth-first first-appearance order.
        /// @details Deduplicates repeated references to the same SlotId while assigning separate zero-based
        /// indices `at` across scalar and ARD slots.
        /// @return Vector of unique SlotDescriptor entries.
        [[nodiscard]] std::vector<SlotDescriptor> slots() const;

        /// @brief Returns the number of scalar distance slots referenced by this kernel.
        /// @details Counts entries in slots() where ard_dims is std::nullopt.
        /// @return Number of unique scalar distance slots.
        [[nodiscard]] std::size_t num_scalar_slots() const;

        /// @brief Returns the number of ARD distance slots referenced by this kernel.
        /// @details Counts entries in slots() where ard_dims has a value.
        /// @return Number of unique ARD distance slots.
        [[nodiscard]] std::size_t num_ard_slots() const;

        /// @brief Compiles this specification into an execution-layer CompiledDistanceKernel<T>.
        /// @details Flattens sums and products associatively, caches ARD lengthscales, and records unique
        /// SlotDescriptor mappings.
        /// @tparam T Kernel evaluation scalar type (double or float).
        /// @return Compiled execution-layer CompiledDistanceKernel<T>.
        template <KernelScalarConcept T> [[nodiscard]] CompiledDistanceKernel<T> compile() const;

    private:
        /// @brief Internal immutable AST node variant (supplied scalar, supplied ARD, coord leaf, sum, product).
        struct InternalNode;

        /// @brief Private constructor initializing DistanceKernel from a shared immutable root node.
        /// @details Used internally by factories, composition operators, and promotion constructors.
        /// @param root Shared immutable root AST node.
        explicit DistanceKernel(std::shared_ptr<const InternalNode> root) noexcept : root_(std::move(root)) {}

        /// @brief Shared immutable root node of the distance kernel AST.
        std::shared_ptr<const InternalNode> root_;

        friend class ScalarDistance;
        friend class ArdDistance;
        template <DistanceModeConcept OtherMode> friend class DistanceKernel;
        template <KernelScalarConcept T> friend class CompiledDistanceKernel;
        friend struct DistanceKernelAccess;
    };

    /// @brief Combines two DistanceOnly kernels by addition.
    /// @details Produces a composite sum node k_lhs + k_rhs in DistanceOnly mode.
    /// @param lhs Left DistanceOnly addend.
    /// @param rhs Right DistanceOnly addend.
    /// @return Composite DistanceKernel<DistanceOnly>.
    [[nodiscard]] DistanceKernel<DistanceOnly> operator+(
        DistanceKernel<DistanceOnly> lhs, DistanceKernel<DistanceOnly> rhs
    );

    /// @brief Combines two DistanceOnly kernels by multiplication.
    /// @details Produces a composite product node k_lhs * k_rhs in DistanceOnly mode.
    /// @param lhs Left DistanceOnly factor.
    /// @param rhs Right DistanceOnly factor.
    /// @return Composite DistanceKernel<DistanceOnly>.
    [[nodiscard]] DistanceKernel<DistanceOnly> operator*(
        DistanceKernel<DistanceOnly> lhs, DistanceKernel<DistanceOnly> rhs
    );

    /// @brief Combines two WithPoints kernels by addition.
    /// @details Produces a composite sum node k_lhs + k_rhs in WithPoints mode.
    /// @param lhs Left WithPoints addend.
    /// @param rhs Right WithPoints addend.
    /// @return Composite DistanceKernel<WithPoints>.
    [[nodiscard]] DistanceKernel<WithPoints> operator+(DistanceKernel<WithPoints> lhs, DistanceKernel<WithPoints> rhs);

    /// @brief Combines two WithPoints kernels by multiplication.
    /// @details Produces a composite product node k_lhs * k_rhs in WithPoints mode.
    /// @param lhs Left WithPoints factor.
    /// @param rhs Right WithPoints factor.
    /// @return Composite DistanceKernel<WithPoints>.
    [[nodiscard]] DistanceKernel<WithPoints> operator*(DistanceKernel<WithPoints> lhs, DistanceKernel<WithPoints> rhs);

    /// @brief Combines a DistanceKernel and a coordinate KernelSpec by addition in WithPoints mode.
    /// @details Promotes lhs and converts rhs into WithPoints subtrees before summing.
    /// @param lhs Supplied-distance kernel addend.
    /// @param rhs Coordinate-evaluated KernelSpec addend.
    /// @return Composite DistanceKernel<WithPoints>.
    [[nodiscard]] DistanceKernel<WithPoints> operator+(const DistanceKernel<DistanceOnly>& lhs, const KernelSpec& rhs);

    /// @brief Combines a coordinate KernelSpec and a DistanceKernel by addition in WithPoints mode.
    /// @details Converts lhs and promotes rhs into WithPoints subtrees before summing.
    /// @param lhs Coordinate-evaluated KernelSpec addend.
    /// @param rhs Supplied-distance kernel addend.
    /// @return Composite DistanceKernel<WithPoints>.
    [[nodiscard]] DistanceKernel<WithPoints> operator+(const KernelSpec& lhs, const DistanceKernel<DistanceOnly>& rhs);

    /// @brief Combines a DistanceKernel and a coordinate KernelSpec by multiplication in WithPoints mode.
    /// @details Promotes lhs and converts rhs into WithPoints subtrees before multiplying.
    /// @param lhs Supplied-distance kernel factor.
    /// @param rhs Coordinate-evaluated KernelSpec factor.
    /// @return Composite DistanceKernel<WithPoints>.
    [[nodiscard]] DistanceKernel<WithPoints> operator*(const DistanceKernel<DistanceOnly>& lhs, const KernelSpec& rhs);

    /// @brief Combines a coordinate KernelSpec and a DistanceKernel by multiplication in WithPoints mode.
    /// @details Converts lhs and promotes rhs into WithPoints subtrees before multiplying.
    /// @param lhs Coordinate-evaluated KernelSpec factor.
    /// @param rhs Supplied-distance kernel factor.
    /// @return Composite DistanceKernel<WithPoints>.
    [[nodiscard]] DistanceKernel<WithPoints> operator*(const KernelSpec& lhs, const DistanceKernel<DistanceOnly>& rhs);

    /// @brief Binds caller-supplied squared-distance tables to slots for a single preparation call.
    /// @details Collects non-owning double-precision MatrixRef<double> views paired with SlotIds plus an
    /// optional tidy relative tolerance prior to CheckedDistanceStore validation and packing.
    class DistanceSource
    {
    public:
        /// @brief Binds a single scalar squared-distance table (n x n for square, n x m for cross) to slot.
        /// @details Appends a scalar slot binding validated later during prepare_square() or prepare_cross().
        /// @param slot Target scalar SlotId obtained from ScalarDistance::slot().
        /// @param d2 Column-major squared-distance matrix view in double precision.
        /// @return Reference to *this for fluent chaining.
        DistanceSource& bind_scalar(SlotId slot, MatrixRef<double> d2);

        /// @brief Binds per-dimension ARD squared-distance tables (length dims) to slot.
        /// @details Copies the span of MatrixRef<double> views into internal entry storage.
        /// @param slot Target ARD SlotId obtained from ArdDistance::slot().
        /// @param d2_per_dim Span of column-major squared-distance matrices, one per ARD dimension.
        /// @return Reference to *this for fluent chaining.
        DistanceSource& bind_ard(SlotId slot, std::span<const MatrixRef<double>> d2_per_dim);

        /// @brief Enables repair of minor negative/diagonal/asymmetry violations within relative tolerance.
        /// @details Permits negative entries >= -rel_tol * max_abs, diagonal |d2(i,i)| <= rel_tol * max_abs,
        /// and asymmetry |d2(i,j) - d2(j,i)| <= rel_tol * max_abs, clamping negative values to 0, zeroing
        /// square diagonals, and symmetrizing off-diagonal pairs by averaging.
        /// @param rel_tol Non-negative finite relative tolerance multiplier against max_abs.
        /// @return Reference to *this for fluent chaining.
        DistanceSource& tidy(double rel_tol);

        /// @brief Returns the configured tidy relative tolerance, if enabled.
        /// @details Returns std::nullopt when strict exact validation is in effect.
        /// @return Configured relative tolerance or std::nullopt.
        [[nodiscard]] std::optional<double> tidy_tolerance() const noexcept { return tidy_rel_tol_; }

    private:
        /// @brief Binding entry associating one scalar slot ID with a supplied squared-distance view.
        struct ScalarEntry
        {
            /// @brief Bound scalar slot identifier.
            SlotId slot;
            /// @brief Non-owning column-major view of supplied squared distances.
            MatrixRef<double> d2;
        };

        /// @brief Binding entry associating one ARD slot ID with per-dimension squared-distance views.
        struct ArdEntry
        {
            /// @brief Bound ARD slot identifier.
            SlotId slot;
            /// @brief Vector of non-owning per-dimension squared-distance views.
            std::vector<MatrixRef<double>> d2_per_dim;
        };

        /// @brief Registered scalar slot bindings.
        std::vector<ScalarEntry> scalars_;
        /// @brief Registered ARD slot bindings.
        std::vector<ArdEntry> ards_;
        /// @brief Optional relative tolerance enabling tidy repair during store preparation.
        std::optional<double> tidy_rel_tol_;

        friend struct DistanceStoreBuilder;
    };

    /// @brief Validated (and optionally repaired) supplied squared-distance store ready for zero-alloc eval.
    /// @details Holds validated scalar distance matrices and packed lower-triangular (for square) or
    /// rectangular ARD distance tables converted to scalar T.
    /// @tparam T Kernel evaluation scalar type (double or float).
    template <KernelScalarConcept T> class CheckedDistanceStore
    {
    public:
        /// @brief Copy-constructs a checked distance store.
        /// @details Deep-copies the validated distance buffers.
        /// @param other CheckedDistanceStore instance to copy.
        CheckedDistanceStore(const CheckedDistanceStore& other);

        /// @brief Copy-assigns a checked distance store.
        /// @details Deep-copies the validated distance buffers.
        /// @param other CheckedDistanceStore instance to copy.
        /// @return Reference to this CheckedDistanceStore.
        CheckedDistanceStore& operator=(const CheckedDistanceStore& other);

        /// @brief Move-constructs a checked distance store.
        /// @details Transfers ownership of the internal validated store state.
        /// @param other CheckedDistanceStore instance to move from.
        CheckedDistanceStore(CheckedDistanceStore&& other) noexcept = default;

        /// @brief Move-assigns a checked distance store.
        /// @details Transfers ownership of the internal validated store state.
        /// @param other CheckedDistanceStore instance to move from.
        /// @return Reference to this CheckedDistanceStore.
        CheckedDistanceStore& operator=(CheckedDistanceStore&& other) noexcept = default;

        /// @brief Destroys the checked distance store.
        /// @details Frees internal distance buffers out of line where Impl is complete.
        ~CheckedDistanceStore();

        /// @brief Returns the row count of each distance block.
        /// @details Equals the number of training points for square stores or query rows for cross stores.
        /// @return Block row count.
        [[nodiscard]] std::size_t rows() const noexcept;

        /// @brief Returns the column count of each distance block.
        /// @details Equals rows() for square stores or target point count for cross stores.
        /// @return Block column count.
        [[nodiscard]] std::size_t cols() const noexcept;

        /// @brief Returns true if this store holds square training blocks (as opposed to cross blocks).
        /// @details Distinguishes stores produced by prepare_square() from prepare_cross().
        /// @return True for square training distance stores, false for rectangular cross stores.
        [[nodiscard]] bool is_square() const noexcept;

    private:
        /// @brief Pimpl state storing validated scalar blocks, packed/rect ARD blocks, and slot layout.
        struct Impl;

        /// @brief Private constructor taking ownership of a built Impl state.
        /// @details Invoked by DistanceStoreBuilder after validation succeeds.
        /// @param impl Heap-allocated validated store state.
        explicit CheckedDistanceStore(std::unique_ptr<Impl> impl) noexcept;

        /// @brief Owning pointer to the validated distance store implementation.
        std::unique_ptr<Impl> impl_;

        template <KernelScalarConcept U> friend class CompiledDistanceKernel;
        friend struct DistanceStoreBuilder;
    };

    /// @brief Flattened execution-layer kernel reading validated DistanceSource slots (and optional points).
    /// @details Compiles a DistanceKernel<Mode> into an associatively flattened execution tree supporting
    /// distance-only and mixed distance-plus-coordinate evaluation, gradients, and Hessians.
    /// @tparam T Kernel evaluation scalar type (double or float).
    template <KernelScalarConcept T> class CompiledDistanceKernel
    {
    public:
        /// @brief Copy-constructs a compiled distance kernel with independent mutable leaf state.
        /// @details Deep-copies flattened execution AST, leaves, slot descriptors, and cached lengthscales.
        /// @param other CompiledDistanceKernel instance to copy.
        CompiledDistanceKernel(const CompiledDistanceKernel& other);

        /// @brief Copy-assigns a compiled distance kernel with independent mutable leaf state.
        /// @details Deep-copies flattened execution AST and leaves when not self-assigned.
        /// @param other CompiledDistanceKernel instance to copy.
        /// @return Reference to this CompiledDistanceKernel.
        CompiledDistanceKernel& operator=(const CompiledDistanceKernel& other);

        /// @brief Move-constructs a compiled distance kernel.
        /// @details Transfers ownership of the internal compiled implementation pointer.
        /// @param other CompiledDistanceKernel instance to move from.
        CompiledDistanceKernel(CompiledDistanceKernel&& other) noexcept = default;

        /// @brief Move-assigns a compiled distance kernel.
        /// @details Transfers ownership of the internal compiled implementation pointer.
        /// @param other CompiledDistanceKernel instance to move from.
        /// @return Reference to this CompiledDistanceKernel.
        CompiledDistanceKernel& operator=(CompiledDistanceKernel&& other) noexcept = default;

        /// @brief Destroys the compiled distance kernel.
        /// @details Releases the internal Pimpl state out of line where Impl is complete.
        ~CompiledDistanceKernel();

        /// @brief Returns the total number of log-space hyperparameters.
        /// @details Equals parameter_bindings().size().
        /// @return Total number of hyperparameters across all leaves.
        [[nodiscard]] std::size_t num_params() const noexcept;

        /// @brief Returns the number of leaf nodes in depth-first left-to-right order.
        /// @details Includes supplied scalar/ARD leaves and any coordinate or Constant/White leaves.
        /// @return Number of leaves in the flattened tree.
        [[nodiscard]] std::size_t num_leaves() const noexcept;

        /// @brief Returns true if the kernel contains coordinate-dependent (non-supplied) leaves.
        /// @details When true, evaluation calls require a valid point matrix x (and x1/x2 for cross).
        /// @return True when coordinate-evaluated leaves are present.
        [[nodiscard]] bool requires_points() const noexcept;

        /// @brief Returns the number of nested intermediate buffers required for tree evaluation.
        /// @details Matches the depth passed to NestedBuffers<T> by make_nested_buffers().
        /// @return Required NestedBuffers depth.
        [[nodiscard]] std::size_t nested_depth() const noexcept;

        /// @brief Returns the exponential evaluation backend used during kernel evaluation.
        /// @details Defaults to KernelExp::Accurate upon compilation.
        /// @return Active KernelExp selector.
        [[nodiscard]] KernelExp kernel_exp() const noexcept;

        /// @brief Sets the exponential evaluation backend used during kernel evaluation.
        /// @details Selects KernelExp::Accurate or KernelExp::FastApprox for built-in exponential leaves.
        /// @param mode New exponential evaluation backend.
        void set_kernel_exp(KernelExp mode) noexcept;

        /// @brief Allocates NestedBuffers sized for this kernel and an output matrix of shape rows x cols.
        /// @details Allocates nested_depth() intermediate matrices of dimensions rows x cols.
        /// @param rows Target matrix row count.
        /// @param cols Target matrix column count.
        /// @return Pre-sized NestedBuffers<T> ready for zero-allocation evaluation calls.
        [[nodiscard]] NestedBuffers<T> make_nested_buffers(std::size_t rows, std::size_t cols) const;

        /// @brief Writes flattened log-space hyperparameters into out.
        /// @details Traverses leaves in depth-first left-to-right order.
        /// @param out Destination span of length num_params().
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Atomically updates all leaf parameters from log-space params.
        /// @details Stages all leaf updates and refreshes cached ARD lengthscales before committing;
        /// failed updates leave parameters unchanged.
        /// @param params Input span of unconstrained log-space parameters of length num_params().
        /// @return Empty expected on success or GprError on count mismatch, non-finite input, or bound error.
        /// @retval GprErrorCode::LengthMismatch When params.size() != num_params().
        /// @retval GprErrorCode::NonFiniteInput When any parameter is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When any exponentiated value exits its leaf interval.
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes flattened open intervals into out.
        /// @details Populates intervals in depth-first left-to-right parameter order.
        /// @param out Destination span of length num_params().
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Returns flattened parameter-to-leaf bindings.
        /// @details Ordered identically to get_params() and set_params().
        /// @return Const reference to the vector of ParameterBinding descriptors.
        [[nodiscard]] const std::vector<ParameterBinding>& parameter_bindings() const noexcept;

        /// @brief Returns unique distance slots referenced by this kernel in depth-first order.
        /// @details Matches the slot descriptors required by prepare_square() and prepare_cross().
        /// @return Const reference to the vector of SlotDescriptor entries.
        [[nodiscard]] const std::vector<SlotDescriptor>& slots() const noexcept;

        /// @brief Validates (and optionally repairs via tidy) square training distance supplies and packs ARD tables.
        /// @details Verifies slot coverage, shapes, finiteness, non-negativity, zero diagonals, and symmetry,
        /// packs square ARD blocks in lower-triangular order, and converts values to scalar T.
        /// @param source Bound DistanceSource tables and optional tidy relative tolerance.
        /// @return Validated CheckedDistanceStore<T> or GprError if slots, shapes, or distances are invalid.
        /// @retval GprErrorCode::InvalidConfig When tidy relative tolerance is negative or non-finite.
        /// @retval GprErrorCode::UnsupportedKernelOperation When source binds an unknown slot or kernel has no slots.
        /// @retval GprErrorCode::DistanceSlot When a slot is missing, duplicated, or mismatched (scalar vs ARD).
        /// @retval GprErrorCode::EmptyInput When any supplied matrix or ARD span is empty.
        /// @retval GprErrorCode::LengthMismatch When distance blocks are non-square or differ in shape.
        /// @retval GprErrorCode::DimensionMismatch When ARD table count differs from slot ard_dims.
        /// @retval GprErrorCode::InvalidDistance When values are non-finite, negative, asymmetric, nonzero on diag,
        /// or overflow float precision.
        /// @retval GprErrorCode::SizeOverflow When packed buffer size overflows std::size_t.
        [[nodiscard]] std::expected<CheckedDistanceStore<T>, GprError> prepare_square(
            const DistanceSource& source
        ) const;

        /// @brief Validates (and optionally repairs via tidy) rectangular cross-distance supplies.
        /// @details Verifies slot coverage, matching n1 x n2 block shapes, finiteness, and non-negativity
        /// (with optional tidy clamping), converting blocks into scalar T.
        /// @param source Bound DistanceSource tables and optional tidy relative tolerance.
        /// @return Validated rectangular CheckedDistanceStore<T> or GprError on invalid source.
        /// @retval GprErrorCode::InvalidConfig When tidy relative tolerance is negative or non-finite.
        /// @retval GprErrorCode::UnsupportedKernelOperation When source binds an unknown slot or kernel has no slots.
        /// @retval GprErrorCode::DistanceSlot When a slot is missing, duplicated, or mismatched (scalar vs ARD).
        /// @retval GprErrorCode::EmptyInput When any supplied matrix or ARD span is empty.
        /// @retval GprErrorCode::LengthMismatch When block shapes differ across bound slots.
        /// @retval GprErrorCode::DimensionMismatch When ARD table count differs from slot ard_dims.
        /// @retval GprErrorCode::InvalidDistance When values are non-finite, negative beyond tolerance,
        /// or overflow float precision.
        /// @retval GprErrorCode::SizeOverflow When buffer size overflows std::size_t.
        [[nodiscard]] std::expected<CheckedDistanceStore<T>, GprError> prepare_cross(
            const DistanceSource& source
        ) const;

        /// @brief Evaluates the training Gram matrix from a validated store (and optional points).
        /// @details Uses caller-supplied scratch and nested buffers for zero-allocation execution.
        /// @param store Validated square CheckedDistanceStore<T> matching this kernel's slots.
        /// @param x Optional n x d training coordinate matrix (required when requires_points() is true).
        /// @param out Column-major n x n destination Gram matrix.
        /// @param uplo Matrix triangle or full region to populate.
        /// @param scratch Non-aliasing n x n scratch matrix buffer.
        /// @param nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @return Empty expected on success or GprError on store, point, or buffer validation failure.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::UnsupportedKernelOperation When store slots/mode disagree or x is missing.
        /// @retval GprErrorCode::EmptyInput When store or x is empty.
        /// @retval GprErrorCode::LengthMismatch When out or x row counts disagree with store.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with a coordinate ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        [[nodiscard]] std::expected<void, GprError> apply(
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x,
            MatrixMut<T> out,
            Triangle uplo,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates the training Gram matrix from a validated store (and optional points).
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param store Validated square CheckedDistanceStore<T> matching this kernel's slots.
        /// @param x Optional n x d training coordinate matrix (required when requires_points() is true).
        /// @param out Column-major n x n destination Gram matrix.
        /// @param uplo Matrix triangle or full region to populate.
        /// @param scratch Non-aliasing n x n scratch matrix buffer.
        /// @return Empty expected on success or GprError on store, point, or buffer validation failure.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch is undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::UnsupportedKernelOperation When store slots/mode disagree or x is missing.
        /// @retval GprErrorCode::EmptyInput When store or x is empty.
        /// @retval GprErrorCode::LengthMismatch When out or x row counts disagree with store.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with a coordinate ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        [[nodiscard]] std::expected<void, GprError> apply(
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x,
            MatrixMut<T> out,
            Triangle uplo,
            MatrixMut<T> scratch
        ) const;

        /// @brief Evaluates the cross-covariance matrix from a validated rectangular store.
        /// @details Uses caller-supplied scratch and nested buffers for zero-allocation execution.
        /// @param store Validated rectangular CheckedDistanceStore<T> matching this kernel's slots.
        /// @param x1 Optional n1 x d first point matrix (required when requires_points() is true).
        /// @param x2 Optional n2 x d second point matrix (required when requires_points() is true).
        /// @param out Column-major n1 x n2 destination cross-covariance matrix.
        /// @param scratch Non-aliasing n1 x n2 scratch matrix buffer.
        /// @param nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n1 x n2.
        /// @return Empty expected on success or GprError on store, point, or buffer validation failure.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::UnsupportedKernelOperation When store slots/mode disagree or x1/x2 are missing.
        /// @retval GprErrorCode::EmptyInput When store, x1, or x2 is empty.
        /// @retval GprErrorCode::LengthMismatch When out or point row counts disagree with store.
        /// @retval GprErrorCode::DimensionMismatch When x1/x2 columns mismatch each other or coordinate ARD dims.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x1 or x2 is non-finite.
        [[nodiscard]] std::expected<void, GprError> apply_cross(
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x1,
            std::optional<MatrixRef<T>> x2,
            MatrixMut<T> out,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates the cross-covariance matrix from a validated rectangular store.
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param store Validated rectangular CheckedDistanceStore<T> matching this kernel's slots.
        /// @param x1 Optional n1 x d first point matrix (required when requires_points() is true).
        /// @param x2 Optional n2 x d second point matrix (required when requires_points() is true).
        /// @param out Column-major n1 x n2 destination cross-covariance matrix.
        /// @param scratch Non-aliasing n1 x n2 scratch matrix buffer.
        /// @return Empty expected on success or GprError on store, point, or buffer validation failure.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch is undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch aliases out.
        /// @retval GprErrorCode::UnsupportedKernelOperation When store slots/mode disagree or x1/x2 are missing.
        /// @retval GprErrorCode::EmptyInput When store, x1, or x2 is empty.
        /// @retval GprErrorCode::LengthMismatch When out or point row counts disagree with store.
        /// @retval GprErrorCode::DimensionMismatch When x1/x2 columns mismatch each other or coordinate ARD dims.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x1 or x2 is non-finite.
        [[nodiscard]] std::expected<void, GprError> apply_cross(
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x1,
            std::optional<MatrixRef<T>> x2,
            MatrixMut<T> out,
            MatrixMut<T> scratch
        ) const;

        /// @brief Fills the diagonal k(x_i, x_i) for n points.
        /// @details Requires x when coordinate leaves are present; supplied stationary leaves evaluate at
        /// zero distance without requiring a CheckedDistanceStore.
        /// @param n Number of diagonal entries to evaluate.
        /// @param x Optional n x d coordinate matrix (required when requires_points() is true).
        /// @param out Destination span of length n receiving k(x_i, x_i).
        /// @return Empty expected on success or GprError on invalid count, missing x, or invalid coordinates.
        /// @retval GprErrorCode::EmptyInput When n == 0 or x is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != n or x.rows() != n.
        /// @retval GprErrorCode::UnsupportedKernelOperation When requires_points() is true and x is std::nullopt.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with a coordinate ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        [[nodiscard]] std::expected<void, GprError> fill_diag(
            std::size_t n, std::optional<MatrixRef<T>> x, std::span<T> out
        ) const;

        /// @brief Evaluates the first derivative wrt parameter param_idx.
        /// @details Differentiates supplied distance and coordinate leaves with caller-supplied scratch
        /// and nested buffers.
        /// @param store Validated square CheckedDistanceStore<T> matching this kernel's slots.
        /// @param x Optional n x d training coordinate matrix (required when requires_points() is true).
        /// @param d_k Column-major n x n destination derivative matrix.
        /// @param param_idx Zero-based flattened hyperparameter index in [0, num_params()).
        /// @param uplo Matrix triangle or full region to populate.
        /// @param scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @param nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @return Empty expected on success or GprError on store, point, buffer, or index error.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch buffers alias d_k or each other.
        /// @retval GprErrorCode::UnsupportedKernelOperation When store slots/mode disagree or x is missing.
        /// @retval GprErrorCode::EmptyInput When store or x is empty.
        /// @retval GprErrorCode::LengthMismatch When d_k or x row counts disagree with store.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with a coordinate ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When param_idx >= num_params().
        [[nodiscard]] std::expected<void, GprError> grad(
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x,
            MatrixMut<T> d_k,
            std::size_t param_idx,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates the first derivative wrt parameter param_idx.
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param store Validated square CheckedDistanceStore<T> matching this kernel's slots.
        /// @param x Optional n x d training coordinate matrix (required when requires_points() is true).
        /// @param d_k Column-major n x n destination derivative matrix.
        /// @param param_idx Zero-based flattened hyperparameter index in [0, num_params()).
        /// @param uplo Matrix triangle or full region to populate.
        /// @param scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @return Empty expected on success or GprError on store, point, buffer, or index error.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When scratch buffers alias d_k or each other.
        /// @retval GprErrorCode::UnsupportedKernelOperation When store slots/mode disagree or x is missing.
        /// @retval GprErrorCode::EmptyInput When store or x is empty.
        /// @retval GprErrorCode::LengthMismatch When d_k or x row counts disagree with store.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with a coordinate ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When param_idx >= num_params().
        [[nodiscard]] std::expected<void, GprError> grad(
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x,
            MatrixMut<T> d_k,
            std::size_t param_idx,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b
        ) const;

        /// @brief Evaluates the second derivative wrt parameters (i, j).
        /// @details Differentiates supplied distance and coordinate leaves with three non-aliasing
        /// scratch matrices and caller-supplied nested buffers.
        /// @param store Validated square CheckedDistanceStore<T> matching this kernel's slots.
        /// @param x Optional n x d training coordinate matrix (required when requires_points() is true).
        /// @param d2_k Column-major n x n destination second-derivative matrix.
        /// @param i First flattened hyperparameter index in [0, num_params()).
        /// @param j Second flattened hyperparameter index in [0, num_params()).
        /// @param uplo Matrix triangle or full region to populate.
        /// @param scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @param scratch_c Third non-aliasing n x n scratch matrix buffer.
        /// @param nested Pre-allocated NestedBuffers with depth >= nested_depth() and shape >= n x n.
        /// @return Empty expected on success or GprError on store, point, buffer, or index error.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch or nested buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When any scratch buffers alias d2_k or each other.
        /// @retval GprErrorCode::UnsupportedKernelOperation When store slots/mode disagree or x is missing.
        /// @retval GprErrorCode::EmptyInput When store or x is empty.
        /// @retval GprErrorCode::LengthMismatch When d2_k or x row counts disagree with store.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with a coordinate ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When i >= num_params() or j >= num_params().
        [[nodiscard]] std::expected<void, GprError> hess(
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x,
            MatrixMut<T> d2_k,
            std::size_t i,
            std::size_t j,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            MatrixMut<T> scratch_c,
            NestedBuffers<T>& nested
        ) const;

        /// @brief Evaluates the second derivative wrt parameters (i, j).
        /// @details Convenience overload allocating temporary NestedBuffers via make_nested_buffers().
        /// @param store Validated square CheckedDistanceStore<T> matching this kernel's slots.
        /// @param x Optional n x d training coordinate matrix (required when requires_points() is true).
        /// @param d2_k Column-major n x n destination second-derivative matrix.
        /// @param i First flattened hyperparameter index in [0, num_params()).
        /// @param j Second flattened hyperparameter index in [0, num_params()).
        /// @param uplo Matrix triangle or full region to populate.
        /// @param scratch_a First non-aliasing n x n scratch matrix buffer.
        /// @param scratch_b Second non-aliasing n x n scratch matrix buffer.
        /// @param scratch_c Third non-aliasing n x n scratch matrix buffer.
        /// @return Empty expected on success or GprError on store, point, buffer, or index error.
        /// @retval GprErrorCode::WorkspaceTooSmall When scratch buffers are undersized.
        /// @retval GprErrorCode::InvalidConfig When any scratch buffers alias d2_k or each other.
        /// @retval GprErrorCode::UnsupportedKernelOperation When store slots/mode disagree or x is missing.
        /// @retval GprErrorCode::EmptyInput When store or x is empty.
        /// @retval GprErrorCode::LengthMismatch When d2_k or x row counts disagree with store.
        /// @retval GprErrorCode::DimensionMismatch When x.cols() disagrees with a coordinate ARD leaf.
        /// @retval GprErrorCode::NonFiniteInput When any entry of x is non-finite.
        /// @retval GprErrorCode::IndexOutOfRange When i >= num_params() or j >= num_params().
        [[nodiscard]] std::expected<void, GprError> hess(
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x,
            MatrixMut<T> d2_k,
            std::size_t i,
            std::size_t j,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            MatrixMut<T> scratch_c
        ) const;

    private:
        /// @brief Pimpl state storing the flattened execution AST, leaves, slot descriptors, and bindings.
        struct Impl;

        /// @brief Private constructor taking ownership of a compiled Impl state.
        /// @details Invoked by DistanceKernel<Mode>::compile<T>().
        /// @param impl Heap-allocated compiled distance kernel implementation.
        explicit CompiledDistanceKernel(std::unique_ptr<Impl> impl) noexcept;

        /// @brief Owning pointer to the compiled distance kernel implementation.
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

}  // namespace cxxgpr
