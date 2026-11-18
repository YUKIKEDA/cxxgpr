// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/kernel/exp.hpp"
#include "cxxgpr/kernel/spec.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <cstddef>
#include <expected>
#include <span>

namespace cxxgpr::detail
{

    [[nodiscard]] CoordMode leaf_coord_mode(const KernelLeaf& leaf) noexcept;
    [[nodiscard]] std::size_t leaf_num_params(const KernelLeaf& leaf) noexcept;
    [[nodiscard]] std::expected<void, GprError> leaf_get_params(const KernelLeaf& leaf, std::span<double> out);
    [[nodiscard]] std::expected<void, GprError> leaf_set_params(KernelLeaf& leaf, std::span<const double> params);
    [[nodiscard]] std::expected<void, GprError> leaf_bounds_into(const KernelLeaf& leaf, std::span<Interval> out);

    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_apply_dist(
        const KernelLeaf& leaf,
        MatrixRef<T> dist,
        MatrixMut<T> out,
        Triangle uplo,
        KernelExp exp_mode = KernelExp::Accurate
    );

    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_apply_points(
        const KernelLeaf& leaf,
        MatrixRef<T> x,
        MatrixMut<T> out,
        Triangle uplo,
        KernelExp exp_mode = KernelExp::Accurate,
        std::span<const double> cached_ls = {}
    );

    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_apply_cross_dist(
        const KernelLeaf& leaf, MatrixRef<T> dist, MatrixMut<T> out, KernelExp exp_mode = KernelExp::Accurate
    );

    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_apply_cross_points(
        const KernelLeaf& leaf,
        MatrixRef<T> x1,
        MatrixRef<T> x2,
        MatrixMut<T> out,
        KernelExp exp_mode = KernelExp::Accurate,
        std::span<const double> cached_ls = {}
    );

    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_fill_diag(
        const KernelLeaf& leaf, MatrixRef<T> x, std::span<T> out, std::span<const double> cached_ls = {}
    );

    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> leaf_grad_dist(
        const KernelLeaf& leaf,
        MatrixRef<T> dist,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        KernelExp exp_mode = KernelExp::Accurate
    );

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
