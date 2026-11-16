// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <expected>
#include <span>

namespace cxxgpr::linalg {

/// Adds a finite scalar offset to every diagonal entry of a square matrix in-place.
template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> add_to_diag(MatrixMut<T> a, T value);

/// Copies the lower triangle (including diagonal) of a square matrix src into dst.
template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> copy_lower_triangle(MatrixRef<T> src, MatrixMut<T> dst);

/// Mirrors the strictly lower triangle of a square matrix into its strictly upper triangle.
template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> symmetrize_lower(MatrixMut<T> a);

/// Computes the symmetric Frobenius inner product <A, B>_F reading only the lower triangles.
template <KernelScalarConcept T>
[[nodiscard]] std::expected<double, GprError> frobenius_lower(MatrixRef<T> a, MatrixRef<T> b);

/// Computes the Euclidean dot product of two equal-length vectors in deterministic ascending order.
template <KernelScalarConcept T>
[[nodiscard]] std::expected<double, GprError> dot(std::span<const T> a, std::span<const T> b);

/// Computes out = A * x in column-major order for a matrix A and vector x.
template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> matvec_columns(MatrixRef<T> a, std::span<const T> x,
                                                           std::span<T> out);

} // namespace cxxgpr::linalg
