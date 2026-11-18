// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace cxxgpr
{

    /// Selects which triangular or full region of a symmetric matrix is written or read.
    enum class Triangle : std::uint8_t
    {
        Lower,
        Upper,
        Full,
    };

    /// Selects which predictive variance a fitted model reports.
    enum class VarianceKind : std::uint8_t
    {
        /// Predictive variance of the latent function f*, excluding observation noise.
        Latent,
        /// Predictive variance of a new observation y*, including sigma_n^2.
        Observation,
    };

    /// Configuration options controlling prediction calls.
    struct PredictOptions
    {
        /// Selects latent or observation predictive variance (defaults to Observation).
        VarianceKind variance_kind{VarianceKind::Observation};

        /// Checks value equality between two PredictOptions instances.
        [[nodiscard]] constexpr bool operator==(const PredictOptions& other) const noexcept = default;
    };

    /// Predictive mean and diagonal variance on the original target scale.
    template <typename T = double> struct Prediction
    {
        /// Predictive mean values on the original target scale.
        std::vector<T> mean;
        /// Predictive diagonal variance values on the original target scale.
        std::vector<T> variance;
        /// Indicates whether variance excludes (Latent) or includes (Observation) noise variance.
        VarianceKind variance_kind{VarianceKind::Observation};

        /// Checks value equality between two Prediction instances.
        [[nodiscard]] bool operator==(const Prediction& other) const noexcept = default;
    };

    /// Predictive mean and full column-major query-query covariance matrix on the original target
    /// scale.
    template <typename T = double> struct PredictiveCovariance
    {
        /// Predictive mean values of length n_points.
        std::vector<T> mean;
        /// Column-major m x m posterior covariance matrix packed as col * n_points + row.
        std::vector<T> covariance;
        /// Number of query points m.
        std::size_t n_points{0};
        /// Indicates whether covariance excludes (Latent) or includes (Observation) noise variance.
        VarianceKind variance_kind{VarianceKind::Observation};

        /// Checks value equality between two PredictiveCovariance instances.
        [[nodiscard]] bool operator==(const PredictiveCovariance& other) const noexcept = default;
    };

    /// Non-owning read-only view over a column-major matrix buffer.
    template <typename T> class MatrixRef
    {
    public:
        /// Constructs an empty matrix view.
        constexpr MatrixRef() noexcept = default;

        /// Constructs a column-major view from raw pointer, dimensions, and column stride.
        constexpr MatrixRef(const T* data, std::size_t rows, std::size_t cols, std::size_t col_stride) noexcept :
            data_(data),
            rows_(rows),
            cols_(cols),
            col_stride_(col_stride)
        {
        }

        /// Constructs a tightly packed column-major view where col_stride equals rows.
        constexpr MatrixRef(const T* data, std::size_t rows, std::size_t cols) noexcept :
            MatrixRef(data, rows, cols, rows)
        {
        }

        /// Constructs a tightly packed column-major view over a contiguous span.
        constexpr MatrixRef(std::span<const T> data, std::size_t rows, std::size_t cols) noexcept :
            MatrixRef(data.data(), rows, cols, rows)
        {
        }

        /// Returns the base pointer to element (0, 0).
        [[nodiscard]] constexpr const T* data() const noexcept { return data_; }

        /// Returns the number of rows.
        [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }

        /// Returns the number of columns.
        [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }

        /// Returns the distance in elements between adjacent columns.
        [[nodiscard]] constexpr std::size_t col_stride() const noexcept { return col_stride_; }

        /// Returns true if the matrix has zero rows or zero columns.
        [[nodiscard]] constexpr bool is_empty() const noexcept { return rows_ == 0 || cols_ == 0; }

        /// Returns true if the matrix is square (rows == cols).
        [[nodiscard]] constexpr bool is_square() const noexcept { return rows_ == cols_; }

        /// Returns true if columns are tightly packed without stride padding.
        [[nodiscard]] constexpr bool is_contiguous() const noexcept { return col_stride_ == rows_; }

        /// Returns a const reference to element (row, col) using column-major addressing.
        [[nodiscard]] constexpr const T& operator()(std::size_t row, std::size_t col) const noexcept
        {
            return data_[col * col_stride_ + row];
        }

        /// Returns a contiguous read-only span over column col_idx.
        [[nodiscard]] constexpr std::span<const T> col(std::size_t col_idx) const noexcept
        {
            if (data_ == nullptr || rows_ == 0 || col_idx >= cols_)
            {
                return {};
            }
            return {data_ + col_idx * col_stride_, rows_};
        }

    private:
        const T* data_{nullptr};
        std::size_t rows_{0};
        std::size_t cols_{0};
        std::size_t col_stride_{0};
    };

    /// Non-owning mutable view over a column-major matrix buffer.
    template <typename T> class MatrixMut
    {
    public:
        /// Constructs an empty mutable matrix view.
        constexpr MatrixMut() noexcept = default;

        /// Constructs a column-major mutable view from raw pointer, dimensions, and column stride.
        constexpr MatrixMut(T* data, std::size_t rows, std::size_t cols, std::size_t col_stride) noexcept :
            data_(data),
            rows_(rows),
            cols_(cols),
            col_stride_(col_stride)
        {
        }

        /// Constructs a tightly packed mutable column-major view where col_stride equals rows.
        constexpr MatrixMut(T* data, std::size_t rows, std::size_t cols) noexcept : MatrixMut(data, rows, cols, rows) {}

        /// Constructs a tightly packed mutable column-major view over a contiguous span.
        constexpr MatrixMut(std::span<T> data, std::size_t rows, std::size_t cols) noexcept :
            MatrixMut(data.data(), rows, cols, rows)
        {
        }

        /// Returns the base mutable pointer to element (0, 0).
        [[nodiscard]] constexpr T* data() const noexcept { return data_; }

        /// Returns the number of rows.
        [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }

        /// Returns the number of columns.
        [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }

        /// Returns the distance in elements between adjacent columns.
        [[nodiscard]] constexpr std::size_t col_stride() const noexcept { return col_stride_; }

        /// Returns true if the matrix has zero rows or zero columns.
        [[nodiscard]] constexpr bool is_empty() const noexcept { return rows_ == 0 || cols_ == 0; }

        /// Returns true if the matrix is square (rows == cols).
        [[nodiscard]] constexpr bool is_square() const noexcept { return rows_ == cols_; }

        /// Returns true if columns are tightly packed without stride padding.
        [[nodiscard]] constexpr bool is_contiguous() const noexcept { return col_stride_ == rows_; }

        /// Returns a mutable reference to element (row, col) using column-major addressing.
        [[nodiscard]] constexpr T& operator()(std::size_t row, std::size_t col) const noexcept
        {
            return data_[col * col_stride_ + row];
        }

        /// Returns a contiguous mutable span over column col_idx.
        [[nodiscard]] constexpr std::span<T> col(std::size_t col_idx) const noexcept
        {
            if (data_ == nullptr || rows_ == 0 || col_idx >= cols_)
            {
                return {};
            }
            return {data_ + col_idx * col_stride_, rows_};
        }

        /// Returns a read-only view referencing the same underlying matrix region.
        [[nodiscard]] constexpr MatrixRef<T> as_ref() const noexcept { return {data_, rows_, cols_, col_stride_}; }

        /// Implicitly converts a mutable view to a read-only view.
        // NOLINTNEXTLINE(google-explicit-constructor)
        [[nodiscard]] constexpr operator MatrixRef<T>() const noexcept { return as_ref(); }

    private:
        T* data_{nullptr};
        std::size_t rows_{0};
        std::size_t cols_{0};
        std::size_t col_stride_{0};
    };

    namespace kernel
    {

        /// Re-exports Triangle into the kernel namespace.
        using cxxgpr::Triangle;

    }  // namespace kernel

}  // namespace cxxgpr
