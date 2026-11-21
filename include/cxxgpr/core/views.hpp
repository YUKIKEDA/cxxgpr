// SPDX-License-Identifier: Apache-2.0
/// @file views.hpp
/// @brief Matrix views, prediction result types, and triangular/variance enumerations.
/// @details Defines lightweight non-owning column-major matrix views (MatrixRef, MatrixMut) that
/// keep Eigen and internal storage types out of public headers, together with prediction options
/// and output structs.

#pragma once

#include "cxxgpr/core/error.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <utility>
#include <vector>

/// @brief Top-level namespace for the cxxgpr Gaussian process regression library.
namespace cxxgpr
{

    /// @brief Selects which triangular or full region of a symmetric matrix is written or read.
    /// @details Used by kernel evaluation and distance-matrix routines to avoid redundant symmetric work.
    enum class Triangle : std::uint8_t
    {
        /// @brief Lower triangular entries (row >= col) including the main diagonal.
        Lower,
        /// @brief Upper triangular entries (row <= col) including the main diagonal.
        Upper,
        /// @brief All matrix entries across both triangles.
        Full,
    };

    /// @brief Selects which predictive variance a fitted model reports.
    /// @details Distinguishes noise-free latent function uncertainty from noisy observation uncertainty.
    enum class VarianceKind : std::uint8_t
    {
        /// @brief Predictive variance of the latent function f*, excluding observation noise.
        Latent,
        /// @brief Predictive variance of a new observation y*, including sigma_n^2.
        Observation,
    };

    /// @brief Configuration options controlling prediction calls.
    /// @details Bundles predictive variance kind selection for FittedGpr prediction routines.
    struct PredictOptions
    {
        /// @brief Selects latent or observation predictive variance (defaults to Observation).
        VarianceKind variance_kind{VarianceKind::Observation};

        /// @brief Checks value equality between two PredictOptions instances.
        /// @details Compares variance_kind directly.
        /// @param[in] other PredictOptions instance to compare against.
        /// @return True if options match.
        [[nodiscard]] constexpr bool operator==(const PredictOptions& other) const noexcept = default;
    };

    /// @brief Predictive mean and diagonal variance on the original target scale.
    /// @details Returned by pointwise prediction calls over m query inputs.
    /// @tparam T Scalar type of returned predictions (double or float).
    template <typename T = double> struct Prediction
    {
        /// @brief Predictive mean values on the original target scale.
        std::vector<T> mean;
        /// @brief Predictive diagonal variance values on the original target scale.
        std::vector<T> variance;
        /// @brief Indicates whether variance excludes (Latent) or includes (Observation) noise variance.
        VarianceKind variance_kind{VarianceKind::Observation};

        /// @brief Checks value equality between two Prediction instances.
        /// @details Compares mean, variance, and variance_kind elementwise.
        /// @param[in] other Prediction instance to compare against.
        /// @return True if all fields compare equal.
        [[nodiscard]] bool operator==(const Prediction& other) const noexcept = default;
    };

    /// @brief Predictive mean and full column-major query-query covariance matrix on the original target scale.
    /// @details Returned by joint posterior covariance prediction calls over m query inputs.
    /// @tparam T Scalar type of returned predictions (double or float).
    template <typename T = double> struct PredictiveCovariance
    {
        /// @brief Predictive mean values of length n_points.
        std::vector<T> mean;
        /// @brief Column-major m x m posterior covariance matrix packed as col * n_points + row.
        std::vector<T> covariance;
        /// @brief Number of query points m.
        std::size_t n_points{0};
        /// @brief Indicates whether covariance excludes (Latent) or includes (Observation) noise variance.
        VarianceKind variance_kind{VarianceKind::Observation};

        /// @brief Checks value equality between two PredictiveCovariance instances.
        /// @details Compares mean, covariance, n_points, and variance_kind.
        /// @param[in] other PredictiveCovariance instance to compare against.
        /// @return True if all fields compare equal.
        [[nodiscard]] bool operator==(const PredictiveCovariance& other) const noexcept = default;
    };

    /// @brief Internal implementation helpers for matrix views.
    namespace detail
    {

        /// @brief Checks that a contiguous buffer of span_size elements holds exactly a rows x cols matrix.
        /// @details Applies, in this order, the EmptyInput check (span_size, rows, or cols is zero), the
        /// SizeOverflow check (rows * cols overflows std::size_t), and the LengthMismatch check
        /// (span_size != rows * cols). An empty buffer is rejected even when its pointer is non-null, because
        /// such a pointer may address no readable element.
        /// @param[in] span_size Number of elements in the supplied buffer.
        /// @param[in] rows Declared number of matrix rows.
        /// @param[in] cols Declared number of matrix columns.
        /// @return Empty expected when the shape is valid, or the first violated GprError.
        /// @retval GprErrorCode::EmptyInput When span_size == 0, rows == 0, or cols == 0.
        /// @retval GprErrorCode::SizeOverflow When rows * cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When span_size != rows * cols.
        [[nodiscard]] std::expected<void, GprError> check_matrix_span_shape(
            std::size_t span_size, std::size_t rows, std::size_t cols
        );

    }  // namespace detail

    /// @brief Non-owning read-only view over a column-major matrix buffer.
    /// @details Addresses element (row, col) at offset col * col_stride_ + row without exposing
    /// third-party matrix types across the public API.
    /// @tparam T Matrix element scalar type.
    template <typename T> class MatrixRef
    {
    public:
        /// @brief Constructs an empty matrix view.
        /// @details Sets data_ to nullptr and all extents/strides to zero.
        constexpr MatrixRef() noexcept = default;

        /// @brief Constructs a column-major view from raw pointer, dimensions, and column stride.
        /// @details Allows non-tightly-packed strided submatrix views when col_stride >= rows.
        /// @param[in] data Pointer to element (0, 0) in column-major layout.
        /// @param[in] rows Number of matrix rows.
        /// @param[in] cols Number of matrix columns.
        /// @param[in] col_stride Distance in elements between consecutive columns.
        /// @pre data addresses at least (cols - 1) * col_stride + rows readable elements when rows and cols are
        /// non-zero. The constructor does not check this; use from_span() to build a checked view over a span.
        constexpr MatrixRef(const T* data, std::size_t rows, std::size_t cols, std::size_t col_stride) noexcept :
            data_(data),
            rows_(rows),
            cols_(cols),
            col_stride_(col_stride)
        {
        }

        /// @brief Constructs a tightly packed column-major view where col_stride equals rows.
        /// @details Delegates to the strided constructor with col_stride == rows.
        /// @param[in] data Pointer to tightly packed column-major buffer.
        /// @param[in] rows Number of matrix rows.
        /// @param[in] cols Number of matrix columns.
        /// @pre data addresses at least rows * cols readable elements. The constructor does not check this; use
        /// from_span() to build a checked view over a span.
        constexpr MatrixRef(const T* data, std::size_t rows, std::size_t cols) noexcept :
            MatrixRef(data, rows, cols, rows)
        {
        }

        /// @brief Builds a tightly packed column-major view over a span after checking its length.
        /// @details Checks, in this order, that data, rows, and cols are non-empty, that rows * cols does not
        /// overflow, and that data.size() equals rows * cols. The view binds data.data() with col_stride == rows
        /// only after all three hold, so it never addresses elements past the end of data.
        /// @param[in] data Contiguous span backing the column-major matrix.
        /// @param[in] rows Number of matrix rows.
        /// @param[in] cols Number of matrix columns.
        /// @return View over data, or GprError describing the first violated shape condition.
        /// @retval GprErrorCode::EmptyInput When data is empty, rows == 0, or cols == 0.
        /// @retval GprErrorCode::SizeOverflow When rows * cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When data.size() != rows * cols.
        /// @note Does not allocate on success.
        /// @see MatrixMut::from_span()
        [[nodiscard]] static std::expected<MatrixRef, GprError> from_span(
            std::span<const T> data, std::size_t rows, std::size_t cols
        )
        {
            // Validate the shape before binding the pointer so no view can extend past the span.
            if (auto ok = detail::check_matrix_span_shape(data.size(), rows, cols); !ok.has_value())
            {
                return std::unexpected(std::move(ok).error());
            }
            return MatrixRef(data.data(), rows, cols, rows);
        }

        /// @brief Returns the base pointer to element (0, 0).
        /// @details May be nullptr for default-constructed empty views.
        /// @return Const pointer to the start of the underlying matrix buffer.
        [[nodiscard]] constexpr const T* data() const noexcept { return data_; }

        /// @brief Returns the number of rows.
        /// @details Row indices span 0 .. rows() - 1.
        /// @return Matrix row count.
        [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }

        /// @brief Returns the number of columns.
        /// @details Column indices span 0 .. cols() - 1.
        /// @return Matrix column count.
        [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }

        /// @brief Returns the distance in elements between adjacent columns.
        /// @details Equals rows() when tightly packed.
        /// @return Leading-dimension column stride in elements.
        [[nodiscard]] constexpr std::size_t col_stride() const noexcept { return col_stride_; }

        /// @brief Returns true if the matrix has zero rows or zero columns.
        /// @details Indicates no elements are addressed by this view.
        /// @return True iff rows_ == 0 || cols_ == 0.
        [[nodiscard]] constexpr bool is_empty() const noexcept { return rows_ == 0 || cols_ == 0; }

        /// @brief Returns true if the matrix is square (rows == cols).
        /// @details Note that a 0 x 0 view is considered square.
        /// @return True iff rows_ == cols_.
        [[nodiscard]] constexpr bool is_square() const noexcept { return rows_ == cols_; }

        /// @brief Returns true if columns are tightly packed without stride padding.
        /// @details Equivalent to col_stride_ == rows_.
        /// @return True iff there is no inter-column padding.
        [[nodiscard]] constexpr bool is_contiguous() const noexcept { return col_stride_ == rows_; }

        /// @brief Returns a const reference to element (row, col) using column-major addressing.
        /// @details Caller must ensure row < rows() and col < cols().
        /// @param[in] row Zero-based row index.
        /// @param[in] col Zero-based column index.
        /// @return Const reference to element (row, col).
        [[nodiscard]] constexpr const T& operator()(std::size_t row, std::size_t col) const noexcept
        {
            // Column-major index advances by col_stride_ per column and 1 per row.
            return data_[col * col_stride_ + row];
        }

        /// @brief Returns a contiguous read-only span over column col_idx.
        /// @details Returns an empty span when the view is null/empty or col_idx >= cols().
        /// @param[in] col_idx Zero-based column index.
        /// @return Contiguous span of length rows() for column col_idx, or empty span if invalid.
        [[nodiscard]] constexpr std::span<const T> col(std::size_t col_idx) const noexcept
        {
            // Guard against null backing pointer, zero rows, or out-of-bounds column index.
            if (data_ == nullptr || rows_ == 0 || col_idx >= cols_)
            {
                return {};
            }
            return {data_ + col_idx * col_stride_, rows_};
        }

    private:
        /// @brief Pointer to element (0, 0) in the backing column-major buffer.
        const T* data_{nullptr};
        /// @brief Number of matrix rows.
        std::size_t rows_{0};
        /// @brief Number of matrix columns.
        std::size_t cols_{0};
        /// @brief Element stride between the start of consecutive columns.
        std::size_t col_stride_{0};
    };

    /// @brief Non-owning mutable view over a column-major matrix buffer.
    /// @details Provides read/write column-major access and implicit conversion to MatrixRef<T>
    /// for custom kernel implementations and zero-allocation workspace buffers.
    /// @tparam T Matrix element scalar type.
    template <typename T> class MatrixMut
    {
    public:
        /// @brief Constructs an empty mutable matrix view.
        /// @details Sets data_ to nullptr and all extents/strides to zero.
        constexpr MatrixMut() noexcept = default;

        /// @brief Constructs a column-major mutable view from raw pointer, dimensions, and column stride.
        /// @details Supports strided submatrix targets where col_stride >= rows.
        /// @param[in] data Mutable pointer to element (0, 0) in column-major layout.
        /// @param[in] rows Number of matrix rows.
        /// @param[in] cols Number of matrix columns.
        /// @param[in] col_stride Distance in elements between consecutive columns.
        /// @pre data addresses at least (cols - 1) * col_stride + rows writable elements when rows and cols are
        /// non-zero. The constructor does not check this; use from_span() to build a checked view over a span.
        constexpr MatrixMut(T* data, std::size_t rows, std::size_t cols, std::size_t col_stride) noexcept :
            data_(data),
            rows_(rows),
            cols_(cols),
            col_stride_(col_stride)
        {
        }

        /// @brief Constructs a tightly packed mutable column-major view where col_stride equals rows.
        /// @details Delegates to the strided constructor with col_stride == rows.
        /// @param[in] data Mutable pointer to tightly packed column-major buffer.
        /// @param[in] rows Number of matrix rows.
        /// @param[in] cols Number of matrix columns.
        /// @pre data addresses at least rows * cols writable elements. The constructor does not check this; use
        /// from_span() to build a checked view over a span.
        constexpr MatrixMut(T* data, std::size_t rows, std::size_t cols) noexcept : MatrixMut(data, rows, cols, rows) {}

        /// @brief Builds a tightly packed mutable column-major view over a span after checking its length.
        /// @details Checks, in this order, that data, rows, and cols are non-empty, that rows * cols does not
        /// overflow, and that data.size() equals rows * cols. The view binds data.data() with col_stride == rows
        /// only after all three hold, so it never addresses elements past the end of data.
        /// @param[in] data Contiguous mutable span backing the column-major matrix.
        /// @param[in] rows Number of matrix rows.
        /// @param[in] cols Number of matrix columns.
        /// @return Mutable view over data, or GprError describing the first violated shape condition.
        /// @retval GprErrorCode::EmptyInput When data is empty, rows == 0, or cols == 0.
        /// @retval GprErrorCode::SizeOverflow When rows * cols overflows std::size_t.
        /// @retval GprErrorCode::LengthMismatch When data.size() != rows * cols.
        /// @note Does not allocate on success.
        /// @see MatrixRef::from_span()
        [[nodiscard]] static std::expected<MatrixMut, GprError> from_span(
            std::span<T> data, std::size_t rows, std::size_t cols
        )
        {
            // Validate the shape before binding the pointer so no view can extend past the span.
            if (auto ok = detail::check_matrix_span_shape(data.size(), rows, cols); !ok.has_value())
            {
                return std::unexpected(std::move(ok).error());
            }
            return MatrixMut(data.data(), rows, cols, rows);
        }

        /// @brief Returns the base mutable pointer to element (0, 0).
        /// @details May be nullptr for default-constructed empty views.
        /// @return Mutable pointer to the start of the underlying matrix buffer.
        [[nodiscard]] constexpr T* data() const noexcept { return data_; }

        /// @brief Returns the number of rows.
        /// @details Row indices span 0 .. rows() - 1.
        /// @return Matrix row count.
        [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }

        /// @brief Returns the number of columns.
        /// @details Column indices span 0 .. cols() - 1.
        /// @return Matrix column count.
        [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }

        /// @brief Returns the distance in elements between adjacent columns.
        /// @details Equals rows() when tightly packed.
        /// @return Leading-dimension column stride in elements.
        [[nodiscard]] constexpr std::size_t col_stride() const noexcept { return col_stride_; }

        /// @brief Returns true if the matrix has zero rows or zero columns.
        /// @details Indicates no elements are addressed by this view.
        /// @return True iff rows_ == 0 || cols_ == 0.
        [[nodiscard]] constexpr bool is_empty() const noexcept { return rows_ == 0 || cols_ == 0; }

        /// @brief Returns true if the matrix is square (rows == cols).
        /// @details Note that a 0 x 0 view is considered square.
        /// @return True iff rows_ == cols_.
        [[nodiscard]] constexpr bool is_square() const noexcept { return rows_ == cols_; }

        /// @brief Returns true if columns are tightly packed without stride padding.
        /// @details Equivalent to col_stride_ == rows_.
        /// @return True iff there is no inter-column padding.
        [[nodiscard]] constexpr bool is_contiguous() const noexcept { return col_stride_ == rows_; }

        /// @brief Returns a mutable reference to element (row, col) using column-major addressing.
        /// @details Caller must ensure row < rows() and col < cols().
        /// @param[in] row Zero-based row index.
        /// @param[in] col Zero-based column index.
        /// @return Mutable reference to element (row, col).
        [[nodiscard]] constexpr T& operator()(std::size_t row, std::size_t col) const noexcept
        {
            // Column-major index advances by col_stride_ per column and 1 per row.
            return data_[col * col_stride_ + row];
        }

        /// @brief Returns a contiguous mutable span over column col_idx.
        /// @details Returns an empty span when the view is null/empty or col_idx >= cols().
        /// @param[in] col_idx Zero-based column index.
        /// @return Contiguous mutable span of length rows() for column col_idx, or empty span if invalid.
        [[nodiscard]] constexpr std::span<T> col(std::size_t col_idx) const noexcept
        {
            // Guard against null backing pointer, zero rows, or out-of-bounds column index.
            if (data_ == nullptr || rows_ == 0 || col_idx >= cols_)
            {
                return {};
            }
            return {data_ + col_idx * col_stride_, rows_};
        }

        /// @brief Returns a read-only view referencing the same underlying matrix region.
        /// @details Preserves data pointer, dimensions, and column stride.
        /// @return MatrixRef<T> over the same buffer.
        [[nodiscard]] constexpr MatrixRef<T> as_ref() const noexcept { return {data_, rows_, cols_, col_stride_}; }

        /// @brief Implicitly converts a mutable view to a read-only view.
        /// @details Delegates to as_ref() so MatrixMut<T> binds seamlessly to MatrixRef<T> parameters.
        /// @return MatrixRef<T> referencing the same buffer.
        // NOLINTNEXTLINE(google-explicit-constructor)
        [[nodiscard]] constexpr operator MatrixRef<T>() const noexcept { return as_ref(); }

    private:
        /// @brief Mutable pointer to element (0, 0) in the backing column-major buffer.
        T* data_{nullptr};
        /// @brief Number of matrix rows.
        std::size_t rows_{0};
        /// @brief Number of matrix columns.
        std::size_t cols_{0};
        /// @brief Element stride between the start of consecutive columns.
        std::size_t col_stride_{0};
    };

    /// @brief Kernel specification, evaluation, and distance-matrix utilities.
    namespace kernel
    {

        /// @brief Re-exports Triangle into the kernel namespace.
        using cxxgpr::Triangle;

    }  // namespace kernel

}  // namespace cxxgpr
