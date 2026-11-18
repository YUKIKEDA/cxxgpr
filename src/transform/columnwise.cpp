// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/transform/columnwise.hpp"

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "transform/detail.hpp"

namespace cxxgpr::transform
{

    namespace
    {

        template <typename Ptr> std::vector<Ptr> clone_ptr_vec(const std::vector<Ptr>& src)
        {
            std::vector<Ptr> out;
            out.reserve(src.size());
            for (const auto& item : src)
            {
                if (item != nullptr)
                {
                    out.push_back(item->clone());
                }
            }
            return out;
        }

    }  // namespace

    // --- FittedColumnwiseInput ---

    FittedColumnwiseInput::FittedColumnwiseInput(std::vector<std::unique_ptr<FittedInputTransform>> maps) noexcept :
        maps_(std::move(maps))
    {
    }

    FittedColumnwiseInput::FittedColumnwiseInput(const FittedColumnwiseInput& other) : maps_(clone_ptr_vec(other.maps_))
    {
    }

    FittedColumnwiseInput& FittedColumnwiseInput::operator=(const FittedColumnwiseInput& other)
    {
        if (this != &other)
        {
            maps_ = clone_ptr_vec(other.maps_);
        }
        return *this;
    }

    std::expected<void, GprError> FittedColumnwiseInput::apply(
        std::span<double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, maps_.size()); !check)
        {
            return check;
        }
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            if (maps_[col] == nullptr)
            {
                return std::unexpected(GprError::invalid_config("columnwise map contains null entry"));
            }
            if (auto res = maps_[col]->apply(x.subspan(col * n_rows, n_rows), n_rows, 1); !res)
            {
                return res;
            }
        }
        return {};
    }

    std::expected<void, GprError> FittedColumnwiseInput::apply_into(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, maps_.size(), out.size()); !check)
        {
            return check;
        }
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            if (maps_[col] == nullptr)
            {
                return std::unexpected(GprError::invalid_config("columnwise map contains null entry"));
            }
            if (auto res = maps_[col]->apply_into(
                    x.subspan(col * n_rows, n_rows), n_rows, 1, out.subspan(col * n_rows, n_rows)
                );
                !res)
            {
                return res;
            }
        }
        return {};
    }

    std::expected<void, GprError> FittedColumnwiseInput::inverse_apply(
        std::span<double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, maps_.size()); !check)
        {
            return check;
        }
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            if (maps_[col] == nullptr)
            {
                return std::unexpected(GprError::invalid_config("columnwise map contains null entry"));
            }
            if (auto res = maps_[col]->inverse_apply(x.subspan(col * n_rows, n_rows), n_rows, 1); !res)
            {
                return res;
            }
        }
        return {};
    }

    std::expected<void, GprError> FittedColumnwiseInput::inverse_apply_into(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, maps_.size(), out.size()); !check)
        {
            return check;
        }
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            if (maps_[col] == nullptr)
            {
                return std::unexpected(GprError::invalid_config("columnwise map contains null entry"));
            }
            if (auto res = maps_[col]->inverse_apply_into(
                    x.subspan(col * n_rows, n_rows), n_rows, 1, out.subspan(col * n_rows, n_rows)
                );
                !res)
            {
                return res;
            }
        }
        return {};
    }

    std::unique_ptr<FittedInputTransform> FittedColumnwiseInput::clone() const
    {
        return std::make_unique<FittedColumnwiseInput>(*this);
    }

    // --- ColumnwiseInput ---

    ColumnwiseInput::ColumnwiseInput(std::vector<std::unique_ptr<UnfittedInputTransform>> maps) noexcept :
        maps_(std::move(maps))
    {
    }

    ColumnwiseInput::ColumnwiseInput(const ColumnwiseInput& other) : maps_(clone_ptr_vec(other.maps_)) {}

    ColumnwiseInput& ColumnwiseInput::operator=(const ColumnwiseInput& other)
    {
        if (this != &other)
        {
            maps_ = clone_ptr_vec(other.maps_);
        }
        return *this;
    }

    ColumnwiseInput& ColumnwiseInput::then(const UnfittedInputTransform& map) &
    {
        maps_.push_back(map.clone());
        return *this;
    }

    ColumnwiseInput&& ColumnwiseInput::then(const UnfittedInputTransform& map) &&
    {
        maps_.push_back(map.clone());
        return std::move(*this);
    }

    ColumnwiseInput& ColumnwiseInput::then(std::unique_ptr<UnfittedInputTransform> map) &
    {
        if (map != nullptr)
        {
            maps_.push_back(std::move(map));
        }
        return *this;
    }

    ColumnwiseInput&& ColumnwiseInput::then(std::unique_ptr<UnfittedInputTransform> map) &&
    {
        if (map != nullptr)
        {
            maps_.push_back(std::move(map));
        }
        return std::move(*this);
    }

    std::expected<FittedColumnwiseInput, GprError> ColumnwiseInput::fit_typed(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, maps_.size()); !check)
        {
            return std::unexpected(std::move(check).error());
        }
        std::vector<std::unique_ptr<FittedInputTransform>> fitted;
        fitted.reserve(maps_.size());
        for (std::size_t col = 0; col < n_cols; ++col)
        {
            if (maps_[col] == nullptr)
            {
                return std::unexpected(GprError::invalid_config("columnwise map contains null entry"));
            }
            auto fitted_col = maps_[col]->fit(x.subspan(col * n_rows, n_rows), n_rows, 1);
            if (!fitted_col)
            {
                return std::unexpected(std::move(fitted_col).error());
            }
            fitted.push_back(std::move(*fitted_col));
        }
        return FittedColumnwiseInput(std::move(fitted));
    }

    std::expected<std::unique_ptr<FittedInputTransform>, GprError> ColumnwiseInput::fit(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        auto typed = fit_typed(x, n_rows, n_cols);
        if (!typed)
        {
            return std::unexpected(std::move(typed).error());
        }
        return std::make_unique<FittedColumnwiseInput>(std::move(*typed));
    }

    std::unique_ptr<UnfittedInputTransform> ColumnwiseInput::clone() const
    {
        return std::make_unique<ColumnwiseInput>(*this);
    }

}  // namespace cxxgpr::transform
