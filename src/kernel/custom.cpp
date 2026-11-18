// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/kernel/custom.hpp"

#include "cxxgpr/core/checked.hpp"

#include <bit>
#include <vector>

namespace cxxgpr
{

    std::expected<CustomKernel, GprError> CustomKernel::create(
        std::unique_ptr<KernelTerm<double>> term_double, std::unique_ptr<KernelTerm<float>> term_float
    )
    {
        if (term_double == nullptr || term_float == nullptr)
        {
            return std::unexpected(GprError::invalid_config("custom kernel requires both double and float terms"));
        }
        if (term_double->num_params() != term_float->num_params())
        {
            return std::unexpected(
                GprError::length_mismatch("custom kernel double and float terms must have equal parameter counts")
            );
        }
        if (term_double->persist_id() != term_float->persist_id())
        {
            return std::unexpected(
                GprError::invalid_config("custom kernel double and float terms must share the same persist_id")
            );
        }
        return CustomKernel(std::move(term_double), std::move(term_float));
    }

    CustomKernel::CustomKernel(const CustomKernel& other) :
        term_double_(other.term_double_->clone()),
        term_float_(other.term_float_->clone())
    {
    }

    CustomKernel& CustomKernel::operator=(const CustomKernel& other)
    {
        if (this != &other)
        {
            term_double_ = other.term_double_->clone();
            term_float_ = other.term_float_->clone();
        }
        return *this;
    }

    std::size_t CustomKernel::num_params() const noexcept
    {
        return term_double_->num_params();
    }

    std::expected<void, GprError> CustomKernel::get_params(std::span<double> out) const
    {
        return term_double_->get_params(out);
    }

    std::expected<void, GprError> CustomKernel::set_params(std::span<const double> params)
    {
        if (auto cnt = require_count(params.size(), num_params(), "params"); !cnt.has_value())
        {
            return cnt;
        }
        auto copy_d = term_double_->clone();
        if (auto res_d = copy_d->set_params(params); !res_d.has_value())
        {
            return res_d;
        }
        auto copy_f = term_float_->clone();
        if (auto res_f = copy_f->set_params(params); !res_f.has_value())
        {
            return res_f;
        }
        term_double_ = std::move(copy_d);
        term_float_ = std::move(copy_f);
        return {};
    }

    std::expected<void, GprError> CustomKernel::bounds_into(std::span<Interval> out) const
    {
        return term_double_->bounds_into(out);
    }

    std::string_view CustomKernel::persist_id() const noexcept
    {
        return term_double_->persist_id();
    }

    std::expected<std::string, GprError> CustomKernel::persist_state_json() const
    {
        return term_double_->persist_state_json();
    }

    bool operator==(const CustomKernel& lhs, const CustomKernel& rhs)
    {
        if (lhs.num_params() != rhs.num_params() || lhs.persist_id() != rhs.persist_id())
        {
            return false;
        }
        const std::size_t np = lhs.num_params();
        std::vector<double> pl(np);
        std::vector<double> pr(np);
        if (!lhs.get_params(pl).has_value() || !rhs.get_params(pr).has_value())
        {
            return false;
        }
        for (std::size_t i = 0U; i < np; ++i)
        {
            if (std::bit_cast<std::uint64_t>(pl[i]) != std::bit_cast<std::uint64_t>(pr[i]))
            {
                return false;
            }
        }
        return true;
    }

}  // namespace cxxgpr
