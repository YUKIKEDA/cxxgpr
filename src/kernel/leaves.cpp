// SPDX-License-Identifier: Apache-2.0
/// @file leaves.cpp
/// @brief Implementation of built-in covariance kernel leaf factories and parameter accessors.
/// @details Implements positive interval validation, unconstrained log-space parameter extraction,
/// and atomic bounded parameter updates with rollback-safe staging across all 10 built-in leaves.
#include "cxxgpr/kernel/leaves.hpp"

#include "cxxgpr/core/checked.hpp"

#include <cmath>
#include <utility>

namespace cxxgpr
{

    namespace
    {

        /// @brief Validates that an open interval is strictly positive (0 < lo < hi).
        /// @details Delegates to Interval::positive() so log(value) remains well-defined over (lo, hi).
        /// @param[in] interval Candidate interval to verify.
        /// @return Strictly positive Interval or GprError on non-positive or inverted bounds.
        /// @retval GprErrorCode::InvalidInterval When lo <= 0, lo >= hi, or either endpoint is non-finite.
        [[nodiscard]] std::expected<Interval, GprError> require_positive_interval(Interval interval)
        {
            return Interval::positive(interval.lo(), interval.hi());
        }

        /// @brief Constructs a strictly positive BoundedParam inside the given positive interval.
        /// @details Rejects invalid intervals before checking open interior membership of value.
        /// @param[in] value Candidate parameter value in user units.
        /// @param[in] interval Candidate bounding interval.
        /// @return Validated BoundedParam or GprError on invalid interval or out-of-range value.
        /// @retval GprErrorCode::InvalidInterval When interval is non-positive or excludes value.
        [[nodiscard]] std::expected<BoundedParam, GprError> make_positive_param(double value, Interval interval)
        {
            // Enforce strict positivity (lo > 0) before testing interior membership of value.
            auto pos_iv = require_positive_interval(interval);
            if (!pos_iv.has_value())
            {
                return std::unexpected(std::move(pos_iv).error());
            }
            return BoundedParam::create(value, *pos_iv);
        }

        /// @brief Validates and wraps vectors of positive values and matching open intervals.
        /// @details Requires non-empty values and equal-length intervals before building each BoundedParam.
        /// @param[in] values Parameter values in user units.
        /// @param[in] intervals Matching open bounding intervals.
        /// @return Vector of validated BoundedParam objects or GprError on first invalid entry.
        /// @retval GprErrorCode::EmptyInput When values is empty.
        /// @retval GprErrorCode::LengthMismatch When intervals.size() != values.size().
        /// @retval GprErrorCode::InvalidInterval When any interval or value fails strict positive checks.
        [[nodiscard]] std::expected<std::vector<BoundedParam>, GprError> make_positive_params(
            std::span<const double> values, std::span<const Interval> intervals
        )
        {
            // Check vector non-emptiness and length match before per-element validation.
            if (auto ne = require_nonempty(values.size()); !ne.has_value())
            {
                return std::unexpected(std::move(ne).error());
            }
            if (auto cnt = require_count(intervals.size(), values.size(), "intervals"); !cnt.has_value())
            {
                return std::unexpected(std::move(cnt).error());
            }
            std::vector<BoundedParam> out;
            out.reserve(values.size());
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                auto param = make_positive_param(values[i], intervals[i]);
                if (!param.has_value())
                {
                    return std::unexpected(std::move(param).error());
                }
                out.push_back(*param);
            }
            return out;
        }

        /// @brief Validates and wraps positive parameter values against Interval::default_positive().
        /// @details Requires a non-empty value span and checks every entry against (1e-5, 1e5).
        /// @param[in] values Parameter values in user units.
        /// @return Vector of validated BoundedParam objects or GprError on empty or out-of-range input.
        /// @retval GprErrorCode::EmptyInput When values is empty.
        /// @retval GprErrorCode::InvalidInterval When any value lies outside (1e-5, 1e5).
        [[nodiscard]] std::expected<std::vector<BoundedParam>, GprError> make_default_positive_params(
            std::span<const double> values
        )
        {
            if (auto ne = require_nonempty(values.size()); !ne.has_value())
            {
                return std::unexpected(std::move(ne).error());
            }
            std::vector<BoundedParam> out;
            out.reserve(values.size());
            for (double val : values)
            {
                auto param = BoundedParam::default_positive(val);
                if (!param.has_value())
                {
                    return std::unexpected(std::move(param).error());
                }
                out.push_back(*param);
            }
            return out;
        }

        /// @brief Converts an unconstrained log-space optimizer parameter back to user units and validates bounds.
        /// @details Checks finiteness of log_value, guards exp(log_value) against overflow to +infinity,
        /// and verifies strict containment inside current.interval() without mutating current.
        /// @param[in] current Existing BoundedParam holding the target open interval.
        /// @param[in] log_value Candidate parameter in natural log space.
        /// @return New BoundedParam with value exp(log_value) or GprError on failure.
        /// @retval GprErrorCode::NonFiniteInput When log_value is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When exp(log_value) overflows or falls outside current.interval().
        [[nodiscard]] std::expected<BoundedParam, GprError> update_param_from_log(
            const BoundedParam& current, double log_value
        )
        {
            // Distinguish non-finite optimizer steps from finite steps whose exp() exceeds bounds.
            if (!std::isfinite(log_value))
            {
                return std::unexpected(GprError::non_finite_input());
            }
            const double val = std::exp(log_value);
            if (!std::isfinite(val))
            {
                return std::unexpected(
                    GprError::invalid_interval_out_of_range(val, current.interval().lo(), current.interval().hi())
                );
            }
            return current.with_value(val);
        }

        /// @brief Writes the natural logarithm of a single BoundedParam into a length-1 span.
        /// @details Verifies out.size() == 1 before writing param.ln().
        /// @param[in] param Source bounded parameter.
        /// @param[out] out Destination span of length 1.
        /// @return Empty expected on success or GprError on length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
        [[nodiscard]] std::expected<void, GprError> get_single_param(const BoundedParam& param, std::span<double> out)
        {
            if (auto cnt = require_count(out.size(), 1U, "out"); !cnt.has_value())
            {
                return std::unexpected(std::move(cnt).error());
            }
            out[0] = param.ln();
            return {};
        }

        /// @brief Atomically updates a single BoundedParam from a length-1 log-space span.
        /// @details Computes and validates the replacement BoundedParam before assigning param.
        /// @param[in,out] param Target bounded parameter to update on success.
        /// @param[in] params Input span of length 1 containing the candidate log-space parameter.
        /// @return Empty expected on success or GprError on invalid count, non-finite input, or bound error.
        /// @retval GprErrorCode::LengthMismatch When params.size() != 1.
        /// @retval GprErrorCode::NonFiniteInput When params[0] is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When exp(params[0]) overflows or leaves param.interval().
        [[nodiscard]] std::expected<void, GprError> set_single_param(
            BoundedParam& param, std::span<const double> params
        )
        {
            if (auto cnt = require_count(params.size(), 1U, "params"); !cnt.has_value())
            {
                return std::unexpected(std::move(cnt).error());
            }
            // Validate into a temporary so param stays intact on failure.
            auto next = update_param_from_log(param, params[0]);
            if (!next.has_value())
            {
                return std::unexpected(std::move(next).error());
            }
            param = *next;
            return {};
        }

        /// @brief Writes the open interval of a single BoundedParam into a length-1 span.
        /// @details Verifies out.size() == 1 before writing param.interval().
        /// @param[in] param Source bounded parameter.
        /// @param[out] out Destination span of length 1.
        /// @return Empty expected on success or GprError on length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
        [[nodiscard]] std::expected<void, GprError> get_single_bound(const BoundedParam& param, std::span<Interval> out)
        {
            if (auto cnt = require_count(out.size(), 1U, "out"); !cnt.has_value())
            {
                return std::unexpected(std::move(cnt).error());
            }
            out[0] = param.interval();
            return {};
        }

        /// @brief Extracts user-unit scalar values from a span of BoundedParam objects.
        /// @details Preserves dimension ordering for ARD kernel accessors.
        /// @param[in] params Source span of bounded parameters.
        /// @return Vector of user-unit values.
        [[nodiscard]] std::vector<double> extract_values(std::span<const BoundedParam> params)
        {
            std::vector<double> values;
            values.reserve(params.size());
            for (const auto& param : params)
            {
                values.push_back(param.value());
            }
            return values;
        }

        /// @brief Extracts open bounding intervals from a span of BoundedParam objects.
        /// @details Preserves dimension ordering for ARD kernel accessors.
        /// @param[in] params Source span of bounded parameters.
        /// @return Vector of open Interval descriptors.
        [[nodiscard]] std::vector<Interval> extract_intervals(std::span<const BoundedParam> params)
        {
            std::vector<Interval> intervals;
            intervals.reserve(params.size());
            for (const auto& param : params)
            {
                intervals.push_back(param.interval());
            }
            return intervals;
        }

    }  // namespace

    // --- RbfKernel ---

    /// @brief Constructs an RBF kernel with lengthscale inside Interval::default_positive().
    /// @details Delegates to create(double, Interval) with Interval::default_positive().
    /// @param[in] lengthscale Positive lengthscale in user units.
    /// @return Validated RbfKernel or GprError if out of range.
    /// @retval GprErrorCode::InvalidInterval When lengthscale is non-finite or outside (1e-5, 1e5).
    std::expected<RbfKernel, GprError> RbfKernel::create(double lengthscale)
    {
        return create(lengthscale, Interval::default_positive());
    }

    /// @brief Constructs an RBF kernel with lengthscale strictly inside the given positive interval.
    /// @details Validates strict positivity of interval and open containment of lengthscale.
    /// @param[in] lengthscale Positive lengthscale in user units.
    /// @param[in] interval Strictly positive open bounding interval.
    /// @return Validated RbfKernel or GprError on invalid interval or value.
    /// @retval GprErrorCode::InvalidInterval When interval is non-positive or excludes lengthscale.
    std::expected<RbfKernel, GprError> RbfKernel::create(double lengthscale, Interval interval)
    {
        auto param = make_positive_param(lengthscale, interval);
        if (!param.has_value())
        {
            return std::unexpected(std::move(param).error());
        }
        return RbfKernel(*param);
    }

    /// @brief Writes log(lengthscale) into out[0].
    /// @details Delegates to get_single_param().
    /// @param[out] out Destination span of length 1.
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
    std::expected<void, GprError> RbfKernel::get_params(std::span<double> out) const
    {
        return get_single_param(lengthscale_, out);
    }

    /// @brief Updates lengthscale from log-space params[0], failing atomically if out of bounds.
    /// @details Delegates to set_single_param() so lengthscale_ is unchanged on validation failure.
    /// @param[in] params Input span of length 1 containing log(lengthscale).
    /// @return Empty expected on success or GprError on failure.
    /// @retval GprErrorCode::LengthMismatch When params.size() != 1.
    /// @retval GprErrorCode::NonFiniteInput When params[0] is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When exp(params[0]) overflows or falls outside interval().
    std::expected<void, GprError> RbfKernel::set_params(std::span<const double> params)
    {
        return set_single_param(lengthscale_, params);
    }

    /// @brief Writes the open interval into out[0].
    /// @details Delegates to get_single_bound().
    /// @param[out] out Destination span of length 1.
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
    std::expected<void, GprError> RbfKernel::bounds_into(std::span<Interval> out) const
    {
        return get_single_bound(lengthscale_, out);
    }

    // --- RbfArdKernel ---

    /// @brief Constructs an ARD RBF kernel with positive lengthscales inside Interval::default_positive().
    /// @details Validates every dimension against (1e-5, 1e5).
    /// @param[in] lengthscales Non-empty span of per-dimension lengthscales in user units.
    /// @return Validated RbfArdKernel or GprError on empty or invalid input.
    /// @retval GprErrorCode::EmptyInput When lengthscales is empty.
    /// @retval GprErrorCode::InvalidInterval When any lengthscale lies outside default bounds.
    std::expected<RbfArdKernel, GprError> RbfArdKernel::create(std::span<const double> lengthscales)
    {
        auto params = make_default_positive_params(lengthscales);
        if (!params.has_value())
        {
            return std::unexpected(std::move(params).error());
        }
        return RbfArdKernel(std::move(*params));
    }

    /// @brief Constructs an ARD RBF kernel with per-dimension open intervals.
    /// @details Validates non-empty matching spans and strict positivity of each interval.
    /// @param[in] lengthscales Non-empty span of per-dimension lengthscales in user units.
    /// @param[in] intervals Matching span of strictly positive open intervals.
    /// @return Validated RbfArdKernel or GprError on empty, mismatched, or invalid inputs.
    /// @retval GprErrorCode::EmptyInput When lengthscales is empty.
    /// @retval GprErrorCode::LengthMismatch When intervals.size() != lengthscales.size().
    /// @retval GprErrorCode::InvalidInterval When any interval or lengthscale fails checks.
    std::expected<RbfArdKernel, GprError> RbfArdKernel::create(
        std::span<const double> lengthscales, std::span<const Interval> intervals
    )
    {
        auto params = make_positive_params(lengthscales, intervals);
        if (!params.has_value())
        {
            return std::unexpected(std::move(params).error());
        }
        return RbfArdKernel(std::move(*params));
    }

    /// @brief Returns the lengthscale values in user units.
    /// @details Extracts current values in dimension order.
    /// @return Vector of lengthscale values.
    std::vector<double> RbfArdKernel::lengthscales() const
    {
        return extract_values(lengthscales_);
    }

    /// @brief Returns the open intervals bounding each lengthscale.
    /// @details Extracts bounding intervals in dimension order.
    /// @return Vector of open Interval descriptors.
    std::vector<Interval> RbfArdKernel::intervals() const
    {
        return extract_intervals(lengthscales_);
    }

    /// @brief Writes log(lengthscales) into out.
    /// @details Populates out[i] = log(lengthscale_i) after verifying span length.
    /// @param[out] out Destination span of length num_params().
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
    std::expected<void, GprError> RbfArdKernel::get_params(std::span<double> out) const
    {
        if (auto cnt = require_count(out.size(), lengthscales_.size(), "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        for (std::size_t i = 0; i < lengthscales_.size(); ++i)
        {
            out[i] = lengthscales_[i].ln();
        }
        return {};
    }

    /// @brief Atomically updates all lengthscales from log-space parameters.
    /// @details Builds a complete staging vector before moving into lengthscales_ so any validation
    /// error rolls back without partial mutation.
    /// @param[in] params Input span of log-space lengthscales of length num_params().
    /// @return Empty expected on success or GprError on validation failure.
    /// @retval GprErrorCode::LengthMismatch When params.size() != num_params().
    /// @retval GprErrorCode::NonFiniteInput When any element of params is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When any exp(params[i]) overflows or exits its interval.
    std::expected<void, GprError> RbfArdKernel::set_params(std::span<const double> params)
    {
        if (auto cnt = require_count(params.size(), lengthscales_.size(), "params"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        // Stage all updated parameters first so any out-of-range dimension leaves lengthscales_ unchanged.
        std::vector<BoundedParam> updated;
        updated.reserve(lengthscales_.size());
        for (std::size_t i = 0; i < lengthscales_.size(); ++i)
        {
            auto next = update_param_from_log(lengthscales_[i], params[i]);
            if (!next.has_value())
            {
                return std::unexpected(std::move(next).error());
            }
            updated.push_back(*next);
        }
        lengthscales_ = std::move(updated);
        return {};
    }

    /// @brief Writes per-dimension open intervals into out.
    /// @details Populates out[i] after checking span length.
    /// @param[out] out Destination span of length num_params().
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
    std::expected<void, GprError> RbfArdKernel::bounds_into(std::span<Interval> out) const
    {
        if (auto cnt = require_count(out.size(), lengthscales_.size(), "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        for (std::size_t i = 0; i < lengthscales_.size(); ++i)
        {
            out[i] = lengthscales_[i].interval();
        }
        return {};
    }

    // --- MaternKernel ---

    /// @brief Constructs a Matern kernel with lengthscale inside Interval::default_positive().
    /// @details Delegates to create(double, MaternNu, Interval) with Interval::default_positive().
    /// @param[in] lengthscale Positive lengthscale in user units.
    /// @param[in] nu Discrete Matern smoothness order.
    /// @return Validated MaternKernel or GprError if lengthscale is out of range.
    /// @retval GprErrorCode::InvalidInterval When lengthscale is non-finite or outside (1e-5, 1e5).
    std::expected<MaternKernel, GprError> MaternKernel::create(double lengthscale, MaternNu nu)
    {
        return create(lengthscale, nu, Interval::default_positive());
    }

    /// @brief Constructs a Matern kernel with lengthscale inside the given positive interval.
    /// @details Validates strict positivity of interval and open containment of lengthscale.
    /// @param[in] lengthscale Positive lengthscale in user units.
    /// @param[in] nu Discrete Matern smoothness order.
    /// @param[in] interval Strictly positive open bounding interval.
    /// @return Validated MaternKernel or GprError on invalid interval or value.
    /// @retval GprErrorCode::InvalidInterval When interval is non-positive or excludes lengthscale.
    std::expected<MaternKernel, GprError> MaternKernel::create(double lengthscale, MaternNu nu, Interval interval)
    {
        auto param = make_positive_param(lengthscale, interval);
        if (!param.has_value())
        {
            return std::unexpected(std::move(param).error());
        }
        return MaternKernel(*param, nu);
    }

    /// @brief Writes log(lengthscale) into out[0].
    /// @details Delegates to get_single_param().
    /// @param[out] out Destination span of length 1.
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
    std::expected<void, GprError> MaternKernel::get_params(std::span<double> out) const
    {
        return get_single_param(lengthscale_, out);
    }

    /// @brief Updates lengthscale from log-space params[0], failing atomically if out of bounds.
    /// @details Delegates to set_single_param().
    /// @param[in] params Input span of length 1 containing log(lengthscale).
    /// @return Empty expected on success or GprError on validation failure.
    /// @retval GprErrorCode::LengthMismatch When params.size() != 1.
    /// @retval GprErrorCode::NonFiniteInput When params[0] is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When exp(params[0]) overflows or leaves interval().
    std::expected<void, GprError> MaternKernel::set_params(std::span<const double> params)
    {
        return set_single_param(lengthscale_, params);
    }

    /// @brief Writes the open interval into out[0].
    /// @details Delegates to get_single_bound().
    /// @param[out] out Destination span of length 1.
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
    std::expected<void, GprError> MaternKernel::bounds_into(std::span<Interval> out) const
    {
        return get_single_bound(lengthscale_, out);
    }

    // --- MaternArdKernel ---

    /// @brief Constructs an ARD Matern kernel with lengthscales inside Interval::default_positive().
    /// @details Validates every dimension against (1e-5, 1e5).
    /// @param[in] lengthscales Non-empty span of per-dimension lengthscales in user units.
    /// @param[in] nu Discrete Matern smoothness order.
    /// @return Validated MaternArdKernel or GprError on empty or invalid input.
    /// @retval GprErrorCode::EmptyInput When lengthscales is empty.
    /// @retval GprErrorCode::InvalidInterval When any lengthscale lies outside default bounds.
    std::expected<MaternArdKernel, GprError> MaternArdKernel::create(std::span<const double> lengthscales, MaternNu nu)
    {
        auto params = make_default_positive_params(lengthscales);
        if (!params.has_value())
        {
            return std::unexpected(std::move(params).error());
        }
        return MaternArdKernel(std::move(*params), nu);
    }

    /// @brief Constructs an ARD Matern kernel with per-dimension open intervals.
    /// @details Validates non-empty matching spans and strict positivity of each interval.
    /// @param[in] lengthscales Non-empty span of per-dimension lengthscales in user units.
    /// @param[in] nu Discrete Matern smoothness order.
    /// @param[in] intervals Matching span of strictly positive open intervals.
    /// @return Validated MaternArdKernel or GprError on empty, mismatched, or invalid inputs.
    /// @retval GprErrorCode::EmptyInput When lengthscales is empty.
    /// @retval GprErrorCode::LengthMismatch When intervals.size() != lengthscales.size().
    /// @retval GprErrorCode::InvalidInterval When any interval or lengthscale fails checks.
    std::expected<MaternArdKernel, GprError> MaternArdKernel::create(
        std::span<const double> lengthscales, MaternNu nu, std::span<const Interval> intervals
    )
    {
        auto params = make_positive_params(lengthscales, intervals);
        if (!params.has_value())
        {
            return std::unexpected(std::move(params).error());
        }
        return MaternArdKernel(std::move(*params), nu);
    }

    /// @brief Returns the lengthscale values in user units.
    /// @details Extracts current values in dimension order.
    /// @return Vector of lengthscale values.
    std::vector<double> MaternArdKernel::lengthscales() const
    {
        return extract_values(lengthscales_);
    }

    /// @brief Returns the open intervals bounding each lengthscale.
    /// @details Extracts bounding intervals in dimension order.
    /// @return Vector of open Interval descriptors.
    std::vector<Interval> MaternArdKernel::intervals() const
    {
        return extract_intervals(lengthscales_);
    }

    /// @brief Writes log(lengthscales) into out.
    /// @details Populates out[i] = log(lengthscale_i) after verifying span length.
    /// @param[out] out Destination span of length num_params().
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
    std::expected<void, GprError> MaternArdKernel::get_params(std::span<double> out) const
    {
        if (auto cnt = require_count(out.size(), lengthscales_.size(), "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        for (std::size_t i = 0; i < lengthscales_.size(); ++i)
        {
            out[i] = lengthscales_[i].ln();
        }
        return {};
    }

    /// @brief Atomically updates all lengthscales from log-space parameters.
    /// @details Stages all validated updates in a temporary vector before replacing lengthscales_.
    /// @param[in] params Input span of log-space lengthscales of length num_params().
    /// @return Empty expected on success or GprError on validation failure.
    /// @retval GprErrorCode::LengthMismatch When params.size() != num_params().
    /// @retval GprErrorCode::NonFiniteInput When any element of params is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When any exp(params[i]) overflows or exits its interval.
    std::expected<void, GprError> MaternArdKernel::set_params(std::span<const double> params)
    {
        if (auto cnt = require_count(params.size(), lengthscales_.size(), "params"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        // Stage all updated parameters first so any error preserves all previous lengthscales.
        std::vector<BoundedParam> updated;
        updated.reserve(lengthscales_.size());
        for (std::size_t i = 0; i < lengthscales_.size(); ++i)
        {
            auto next = update_param_from_log(lengthscales_[i], params[i]);
            if (!next.has_value())
            {
                return std::unexpected(std::move(next).error());
            }
            updated.push_back(*next);
        }
        lengthscales_ = std::move(updated);
        return {};
    }

    /// @brief Writes per-dimension open intervals into out.
    /// @details Populates out[i] after checking span length.
    /// @param[out] out Destination span of length num_params().
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
    std::expected<void, GprError> MaternArdKernel::bounds_into(std::span<Interval> out) const
    {
        if (auto cnt = require_count(out.size(), lengthscales_.size(), "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        for (std::size_t i = 0; i < lengthscales_.size(); ++i)
        {
            out[i] = lengthscales_[i].interval();
        }
        return {};
    }

    // --- PeriodicKernel ---

    /// @brief Constructs a periodic kernel with default positive intervals.
    /// @details Validates both lengthscale and period against Interval::default_positive().
    /// @param[in] lengthscale Positive lengthscale in user units.
    /// @param[in] period Positive spatial period in user units.
    /// @return Validated PeriodicKernel or GprError on out-of-range parameters.
    /// @retval GprErrorCode::InvalidInterval When lengthscale or period lies outside (1e-5, 1e5).
    std::expected<PeriodicKernel, GprError> PeriodicKernel::create(double lengthscale, double period)
    {
        return create(lengthscale, period, Interval::default_positive(), Interval::default_positive());
    }

    /// @brief Constructs a periodic kernel with explicit intervals for lengthscale and period.
    /// @details Validates strict positivity of both intervals and containment of both values.
    /// @param[in] lengthscale Positive lengthscale in user units.
    /// @param[in] period Positive spatial period in user units.
    /// @param[in] lengthscale_interval Strictly positive open interval for lengthscale.
    /// @param[in] period_interval Strictly positive open interval for period.
    /// @return Validated PeriodicKernel or GprError on invalid interval or value.
    /// @retval GprErrorCode::InvalidInterval When either interval or value fails validation.
    std::expected<PeriodicKernel, GprError> PeriodicKernel::create(
        double lengthscale, double period, Interval lengthscale_interval, Interval period_interval
    )
    {
        auto ls = make_positive_param(lengthscale, lengthscale_interval);
        if (!ls.has_value())
        {
            return std::unexpected(std::move(ls).error());
        }
        auto p = make_positive_param(period, period_interval);
        if (!p.has_value())
        {
            return std::unexpected(std::move(p).error());
        }
        return PeriodicKernel(*ls, *p);
    }

    /// @brief Writes [log(lengthscale), log(period)] into out.
    /// @details Verifies out.size() == 2 before populating log-space values.
    /// @param[out] out Destination span of length 2.
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 2.
    std::expected<void, GprError> PeriodicKernel::get_params(std::span<double> out) const
    {
        if (auto cnt = require_count(out.size(), 2U, "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        out[0] = lengthscale_.ln();
        out[1] = period_.ln();
        return {};
    }

    /// @brief Atomically updates lengthscale and period from log-space parameters.
    /// @details Validates both log parameters into temporaries before committing either member.
    /// @param[in] params Input span of length 2 ordered [log(lengthscale), log(period)].
    /// @return Empty expected on success or GprError on validation failure.
    /// @retval GprErrorCode::LengthMismatch When params.size() != 2.
    /// @retval GprErrorCode::NonFiniteInput When any parameter is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When any exp(param) overflows or leaves its interval.
    std::expected<void, GprError> PeriodicKernel::set_params(std::span<const double> params)
    {
        if (auto cnt = require_count(params.size(), 2U, "params"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        // Validate both parameters before committing so an invalid period never leaves lengthscale mutated.
        auto next_ls = update_param_from_log(lengthscale_, params[0]);
        if (!next_ls.has_value())
        {
            return std::unexpected(std::move(next_ls).error());
        }
        auto next_p = update_param_from_log(period_, params[1]);
        if (!next_p.has_value())
        {
            return std::unexpected(std::move(next_p).error());
        }
        lengthscale_ = *next_ls;
        period_ = *next_p;
        return {};
    }

    /// @brief Writes [lengthscale_interval, period_interval] into out.
    /// @details Verifies out.size() == 2 before writing intervals.
    /// @param[out] out Destination span of length 2.
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 2.
    std::expected<void, GprError> PeriodicKernel::bounds_into(std::span<Interval> out) const
    {
        if (auto cnt = require_count(out.size(), 2U, "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        out[0] = lengthscale_.interval();
        out[1] = period_.interval();
        return {};
    }

    // --- RationalQuadraticKernel ---

    /// @brief Constructs an isotropic RQ kernel with default positive intervals.
    /// @details Validates both lengthscale and alpha against Interval::default_positive().
    /// @param[in] lengthscale Positive lengthscale in user units.
    /// @param[in] alpha Positive scale-mixture shape parameter.
    /// @return Validated RationalQuadraticKernel or GprError on out-of-range parameters.
    /// @retval GprErrorCode::InvalidInterval When lengthscale or alpha lies outside (1e-5, 1e5).
    std::expected<RationalQuadraticKernel, GprError> RationalQuadraticKernel::create(double lengthscale, double alpha)
    {
        return create(lengthscale, alpha, Interval::default_positive(), Interval::default_positive());
    }

    /// @brief Constructs an isotropic RQ kernel with explicit intervals for lengthscale and alpha.
    /// @details Validates strict positivity of both intervals and containment of both values.
    /// @param[in] lengthscale Positive lengthscale in user units.
    /// @param[in] alpha Positive scale-mixture shape parameter.
    /// @param[in] lengthscale_interval Strictly positive open interval for lengthscale.
    /// @param[in] alpha_interval Strictly positive open interval for alpha.
    /// @return Validated RationalQuadraticKernel or GprError on invalid interval or value.
    /// @retval GprErrorCode::InvalidInterval When either interval or value fails validation.
    std::expected<RationalQuadraticKernel, GprError> RationalQuadraticKernel::create(
        double lengthscale, double alpha, Interval lengthscale_interval, Interval alpha_interval
    )
    {
        auto ls = make_positive_param(lengthscale, lengthscale_interval);
        if (!ls.has_value())
        {
            return std::unexpected(std::move(ls).error());
        }
        auto a = make_positive_param(alpha, alpha_interval);
        if (!a.has_value())
        {
            return std::unexpected(std::move(a).error());
        }
        return RationalQuadraticKernel(*ls, *a);
    }

    /// @brief Writes [log(lengthscale), log(alpha)] into out.
    /// @details Verifies out.size() == 2 before populating log-space values.
    /// @param[out] out Destination span of length 2.
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 2.
    std::expected<void, GprError> RationalQuadraticKernel::get_params(std::span<double> out) const
    {
        if (auto cnt = require_count(out.size(), 2U, "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        out[0] = lengthscale_.ln();
        out[1] = alpha_.ln();
        return {};
    }

    /// @brief Atomically updates lengthscale and alpha from log-space parameters.
    /// @details Validates both log parameters into temporaries before committing either member.
    /// @param[in] params Input span of length 2 ordered [log(lengthscale), log(alpha)].
    /// @return Empty expected on success or GprError on validation failure.
    /// @retval GprErrorCode::LengthMismatch When params.size() != 2.
    /// @retval GprErrorCode::NonFiniteInput When any parameter is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When any exp(param) overflows or leaves its interval.
    std::expected<void, GprError> RationalQuadraticKernel::set_params(std::span<const double> params)
    {
        if (auto cnt = require_count(params.size(), 2U, "params"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        // Validate both lengthscale and alpha before mutating either field.
        auto next_ls = update_param_from_log(lengthscale_, params[0]);
        if (!next_ls.has_value())
        {
            return std::unexpected(std::move(next_ls).error());
        }
        auto next_a = update_param_from_log(alpha_, params[1]);
        if (!next_a.has_value())
        {
            return std::unexpected(std::move(next_a).error());
        }
        lengthscale_ = *next_ls;
        alpha_ = *next_a;
        return {};
    }

    /// @brief Writes [lengthscale_interval, alpha_interval] into out.
    /// @details Verifies out.size() == 2 before writing intervals.
    /// @param[out] out Destination span of length 2.
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 2.
    std::expected<void, GprError> RationalQuadraticKernel::bounds_into(std::span<Interval> out) const
    {
        if (auto cnt = require_count(out.size(), 2U, "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        out[0] = lengthscale_.interval();
        out[1] = alpha_.interval();
        return {};
    }

    // --- RationalQuadraticArdKernel ---

    /// @brief Constructs an ARD RQ kernel with default positive intervals.
    /// @details Validates non-empty lengthscales and alpha against Interval::default_positive().
    /// @param[in] lengthscales Non-empty span of per-dimension lengthscales in user units.
    /// @param[in] alpha Positive scale-mixture shape parameter.
    /// @return Validated RationalQuadraticArdKernel or GprError on empty or out-of-range inputs.
    /// @retval GprErrorCode::EmptyInput When lengthscales is empty.
    /// @retval GprErrorCode::InvalidInterval When any lengthscale or alpha lies outside default bounds.
    std::expected<RationalQuadraticArdKernel, GprError> RationalQuadraticArdKernel::create(
        std::span<const double> lengthscales, double alpha
    )
    {
        auto ls = make_default_positive_params(lengthscales);
        if (!ls.has_value())
        {
            return std::unexpected(std::move(ls).error());
        }
        auto a = BoundedParam::default_positive(alpha);
        if (!a.has_value())
        {
            return std::unexpected(std::move(a).error());
        }
        return RationalQuadraticArdKernel(std::move(*ls), *a);
    }

    /// @brief Constructs an ARD RQ kernel with explicit intervals for lengthscales and alpha.
    /// @details Validates non-empty matching spans for lengthscales and strict positivity of all intervals.
    /// @param[in] lengthscales Non-empty span of per-dimension lengthscales in user units.
    /// @param[in] alpha Positive scale-mixture shape parameter.
    /// @param[in] lengthscale_intervals Matching span of strictly positive open intervals.
    /// @param[in] alpha_interval Strictly positive open interval for alpha.
    /// @return Validated RationalQuadraticArdKernel or GprError on empty, mismatched, or invalid inputs.
    /// @retval GprErrorCode::EmptyInput When lengthscales is empty.
    /// @retval GprErrorCode::LengthMismatch When lengthscale_intervals.size() != lengthscales.size().
    /// @retval GprErrorCode::InvalidInterval When any interval or value fails validation.
    std::expected<RationalQuadraticArdKernel, GprError> RationalQuadraticArdKernel::create(
        std::span<const double> lengthscales,
        double alpha,
        std::span<const Interval> lengthscale_intervals,
        Interval alpha_interval
    )
    {
        auto ls = make_positive_params(lengthscales, lengthscale_intervals);
        if (!ls.has_value())
        {
            return std::unexpected(std::move(ls).error());
        }
        auto a = make_positive_param(alpha, alpha_interval);
        if (!a.has_value())
        {
            return std::unexpected(std::move(a).error());
        }
        return RationalQuadraticArdKernel(std::move(*ls), *a);
    }

    /// @brief Returns the lengthscale values in user units.
    /// @details Extracts current values in dimension order.
    /// @return Vector of lengthscale values.
    std::vector<double> RationalQuadraticArdKernel::lengthscales() const
    {
        return extract_values(lengthscales_);
    }

    /// @brief Returns the open intervals bounding each lengthscale.
    /// @details Extracts bounding intervals in dimension order.
    /// @return Vector of open Interval descriptors.
    std::vector<Interval> RationalQuadraticArdKernel::lengthscale_intervals() const
    {
        return extract_intervals(lengthscales_);
    }

    /// @brief Writes [log(lengthscales)..., log(alpha)] into out.
    /// @details Populates log lengthscales followed by log(alpha) at index lengthscales_.size().
    /// @param[out] out Destination span of length num_params().
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
    std::expected<void, GprError> RationalQuadraticArdKernel::get_params(std::span<double> out) const
    {
        if (auto cnt = require_count(out.size(), num_params(), "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        for (std::size_t i = 0; i < lengthscales_.size(); ++i)
        {
            out[i] = lengthscales_[i].ln();
        }
        out[lengthscales_.size()] = alpha_.ln();
        return {};
    }

    /// @brief Atomically updates lengthscales and alpha from log-space parameters.
    /// @details Stages all lengthscales and alpha before committing so an invalid alpha or lengthscale
    /// preserves the previous kernel parameters untouched.
    /// @param[in] params Input span of length num_params() ordered [log(lengthscales)..., log(alpha)].
    /// @return Empty expected on success or GprError on validation failure.
    /// @retval GprErrorCode::LengthMismatch When params.size() != num_params().
    /// @retval GprErrorCode::NonFiniteInput When any parameter is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When any exp(param) overflows or leaves its interval.
    std::expected<void, GprError> RationalQuadraticArdKernel::set_params(std::span<const double> params)
    {
        if (auto cnt = require_count(params.size(), num_params(), "params"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        // Stage both lengthscale vector and alpha before moving either into *this.
        std::vector<BoundedParam> updated;
        updated.reserve(lengthscales_.size());
        for (std::size_t i = 0; i < lengthscales_.size(); ++i)
        {
            auto next = update_param_from_log(lengthscales_[i], params[i]);
            if (!next.has_value())
            {
                return std::unexpected(std::move(next).error());
            }
            updated.push_back(*next);
        }
        auto next_a = update_param_from_log(alpha_, params[lengthscales_.size()]);
        if (!next_a.has_value())
        {
            return std::unexpected(std::move(next_a).error());
        }
        lengthscales_ = std::move(updated);
        alpha_ = *next_a;
        return {};
    }

    /// @brief Writes [lengthscale_intervals..., alpha_interval] into out.
    /// @details Populates lengthscale intervals followed by alpha_interval at index lengthscales_.size().
    /// @param[out] out Destination span of length num_params().
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
    std::expected<void, GprError> RationalQuadraticArdKernel::bounds_into(std::span<Interval> out) const
    {
        if (auto cnt = require_count(out.size(), num_params(), "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        for (std::size_t i = 0; i < lengthscales_.size(); ++i)
        {
            out[i] = lengthscales_[i].interval();
        }
        out[lengthscales_.size()] = alpha_.interval();
        return {};
    }

    // --- ConstantKernel ---

    /// @brief Constructs a constant kernel with value inside Interval::default_positive().
    /// @details Delegates to create(double, Interval) with Interval::default_positive().
    /// @param[in] constant Positive constant amplitude in user units.
    /// @return Validated ConstantKernel or GprError if out of range.
    /// @retval GprErrorCode::InvalidInterval When constant is non-finite or outside (1e-5, 1e5).
    std::expected<ConstantKernel, GprError> ConstantKernel::create(double constant)
    {
        return create(constant, Interval::default_positive());
    }

    /// @brief Constructs a constant kernel with value inside the given positive interval.
    /// @details Validates strict positivity of interval and open containment of constant.
    /// @param[in] constant Positive constant amplitude in user units.
    /// @param[in] interval Strictly positive open bounding interval.
    /// @return Validated ConstantKernel or GprError on invalid interval or value.
    /// @retval GprErrorCode::InvalidInterval When interval is non-positive or excludes constant.
    std::expected<ConstantKernel, GprError> ConstantKernel::create(double constant, Interval interval)
    {
        auto param = make_positive_param(constant, interval);
        if (!param.has_value())
        {
            return std::unexpected(std::move(param).error());
        }
        return ConstantKernel(*param);
    }

    /// @brief Writes log(constant) into out[0].
    /// @details Delegates to get_single_param().
    /// @param[out] out Destination span of length 1.
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
    std::expected<void, GprError> ConstantKernel::get_params(std::span<double> out) const
    {
        return get_single_param(constant_, out);
    }

    /// @brief Updates constant from log-space params[0], failing atomically if out of bounds.
    /// @details Delegates to set_single_param().
    /// @param[in] params Input span of length 1 containing log(constant).
    /// @return Empty expected on success or GprError on validation failure.
    /// @retval GprErrorCode::LengthMismatch When params.size() != 1.
    /// @retval GprErrorCode::NonFiniteInput When params[0] is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When exp(params[0]) overflows or leaves interval().
    std::expected<void, GprError> ConstantKernel::set_params(std::span<const double> params)
    {
        return set_single_param(constant_, params);
    }

    /// @brief Writes the open interval into out[0].
    /// @details Delegates to get_single_bound().
    /// @param[out] out Destination span of length 1.
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
    std::expected<void, GprError> ConstantKernel::bounds_into(std::span<Interval> out) const
    {
        return get_single_bound(constant_, out);
    }

    // --- LinearKernel ---

    /// @brief Constructs a linear kernel with variance inside Interval::default_positive().
    /// @details Delegates to create(double, Interval) with Interval::default_positive().
    /// @param[in] variance Positive slope variance multiplier in user units.
    /// @return Validated LinearKernel or GprError if out of range.
    /// @retval GprErrorCode::InvalidInterval When variance is non-finite or outside (1e-5, 1e5).
    std::expected<LinearKernel, GprError> LinearKernel::create(double variance)
    {
        return create(variance, Interval::default_positive());
    }

    /// @brief Constructs a linear kernel with variance inside the given positive interval.
    /// @details Validates strict positivity of interval and open containment of variance.
    /// @param[in] variance Positive slope variance multiplier in user units.
    /// @param[in] interval Strictly positive open bounding interval.
    /// @return Validated LinearKernel or GprError on invalid interval or value.
    /// @retval GprErrorCode::InvalidInterval When interval is non-positive or excludes variance.
    std::expected<LinearKernel, GprError> LinearKernel::create(double variance, Interval interval)
    {
        auto param = make_positive_param(variance, interval);
        if (!param.has_value())
        {
            return std::unexpected(std::move(param).error());
        }
        return LinearKernel(*param);
    }

    /// @brief Writes log(variance) into out[0].
    /// @details Delegates to get_single_param().
    /// @param[out] out Destination span of length 1.
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
    std::expected<void, GprError> LinearKernel::get_params(std::span<double> out) const
    {
        return get_single_param(variance_, out);
    }

    /// @brief Updates variance from log-space params[0], failing atomically if out of bounds.
    /// @details Delegates to set_single_param().
    /// @param[in] params Input span of length 1 containing log(variance).
    /// @return Empty expected on success or GprError on validation failure.
    /// @retval GprErrorCode::LengthMismatch When params.size() != 1.
    /// @retval GprErrorCode::NonFiniteInput When params[0] is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When exp(params[0]) overflows or leaves interval().
    std::expected<void, GprError> LinearKernel::set_params(std::span<const double> params)
    {
        return set_single_param(variance_, params);
    }

    /// @brief Writes the open interval into out[0].
    /// @details Delegates to get_single_bound().
    /// @param[out] out Destination span of length 1.
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
    std::expected<void, GprError> LinearKernel::bounds_into(std::span<Interval> out) const
    {
        return get_single_bound(variance_, out);
    }

    // --- WhiteKernel ---

    /// @brief Constructs a white-noise kernel with variance inside Interval::default_positive().
    /// @details Delegates to create(double, Interval) with Interval::default_positive().
    /// @param[in] variance Positive noise variance in user units.
    /// @return Validated WhiteKernel or GprError if out of range.
    /// @retval GprErrorCode::InvalidInterval When variance is non-finite or outside (1e-5, 1e5).
    std::expected<WhiteKernel, GprError> WhiteKernel::create(double variance)
    {
        return create(variance, Interval::default_positive());
    }

    /// @brief Constructs a white-noise kernel with variance inside the given positive interval.
    /// @details Validates strict positivity of interval and open containment of variance.
    /// @param[in] variance Positive noise variance in user units.
    /// @param[in] interval Strictly positive open bounding interval.
    /// @return Validated WhiteKernel or GprError on invalid interval or value.
    /// @retval GprErrorCode::InvalidInterval When interval is non-positive or excludes variance.
    std::expected<WhiteKernel, GprError> WhiteKernel::create(double variance, Interval interval)
    {
        auto param = make_positive_param(variance, interval);
        if (!param.has_value())
        {
            return std::unexpected(std::move(param).error());
        }
        return WhiteKernel(*param);
    }

    /// @brief Writes log(variance) into out[0].
    /// @details Delegates to get_single_param().
    /// @param[out] out Destination span of length 1.
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
    std::expected<void, GprError> WhiteKernel::get_params(std::span<double> out) const
    {
        return get_single_param(variance_, out);
    }

    /// @brief Updates variance from log-space params[0], failing atomically if out of bounds.
    /// @details Delegates to set_single_param().
    /// @param[in] params Input span of length 1 containing log(variance).
    /// @return Empty expected on success or GprError on validation failure.
    /// @retval GprErrorCode::LengthMismatch When params.size() != 1.
    /// @retval GprErrorCode::NonFiniteInput When params[0] is NaN or infinite.
    /// @retval GprErrorCode::InvalidInterval When exp(params[0]) overflows or leaves interval().
    std::expected<void, GprError> WhiteKernel::set_params(std::span<const double> params)
    {
        return set_single_param(variance_, params);
    }

    /// @brief Writes the open interval into out[0].
    /// @details Delegates to get_single_bound().
    /// @param[out] out Destination span of length 1.
    /// @return Empty expected on success or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
    std::expected<void, GprError> WhiteKernel::bounds_into(std::span<Interval> out) const
    {
        return get_single_bound(variance_, out);
    }

}  // namespace cxxgpr
