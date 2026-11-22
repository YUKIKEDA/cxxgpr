// SPDX-License-Identifier: Apache-2.0
/// @file pipeline.cpp
/// @brief Implementations of sequential input and target transform pipelines.
/// @details Implements Pipeline, FittedPipeline, TargetPipeline, and FittedTargetPipeline including forward sequential
/// application, reverse-order inverse transformation, out-of-place zero-allocation execution, and stagewise fitting.

#include "cxxgpr/transform/pipeline.hpp"

#include <algorithm>
#include <cstddef>
#include <expected>
#include <memory>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include "transform/detail.hpp"

namespace cxxgpr::transform
{

    /// @brief Internal friend accessor for iterative pipeline traversal, flattening, and teardown.
    struct PipelineAccess
    {
        /// @brief Returns mutable reference to a pipeline's internal step vector.
        /// @tparam PipeT Concrete pipeline type.
        /// @param[in,out] pipe Target pipeline instance.
        /// @return Reference to `pipe.steps_`.
        template <typename PipeT> static auto& steps(PipeT& pipe) noexcept { return pipe.steps_; }

        /// @brief Returns read-only reference to a pipeline's internal step vector.
        /// @tparam PipeT Concrete pipeline type.
        /// @param[in] pipe Target pipeline instance.
        /// @return Const reference to `pipe.steps_`.
        template <typename PipeT> static const auto& steps(const PipeT& pipe) noexcept { return pipe.steps_; }

        /// @brief Iteratively destroys nested pipeline steps without stack recursion.
        /// @tparam PipeT Concrete pipeline class.
        /// @tparam BaseT Base transform interface stored in `root_steps`.
        /// @param[in,out] root_steps Top-level step vector to drain and destroy.
        template <typename PipeT, typename BaseT>
        static void destroy_steps(std::vector<std::unique_ptr<BaseT>>& root_steps) noexcept
        {
            if (root_steps.empty())
            {
                return;
            }
            try
            {
                std::vector<std::unique_ptr<BaseT>> work;
                auto drain = [&work](std::vector<std::unique_ptr<BaseT>>& vec)
                {
                    for (auto& s : vec)
                    {
                        if (s != nullptr)
                        {
                            work.push_back(std::move(s));
                        }
                    }
                    vec.clear();
                };
                drain(root_steps);
                while (!work.empty())
                {
                    auto cur = std::move(work.back());
                    work.pop_back();
                    if (auto* sub = dynamic_cast<PipeT*>(cur.get()))
                    {
                        drain(sub->steps_);
                    }
                }
            }
            catch (...)
            {
                // Best-effort iterative teardown; remaining links fall back to default vector destruction.
                return;
            }
        }
    };

    namespace
    {

        /// @brief Deep-clones a vector of polymorphic transform unique pointers.
        /// @details Skips null pointers while cloning each transform step into a newly reserved vector.
        /// @tparam Ptr Smart pointer type exposing clone().
        /// @param[in] src Source vector of polymorphic transform pointers.
        /// @return Vector holding deep copies of all non-null elements in src.
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

        /// @brief Iteratively flattens nested fitted pipelines into non-owning leaf steps in forward order.
        /// @tparam PipeT Concrete fitted pipeline class (`FittedPipeline` or `FittedTargetPipeline`).
        /// @tparam BaseT Base fitted transform interface (`FittedInputTransform` or `FittedTargetTransform`).
        /// @param[in] root_steps Top-level fitted step vector.
        /// @param[out] out Destination vector receiving flattened leaf pointers (including nullptr if present).
        template <typename PipeT, typename BaseT>
        void rebuild_flat_steps_impl(
            const std::vector<std::unique_ptr<BaseT>>& root_steps, std::vector<const BaseT*>& out
        ) noexcept
        {
            try
            {
                out.clear();

                struct Frame
                {
                    const std::vector<std::unique_ptr<BaseT>>* steps;
                    std::size_t idx;
                };

                std::vector<Frame> stack;
                stack.push_back(Frame{&root_steps, 0U});
                while (!stack.empty())
                {
                    auto& top = stack.back();
                    if (top.idx >= top.steps->size())
                    {
                        stack.pop_back();
                        continue;
                    }
                    const BaseT* step = (*top.steps)[top.idx++].get();
                    if (const auto* sub = dynamic_cast<const PipeT*>(step))
                    {
                        stack.push_back(Frame{&PipelineAccess::steps(*sub), 0U});
                    }
                    else
                    {
                        out.push_back(step);
                    }
                }
            }
            catch (...)
            {
                out.clear();
            }
        }

    }  // namespace

    // --- FittedPipeline ---

    /// @brief Rebuilds flat_steps_ iteratively from steps_.
    void FittedPipeline::rebuild_flat_steps()
    {
        rebuild_flat_steps_impl<FittedPipeline>(steps_, flat_steps_);
    }

    /// @brief Constructs a fitted input pipeline from an ordered list of fitted steps.
    /// @details Takes ownership of steps in forward application order.
    /// @param[in] steps Ordered vector of fitted input transform steps.
    FittedPipeline::FittedPipeline(std::vector<std::unique_ptr<FittedInputTransform>> steps) noexcept :
        steps_(std::move(steps))
    {
        rebuild_flat_steps();
    }

    /// @brief Deep-copies the fitted input pipeline.
    /// @details Clones each non-null step preserving forward order.
    /// @param[in] other Source FittedPipeline to copy.
    FittedPipeline::FittedPipeline(const FittedPipeline& other) : steps_(clone_ptr_vec(other.steps_))
    {
        rebuild_flat_steps();
    }

    /// @brief Deep-copy assigns the fitted input pipeline.
    /// @details Replaces owned steps with deep clones of other's steps when &other != this.
    /// @param[in] other Source FittedPipeline to copy from.
    /// @return Reference to *this.
    FittedPipeline& FittedPipeline::operator=(const FittedPipeline& other)
    {
        if (this != &other)
        {
            steps_ = clone_ptr_vec(other.steps_);
            rebuild_flat_steps();
        }
        return *this;
    }

    /// @brief Destroys nested fitted pipeline steps iteratively without call-stack recursion.
    FittedPipeline::~FittedPipeline()
    {
        PipelineAccess::destroy_steps<FittedPipeline>(steps_);
    }

    /// @brief Applies all steps in forward order to x in place.
    /// @details Empty pipelines delegate to FittedIdentityInput so validation matches identity behavior.
    /// @param[in,out] x Mutable column-major feature buffer of size n_rows * n_cols.
    /// @param[in] n_rows Number of sample rows.
    /// @param[in] n_cols Number of feature columns.
    /// @return Empty expected on success, or GprError from validation, null step, or step execution.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When any step rejects n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<void, GprError> FittedPipeline::apply(
        std::span<double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        // Delegate empty pipelines to identity so validation semantics remain uniform.
        if (flat_steps_.empty())
        {
            return FittedIdentityInput{}.apply(x, n_rows, n_cols);
        }
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols); !check)
        {
            return check;
        }
        for (const FittedInputTransform* step : flat_steps_)
        {
            if (step == nullptr)
            {
                return std::unexpected(GprError::invalid_config("pipeline contains null step"));
            }
            if (auto res = step->apply(x, n_rows, n_cols); !res)
            {
                return res;
            }
        }
        return {};
    }

    /// @brief Applies all steps in forward order from x into out without allocation.
    /// @details Executes step 0 out of place into out, then executes steps 1..k-1 in place on out.
    /// @param[in] x Read-only column-major feature buffer of size n_rows * n_cols.
    /// @param[in] n_rows Number of sample rows.
    /// @param[in] n_cols Number of feature columns.
    /// @param[out] out Preallocated mutable output buffer of size n_rows * n_cols.
    /// @return Empty expected on success, or GprError from validation, null step, or step execution.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When any step rejects n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<void, GprError> FittedPipeline::apply_into(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
    ) const
    {
        if (flat_steps_.empty())
        {
            return FittedIdentityInput{}.apply_into(x, n_rows, n_cols, out);
        }
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, std::nullopt, out.size()); !check)
        {
            return check;
        }
        // Run step 0 out-of-place into `out`, then mutate `out` in place for subsequent steps (zero allocation).
        if (flat_steps_[0] == nullptr)
        {
            return std::unexpected(GprError::invalid_config("pipeline contains null step"));
        }
        if (auto res = flat_steps_[0]->apply_into(x, n_rows, n_cols, out); !res)
        {
            return res;
        }
        for (std::size_t i = 1; i < flat_steps_.size(); ++i)
        {
            if (flat_steps_[i] == nullptr)
            {
                return std::unexpected(GprError::invalid_config("pipeline contains null step"));
            }
            if (auto res = flat_steps_[i]->apply(out, n_rows, n_cols); !res)
            {
                return res;
            }
        }
        return {};
    }

    /// @brief Applies all steps in reverse order to invert the transform in place.
    /// @details Empty pipelines delegate to FittedIdentityInput; non-empty pipelines traverse steps_ in reverse.
    /// @param[in,out] x Mutable column-major transformed feature buffer of size n_rows * n_cols.
    /// @param[in] n_rows Number of sample rows.
    /// @param[in] n_cols Number of feature columns.
    /// @return Empty expected on success, or GprError from validation, null step, or step execution.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When any step rejects n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<void, GprError> FittedPipeline::inverse_apply(
        std::span<double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        if (flat_steps_.empty())
        {
            return FittedIdentityInput{}.inverse_apply(x, n_rows, n_cols);
        }
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols); !check)
        {
            return check;
        }
        // Invert in reverse composition order: (T_k o ... o T_1)^-1 = T_1^-1 o ... o T_k^-1.
        for (const FittedInputTransform* step : std::ranges::reverse_view(flat_steps_))
        {
            if (step == nullptr)
            {
                return std::unexpected(GprError::invalid_config("pipeline contains null step"));
            }
            if (auto res = step->inverse_apply(x, n_rows, n_cols); !res)
            {
                return res;
            }
        }
        return {};
    }

    /// @brief Applies all steps in reverse order from x into out without allocation.
    /// @details Executes the last step out of place into out via inverse_apply_into and preceding steps in place on
    /// out.
    /// @param[in] x Read-only column-major transformed feature buffer of size n_rows * n_cols.
    /// @param[in] n_rows Number of sample rows.
    /// @param[in] n_cols Number of feature columns.
    /// @param[out] out Preallocated mutable output buffer of size n_rows * n_cols.
    /// @return Empty expected on success, or GprError from validation, null step, or step execution.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() or out.size() differs from n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When any step rejects n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<void, GprError> FittedPipeline::inverse_apply_into(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::span<double> out
    ) const
    {
        if (flat_steps_.empty())
        {
            return FittedIdentityInput{}.inverse_apply_into(x, n_rows, n_cols, out);
        }
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols, std::nullopt, out.size()); !check)
        {
            return check;
        }
        // Start reverse chain out-of-place at the final step, then invert remaining steps in place on `out`.
        const std::size_t last = flat_steps_.size() - 1U;
        if (flat_steps_[last] == nullptr)
        {
            return std::unexpected(GprError::invalid_config("pipeline contains null step"));
        }
        if (auto res = flat_steps_[last]->inverse_apply_into(x, n_rows, n_cols, out); !res)
        {
            return res;
        }
        for (std::size_t idx = last; idx > 0; --idx)
        {
            const FittedInputTransform* step = flat_steps_[idx - 1U];
            if (step == nullptr)
            {
                return std::unexpected(GprError::invalid_config("pipeline contains null step"));
            }
            if (auto res = step->inverse_apply(out, n_rows, n_cols); !res)
            {
                return res;
            }
        }
        return {};
    }

    /// @brief Returns a polymorphic deep copy of this fitted input pipeline.
    /// @details Allocates a new FittedPipeline copy.
    /// @return Unique pointer owning a deep copy of *this.
    std::unique_ptr<FittedInputTransform> FittedPipeline::clone() const
    {
        return std::make_unique<FittedPipeline>(*this);
    }

    // --- Pipeline ---

    /// @brief Constructs an input pipeline from an ordered list of unfitted steps.
    /// @details Takes ownership of steps in forward application order.
    /// @param[in] steps Ordered vector of unfitted input transform steps.
    Pipeline::Pipeline(std::vector<std::unique_ptr<UnfittedInputTransform>> steps) noexcept : steps_(std::move(steps))
    {
    }

    /// @brief Deep-copies the unfitted input pipeline.
    /// @details Clones each non-null step preserving forward order.
    /// @param[in] other Source Pipeline to copy.
    Pipeline::Pipeline(const Pipeline& other) : steps_(clone_ptr_vec(other.steps_)) {}

    /// @brief Deep-copy assigns the unfitted input pipeline.
    /// @details Replaces owned steps with deep clones of other's steps when &other != this.
    /// @param[in] other Source Pipeline to copy from.
    /// @return Reference to *this.
    Pipeline& Pipeline::operator=(const Pipeline& other)
    {
        if (this != &other)
        {
            steps_ = clone_ptr_vec(other.steps_);
        }
        return *this;
    }

    /// @brief Destroys nested unfitted pipeline steps iteratively without call-stack recursion.
    Pipeline::~Pipeline()
    {
        PipelineAccess::destroy_steps<Pipeline>(steps_);
    }

    /// @brief Appends a step by cloning it onto the end of this pipeline.
    /// @details Lvalue overload cloning step onto steps_.
    /// @param[in] step Unfitted input transform to clone and append.
    /// @return Reference to *this.
    Pipeline& Pipeline::then(const UnfittedInputTransform& step) &
    {
        steps_.push_back(step.clone());
        return *this;
    }

    /// @brief Appends a step by cloning it onto the end of this rvalue pipeline.
    /// @details Rvalue overload cloning step onto steps_ and forwarding *this.
    /// @param[in] step Unfitted input transform to clone and append.
    /// @return Rvalue reference to *this.
    Pipeline&& Pipeline::then(const UnfittedInputTransform& step) &&
    {
        steps_.push_back(step.clone());
        return std::move(*this);
    }

    /// @brief Appends an owned step onto the end of this pipeline.
    /// @details Moves non-null step into steps_ and ignores null pointers.
    /// @param[in] step Owned unfitted input transform to append.
    /// @return Reference to *this.
    Pipeline& Pipeline::then(std::unique_ptr<UnfittedInputTransform> step) &
    {
        if (step != nullptr)
        {
            steps_.push_back(std::move(step));
        }
        return *this;
    }

    /// @brief Appends an owned step onto the end of this rvalue pipeline.
    /// @details Moves non-null step into steps_ and forwards *this.
    /// @param[in] step Owned unfitted input transform to append.
    /// @return Rvalue reference to *this.
    Pipeline&& Pipeline::then(std::unique_ptr<UnfittedInputTransform> step) &&
    {
        if (step != nullptr)
        {
            steps_.push_back(std::move(step));
        }
        return std::move(*this);
    }

    /// @brief Fits each step sequentially on the output of the preceding step.
    /// @details Validates training features once upfront, then copies x into a mutable scratch buffer transformed
    /// stagewise after fitting each step.
    /// @param[in] x Read-only column-major training feature matrix.
    /// @param[in] n_rows Number of training sample rows.
    /// @param[in] n_cols Number of feature columns.
    /// @return FittedPipeline on success, or GprError on validation failure, null step, or step fitting error.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When any step rejects n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x or an intermediate output is non-finite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<FittedPipeline, GprError> Pipeline::fit_typed(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        if (auto check = detail::validate_input_matrix(x, n_rows, n_cols); !check)
        {
            return std::unexpected(std::move(check).error());
        }
        if (steps_.empty())
        {
            return FittedPipeline{};
        }
        // Maintain a single mutable working buffer and an explicit frame stack so nested pipelines do not recurse.
        std::vector<double> buf(x.begin(), x.end());

        struct FitFrame
        {
            const std::vector<std::unique_ptr<UnfittedInputTransform>>* steps;
            std::size_t idx;
            std::vector<std::unique_ptr<FittedInputTransform>> fitted;
        };

        std::vector<FitFrame> stack;
        stack.push_back(FitFrame{.steps = &steps_, .idx = 0U, .fitted = {}});
        stack.back().fitted.reserve(steps_.size());

        while (true)
        {
            auto& top = stack.back();
            if (top.idx >= top.steps->size())
            {
                auto finished = std::move(top.fitted);
                stack.pop_back();
                if (stack.empty())
                {
                    return FittedPipeline(std::move(finished));
                }
                stack.back().fitted.push_back(std::make_unique<FittedPipeline>(std::move(finished)));
                continue;
            }
            const auto& step = (*top.steps)[top.idx++];
            if (step == nullptr)
            {
                return std::unexpected(GprError::invalid_config("pipeline contains null step"));
            }
            if (const auto* sub_pipe = dynamic_cast<const Pipeline*>(step.get()))
            {
                const auto& sub_steps = PipelineAccess::steps(*sub_pipe);
                stack.push_back(FitFrame{.steps = &sub_steps, .idx = 0U, .fitted = {}});
                stack.back().fitted.reserve(sub_steps.size());
                continue;
            }
            auto fitted_step = step->fit(buf, n_rows, n_cols);
            if (!fitted_step)
            {
                return std::unexpected(std::move(fitted_step).error());
            }
            if (auto res = (*fitted_step)->apply(buf, n_rows, n_cols); !res)
            {
                return std::unexpected(std::move(res).error());
            }
            top.fitted.push_back(std::move(*fitted_step));
        }
    }

    /// @brief Fits each step sequentially and returns a boxed fitted pipeline.
    /// @details Delegates to fit_typed and boxes the typed result.
    /// @param[in] x Read-only column-major training feature matrix.
    /// @param[in] n_rows Number of training sample rows.
    /// @param[in] n_cols Number of feature columns.
    /// @return Boxed FittedPipeline on success, or GprError on validation failure, null step, or step fitting error.
    /// @retval GprErrorCode::EmptyInput When x, n_rows, or n_cols is empty/zero.
    /// @retval GprErrorCode::SizeOverflow When n_rows * n_cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When x.size() != n_rows * n_cols.
    /// @retval GprErrorCode::DimensionMismatch When any step rejects n_cols.
    /// @retval GprErrorCode::NonFiniteInput When any element of x or an intermediate output is non-finite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<std::unique_ptr<FittedInputTransform>, GprError> Pipeline::fit(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols
    ) const
    {
        auto typed = fit_typed(x, n_rows, n_cols);
        if (!typed)
        {
            return std::unexpected(std::move(typed).error());
        }
        return std::make_unique<FittedPipeline>(std::move(*typed));
    }

    /// @brief Returns a polymorphic deep copy of this unfitted input pipeline.
    /// @details Allocates a new Pipeline copy.
    /// @return Unique pointer owning a deep copy of *this.
    std::unique_ptr<UnfittedInputTransform> Pipeline::clone() const
    {
        return std::make_unique<Pipeline>(*this);
    }

    // --- FittedTargetPipeline ---

    /// @brief Rebuilds flat_steps_ iteratively from steps_.
    void FittedTargetPipeline::rebuild_flat_steps()
    {
        rebuild_flat_steps_impl<FittedTargetPipeline>(steps_, flat_steps_);
    }

    /// @brief Constructs a fitted target pipeline from an ordered list of fitted steps.
    /// @details Takes ownership of steps in forward application order.
    /// @param[in] steps Ordered vector of fitted target transform steps.
    FittedTargetPipeline::FittedTargetPipeline(std::vector<std::unique_ptr<FittedTargetTransform>> steps) noexcept :
        steps_(std::move(steps))
    {
        rebuild_flat_steps();
    }

    /// @brief Deep-copies the fitted target pipeline.
    /// @details Clones each non-null step preserving forward order.
    /// @param[in] other Source FittedTargetPipeline to copy.
    FittedTargetPipeline::FittedTargetPipeline(const FittedTargetPipeline& other) : steps_(clone_ptr_vec(other.steps_))
    {
        rebuild_flat_steps();
    }

    /// @brief Deep-copy assigns the fitted target pipeline.
    /// @details Replaces owned steps with deep clones of other's steps when &other != this.
    /// @param[in] other Source FittedTargetPipeline to copy from.
    /// @return Reference to *this.
    FittedTargetPipeline& FittedTargetPipeline::operator=(const FittedTargetPipeline& other)
    {
        if (this != &other)
        {
            steps_ = clone_ptr_vec(other.steps_);
            rebuild_flat_steps();
        }
        return *this;
    }

    /// @brief Destroys nested fitted target pipeline steps iteratively without call-stack recursion.
    FittedTargetPipeline::~FittedTargetPipeline()
    {
        PipelineAccess::destroy_steps<FittedTargetPipeline>(steps_);
    }

    /// @brief Applies all steps in forward order to y in place.
    /// @details Empty pipelines delegate to FittedIdentityTarget; non-empty pipelines apply steps_[0..k-1]
    /// sequentially.
    /// @param[in,out] y Mutable span of target values updated in place.
    /// @return Empty expected on success, or GprError on empty/non-finite input or null step.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<void, GprError> FittedTargetPipeline::transform(std::span<double> y) const
    {
        if (flat_steps_.empty())
        {
            return FittedIdentityTarget{}.transform(y);
        }
        if (auto check = detail::validate_target_vector(y); !check)
        {
            return check;
        }
        for (const FittedTargetTransform* step : flat_steps_)
        {
            if (step == nullptr)
            {
                return std::unexpected(GprError::invalid_config("target pipeline contains null step"));
            }
            if (auto res = step->transform(y); !res)
            {
                return res;
            }
        }
        return {};
    }

    /// @brief Applies all steps in forward order from y into out without allocation.
    /// @details Executes step 0 out of place into out via transform_into, then executes steps 1..k-1 in place on out.
    /// @param[in] y Read-only span of target values.
    /// @param[out] out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on invalid input, length mismatch, or null step.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != y.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of y is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<void, GprError> FittedTargetPipeline::transform_into(
        std::span<const double> y, std::span<double> out
    ) const
    {
        if (flat_steps_.empty())
        {
            return FittedIdentityTarget{}.transform_into(y, out);
        }
        if (auto check = detail::validate_target_vector(y, out.size()); !check)
        {
            return check;
        }
        // Step 0 writes into `out`, and remaining steps mutate `out` in place without extra buffers.
        if (flat_steps_[0] == nullptr)
        {
            return std::unexpected(GprError::invalid_config("target pipeline contains null step"));
        }
        if (auto res = flat_steps_[0]->transform_into(y, out); !res)
        {
            return res;
        }
        for (std::size_t i = 1; i < flat_steps_.size(); ++i)
        {
            if (flat_steps_[i] == nullptr)
            {
                return std::unexpected(GprError::invalid_config("target pipeline contains null step"));
            }
            if (auto res = flat_steps_[i]->transform(out); !res)
            {
                return res;
            }
        }
        return {};
    }

    namespace
    {

        /// @brief Applies an inverse target operation across non-empty steps in reverse order in place.
        /// @details Validates values before executing fn(*step, values) from the final step back to step 0 so chained
        /// affine/Jacobian scalings compose correctly.
        /// @tparam InPlaceFn Callable invocable as fn(const FittedTargetTransform&, std::span<double>).
        /// @param[in] steps Non-empty vector of fitted target pipeline steps.
        /// @param[in,out] values Mutable prediction or covariance span updated in place.
        /// @param[in] fn Step-wise inverse operation callback.
        /// @return Empty expected on success, or GprError on invalid values, null step, or step failure.
        /// @retval GprErrorCode::EmptyInput When values is empty.
        /// @retval GprErrorCode::NonFiniteInput When any element of values is NaN or infinite.
        /// @retval GprErrorCode::InvalidConfig When any step pointer in steps is null.
        template <typename InPlaceFn>
        std::expected<void, GprError> reverse_target_in_place(
            const std::vector<const FittedTargetTransform*>& steps, std::span<double> values, InPlaceFn&& fn
        )
        {
            if (auto check = detail::validate_target_vector(values); !check)
            {
                return check;
            }
            // Apply inverse mean/variance/covariance maps from outer-most step back to raw target space.
            for (const FittedTargetTransform* step : std::ranges::reverse_view(steps))
            {
                if (step == nullptr)
                {
                    return std::unexpected(GprError::invalid_config("target pipeline contains null step"));
                }
                if (auto res = fn(*step, values); !res)
                {
                    return res;
                }
            }
            return {};
        }

        /// @brief Applies an inverse target operation in reverse order from values into out without allocation.
        /// @details Executes the last step out of place via into_fn and preceding steps in reverse order in place on
        /// out via inplace_fn.
        /// @tparam IntoFn Callable invocable as into_fn(const FittedTargetTransform&, std::span<const double>,
        /// std::span<double>).
        /// @tparam InPlaceFn Callable invocable as inplace_fn(const FittedTargetTransform&, std::span<double>).
        /// @param[in] steps Non-empty vector of fitted target pipeline steps.
        /// @param[in] values Read-only prediction or covariance span.
        /// @param[out] out Preallocated mutable output buffer of matching length.
        /// @param[in] into_fn Out-of-place inverse callback for the last step.
        /// @param[in] inplace_fn In-place inverse callback for preceding steps.
        /// @return Empty expected on success, or GprError on invalid values, length mismatch, null step, or step
        /// failure.
        /// @retval GprErrorCode::EmptyInput When values is empty.
        /// @retval GprErrorCode::LengthMismatch When out.size() != values.size().
        /// @retval GprErrorCode::NonFiniteInput When any element of values is NaN or infinite.
        /// @retval GprErrorCode::InvalidConfig When any step pointer in steps is null.
        template <typename IntoFn, typename InPlaceFn>
        std::expected<void, GprError> reverse_target_into(
            const std::vector<const FittedTargetTransform*>& steps,
            std::span<const double> values,
            std::span<double> out,
            IntoFn&& into_fn,
            InPlaceFn&& inplace_fn
        )
        {
            if (auto check = detail::validate_target_vector(values, out.size()); !check)
            {
                return check;
            }
            // Start reverse chain out-of-place at the final step, then compose earlier inverse steps in place on `out`.
            const std::size_t last = steps.size() - 1U;
            if (steps[last] == nullptr)
            {
                return std::unexpected(GprError::invalid_config("target pipeline contains null step"));
            }
            if (auto res = into_fn(*steps[last], values, out); !res)
            {
                return res;
            }
            for (std::size_t idx = last; idx > 0; --idx)
            {
                const FittedTargetTransform* step = steps[idx - 1U];
                if (step == nullptr)
                {
                    return std::unexpected(GprError::invalid_config("target pipeline contains null step"));
                }
                if (auto res = inplace_fn(*step, out); !res)
                {
                    return res;
                }
            }
            return {};
        }

    }  // namespace

    /// @brief Applies inverse mean transforms in reverse order in place.
    /// @details Empty pipelines delegate to FittedIdentityTarget.
    /// @param[in,out] mean Mutable span of transformed predictive means updated in place.
    /// @return Empty expected on success, or GprError on empty/non-finite input or null step.
    /// @retval GprErrorCode::EmptyInput When mean is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<void, GprError> FittedTargetPipeline::inverse_transform_mean(std::span<double> mean) const
    {
        if (flat_steps_.empty())
        {
            return FittedIdentityTarget{}.inverse_transform_mean(mean);
        }
        return reverse_target_in_place(flat_steps_, mean, [](const FittedTargetTransform& step, std::span<double> buf) {
            return step.inverse_transform_mean(buf);
        });
    }

    /// @brief Applies inverse mean transforms in reverse order from mean into out without allocation.
    /// @details Empty pipelines delegate to FittedIdentityTarget.
    /// @param[in] mean Read-only span of transformed predictive means.
    /// @param[out] out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on invalid input, length mismatch, or null step.
    /// @retval GprErrorCode::EmptyInput When mean is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != mean.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of mean is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<void, GprError> FittedTargetPipeline::inverse_transform_mean_into(
        std::span<const double> mean, std::span<double> out
    ) const
    {
        if (flat_steps_.empty())
        {
            return FittedIdentityTarget{}.inverse_transform_mean_into(mean, out);
        }
        return reverse_target_into(
            flat_steps_, mean,
            out, [](const FittedTargetTransform& step, std::span<const double> src, std::span<double> dst) {
            return step.inverse_transform_mean_into(src, dst);
        }, [](const FittedTargetTransform& step, std::span<double> buf) { return step.inverse_transform_mean(buf); }
        );
    }

    /// @brief Applies inverse variance transforms in reverse order in place.
    /// @details Composes squared derivative scalings across all steps from step k-1 down to 0.
    /// @param[in,out] var Mutable span of transformed predictive variances updated in place.
    /// @return Empty expected on success, or GprError on empty/non-finite input or null step.
    /// @retval GprErrorCode::EmptyInput When var is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<void, GprError> FittedTargetPipeline::inverse_transform_variance(std::span<double> var) const
    {
        if (flat_steps_.empty())
        {
            return FittedIdentityTarget{}.inverse_transform_variance(var);
        }
        return reverse_target_in_place(flat_steps_, var, [](const FittedTargetTransform& step, std::span<double> buf) {
            return step.inverse_transform_variance(buf);
        });
    }

    /// @brief Applies inverse variance transforms in reverse order from var into out without allocation.
    /// @details Empty pipelines delegate to FittedIdentityTarget.
    /// @param[in] var Read-only span of transformed predictive variances.
    /// @param[out] out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on invalid input, length mismatch, or null step.
    /// @retval GprErrorCode::EmptyInput When var is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != var.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of var is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<void, GprError> FittedTargetPipeline::inverse_transform_variance_into(
        std::span<const double> var, std::span<double> out
    ) const
    {
        if (flat_steps_.empty())
        {
            return FittedIdentityTarget{}.inverse_transform_variance_into(var, out);
        }
        return reverse_target_into(
            flat_steps_, var,
            out, [](const FittedTargetTransform& step, std::span<const double> src, std::span<double> dst) {
            return step.inverse_transform_variance_into(src, dst);
        }, [](const FittedTargetTransform& step, std::span<double> buf) { return step.inverse_transform_variance(buf); }
        );
    }

    /// @brief Applies inverse covariance transforms in reverse order in place.
    /// @details Empty pipelines delegate to FittedIdentityTarget.
    /// @param[in,out] cov Mutable flattened m x m covariance buffer updated in place.
    /// @return Empty expected on success, or GprError on empty/non-finite input or null step.
    /// @retval GprErrorCode::EmptyInput When cov is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of cov is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<void, GprError> FittedTargetPipeline::inverse_transform_covariance(std::span<double> cov) const
    {
        if (flat_steps_.empty())
        {
            return FittedIdentityTarget{}.inverse_transform_covariance(cov);
        }
        return reverse_target_in_place(flat_steps_, cov, [](const FittedTargetTransform& step, std::span<double> buf) {
            return step.inverse_transform_covariance(buf);
        });
    }

    /// @brief Applies inverse covariance transforms in reverse order from cov into out without allocation.
    /// @details Empty pipelines delegate to FittedIdentityTarget.
    /// @param[in] cov Read-only flattened m x m transformed covariance buffer.
    /// @param[out] out Preallocated mutable output buffer of matching length.
    /// @return Empty expected on success, or GprError on invalid input, length mismatch, or null step.
    /// @retval GprErrorCode::EmptyInput When cov is empty.
    /// @retval GprErrorCode::LengthMismatch When out.size() != cov.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of cov is NaN or infinite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<void, GprError> FittedTargetPipeline::inverse_transform_covariance_into(
        std::span<const double> cov, std::span<double> out
    ) const
    {
        if (flat_steps_.empty())
        {
            return FittedIdentityTarget{}.inverse_transform_covariance_into(cov, out);
        }
        return reverse_target_into(
            flat_steps_, cov, out,
            [](const FittedTargetTransform& step, std::span<const double> src, std::span<double> dst)
        { return step.inverse_transform_covariance_into(src, dst); },
            [](const FittedTargetTransform& step, std::span<double> buf)
        { return step.inverse_transform_covariance(buf); }
        );
    }

    /// @brief Returns a polymorphic deep copy of this fitted target pipeline.
    /// @details Allocates a new FittedTargetPipeline copy.
    /// @return Unique pointer owning a deep copy of *this.
    std::unique_ptr<FittedTargetTransform> FittedTargetPipeline::clone() const
    {
        return std::make_unique<FittedTargetPipeline>(*this);
    }

    // --- TargetPipeline ---

    /// @brief Constructs a target pipeline from an ordered list of unfitted steps.
    /// @details Takes ownership of steps in forward application order.
    /// @param[in] steps Ordered vector of unfitted target transform steps.
    TargetPipeline::TargetPipeline(std::vector<std::unique_ptr<UnfittedTargetTransform>> steps) noexcept :
        steps_(std::move(steps))
    {
    }

    /// @brief Deep-copies the unfitted target pipeline.
    /// @details Clones each non-null step preserving forward order.
    /// @param[in] other Source TargetPipeline to copy.
    TargetPipeline::TargetPipeline(const TargetPipeline& other) : steps_(clone_ptr_vec(other.steps_)) {}

    /// @brief Deep-copy assigns the unfitted target pipeline.
    /// @details Replaces owned steps with deep clones of other's steps when &other != this.
    /// @param[in] other Source TargetPipeline to copy from.
    /// @return Reference to *this.
    TargetPipeline& TargetPipeline::operator=(const TargetPipeline& other)
    {
        if (this != &other)
        {
            steps_ = clone_ptr_vec(other.steps_);
        }
        return *this;
    }

    /// @brief Destroys nested unfitted target pipeline steps iteratively without call-stack recursion.
    TargetPipeline::~TargetPipeline()
    {
        PipelineAccess::destroy_steps<TargetPipeline>(steps_);
    }

    /// @brief Appends a step by cloning it onto the end of this target pipeline.
    /// @details Lvalue overload cloning step onto steps_.
    /// @param[in] step Unfitted target transform to clone and append.
    /// @return Reference to *this.
    TargetPipeline& TargetPipeline::then(const UnfittedTargetTransform& step) &
    {
        steps_.push_back(step.clone());
        return *this;
    }

    /// @brief Appends a step by cloning it onto the end of this rvalue target pipeline.
    /// @details Rvalue overload cloning step onto steps_ and forwarding *this.
    /// @param[in] step Unfitted target transform to clone and append.
    /// @return Rvalue reference to *this.
    TargetPipeline&& TargetPipeline::then(const UnfittedTargetTransform& step) &&
    {
        steps_.push_back(step.clone());
        return std::move(*this);
    }

    /// @brief Appends an owned step onto the end of this target pipeline.
    /// @details Moves non-null step into steps_ and ignores null pointers.
    /// @param[in] step Owned unfitted target transform to append.
    /// @return Reference to *this.
    TargetPipeline& TargetPipeline::then(std::unique_ptr<UnfittedTargetTransform> step) &
    {
        if (step != nullptr)
        {
            steps_.push_back(std::move(step));
        }
        return *this;
    }

    /// @brief Appends an owned step onto the end of this rvalue target pipeline.
    /// @details Moves non-null step into steps_ and forwards *this.
    /// @param[in] step Owned unfitted target transform to append.
    /// @return Rvalue reference to *this.
    TargetPipeline&& TargetPipeline::then(std::unique_ptr<UnfittedTargetTransform> step) &&
    {
        if (step != nullptr)
        {
            steps_.push_back(std::move(step));
        }
        return std::move(*this);
    }

    /// @brief Fits each step sequentially on the output of the preceding step.
    /// @details Validates training targets upfront, then copies y into a mutable scratch buffer transformed stagewise
    /// after fitting each step.
    /// @param[in] y Read-only span of training target values.
    /// @return FittedTargetPipeline on success, or GprError on invalid targets, null step, or step fitting failure.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of y or an intermediate output is non-finite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<FittedTargetPipeline, GprError> TargetPipeline::fit_typed(std::span<const double> y) const
    {
        if (auto check = detail::validate_target_vector(y); !check)
        {
            return std::unexpected(std::move(check).error());
        }
        if (steps_.empty())
        {
            return FittedTargetPipeline{};
        }
        // Maintain a single mutable working buffer and an explicit frame stack so nested pipelines do not recurse.
        std::vector<double> buf(y.begin(), y.end());

        struct FitFrame
        {
            const std::vector<std::unique_ptr<UnfittedTargetTransform>>* steps;
            std::size_t idx;
            std::vector<std::unique_ptr<FittedTargetTransform>> fitted;
        };

        std::vector<FitFrame> stack;
        stack.push_back(FitFrame{.steps = &steps_, .idx = 0U, .fitted = {}});
        stack.back().fitted.reserve(steps_.size());

        while (true)
        {
            auto& top = stack.back();
            if (top.idx >= top.steps->size())
            {
                auto finished = std::move(top.fitted);
                stack.pop_back();
                if (stack.empty())
                {
                    return FittedTargetPipeline(std::move(finished));
                }
                stack.back().fitted.push_back(std::make_unique<FittedTargetPipeline>(std::move(finished)));
                continue;
            }
            const auto& step = (*top.steps)[top.idx++];
            if (step == nullptr)
            {
                return std::unexpected(GprError::invalid_config("target pipeline contains null step"));
            }
            if (const auto* sub_pipe = dynamic_cast<const TargetPipeline*>(step.get()))
            {
                const auto& sub_steps = PipelineAccess::steps(*sub_pipe);
                stack.push_back(FitFrame{.steps = &sub_steps, .idx = 0U, .fitted = {}});
                stack.back().fitted.reserve(sub_steps.size());
                continue;
            }
            auto fitted_step = step->fit(buf);
            if (!fitted_step)
            {
                return std::unexpected(std::move(fitted_step).error());
            }
            if (auto res = (*fitted_step)->transform(buf); !res)
            {
                return std::unexpected(std::move(res).error());
            }
            top.fitted.push_back(std::move(*fitted_step));
        }
    }

    /// @brief Fits each step sequentially and returns a boxed fitted target pipeline.
    /// @details Delegates to fit_typed and boxes the typed result.
    /// @param[in] y Read-only span of training target values.
    /// @return Boxed FittedTargetPipeline on success, or GprError on invalid targets, null step, or step fitting
    /// failure.
    /// @retval GprErrorCode::EmptyInput When y is empty.
    /// @retval GprErrorCode::NonFiniteInput When any element of y or an intermediate output is non-finite.
    /// @retval GprErrorCode::InvalidConfig When any pipeline step pointer is null.
    std::expected<std::unique_ptr<FittedTargetTransform>, GprError> TargetPipeline::fit(std::span<const double> y) const
    {
        auto typed = fit_typed(y);
        if (!typed)
        {
            return std::unexpected(std::move(typed).error());
        }
        return std::make_unique<FittedTargetPipeline>(std::move(*typed));
    }

    /// @brief Returns a polymorphic deep copy of this unfitted target pipeline.
    /// @details Allocates a new TargetPipeline copy.
    /// @return Unique pointer owning a deep copy of *this.
    std::unique_ptr<UnfittedTargetTransform> TargetPipeline::clone() const
    {
        return std::make_unique<TargetPipeline>(*this);
    }

}  // namespace cxxgpr::transform
