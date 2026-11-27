// SPDX-License-Identifier: Apache-2.0
/// @file config_json.cpp
/// @brief JSON serialization and deserialization for Exact GPR config.json checkpoints.
/// @details Implements encoding and decoding for all built-in and custom kernel leaves, composite sum/product
/// trees, open parameter intervals, likelihood, jitter policies, and unfitted/fitted input and target
/// transforms while enforcing MAX_JSON_NESTING_DEPTH and schema invariants.

#include "config_json.hpp"

#include "cxxgpr/core/interval.hpp"
#include "cxxgpr/kernel/leaves.hpp"
#include "cxxgpr/transform/columnwise.hpp"
#include "cxxgpr/transform/pipeline.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/noinline.hpp"
#include "json_depth.hpp"
#include "persist/save_id.hpp"

namespace cxxgpr
{

    /// @brief Internal friend helper exposing KernelSpec AST nodes for JSON serialization.
    /// @details Allows config_json.cpp to traverse SumNode, ProductNode, and KernelLeaf variants without
    /// leaking internal AST definitions into public headers.
    struct KernelSpecPersistAccess
    {
        /// @brief Returns the root NodeVariant stored inside spec.
        /// @param[in] spec Kernel specification whose variant node is inspected.
        /// @return Constant reference to spec.node_.
        [[nodiscard]] static const KernelSpec::NodeVariant& node(const KernelSpec& spec) noexcept { return spec.node_; }

        /// @brief Extracts child pointers if v holds a SumNode, or {nullptr, nullptr} otherwise.
        /// @param[in] v KernelSpec variant node to inspect.
        /// @return Pair of left and right child pointers for SumNode, or two null pointers.
        [[nodiscard]] static std::pair<const KernelSpec*, const KernelSpec*> as_sum(
            const KernelSpec::NodeVariant& v
        ) noexcept
        {
            if (const auto* s = std::get_if<KernelSpec::SumNode>(&v))
            {
                return {s->lhs.get(), s->rhs.get()};
            }
            return {nullptr, nullptr};
        }

        /// @brief Extracts child pointers if v holds a ProductNode, or {nullptr, nullptr} otherwise.
        /// @param[in] v KernelSpec variant node to inspect.
        /// @return Pair of left and right child pointers for ProductNode, or two null pointers.
        [[nodiscard]] static std::pair<const KernelSpec*, const KernelSpec*> as_product(
            const KernelSpec::NodeVariant& v
        ) noexcept
        {
            if (const auto* p = std::get_if<KernelSpec::ProductNode>(&v))
            {
                return {p->lhs.get(), p->rhs.get()};
            }
            return {nullptr, nullptr};
        }
    };

}  // namespace cxxgpr

namespace cxxgpr::persist::detail
{
    /// @brief Reports whether text is valid UTF-8, that is, whether it can be written as a JSON string.
    /// @param[in] text Bytes to check.
    /// @return True when the JSON serializer accepts text as a string value.
    bool is_valid_utf8(std::string_view text)
    {
        // Ask the serializer itself, so that the answer is exactly "dump() will not throw on this string".
        try
        {
            (void)nlohmann::json(std::string(text)).dump();
            return true;
        }
        catch (const nlohmann::json::exception&)
        {
            return false;
        }
    }

    namespace
    {

        /// @brief Maximum recursive AST depth allowed when encoding or decoding kernel and transform trees.
        constexpr std::size_t k_max_nesting_depth = 128U;

        /// @brief Constructs a PersistFailed(Config) error with reason.
        /// @param[in] reason Diagnostic message describing the configuration error.
        /// @return GprError with PersistErrorKind::Config.
        [[nodiscard]] GprError config_err(std::string reason)
        {
            return GprError::persist_failed(PersistErrorKind::Config, std::move(reason));
        }

        /// @brief Error of one recursive encode or decode pass, recorded by the node that fails first.
        /// @details The recursive functions return nothing and stop as soon as the slot holds an error, so that no
        /// nesting level carries a std::expected (whose GprError alone spans a few hundred bytes) on the stack.
        using ErrorSlot = std::optional<GprError>;

        /// @brief Records PersistFailed(Config) with reason in err, in a stack frame of its own.
        /// @param[out] err Error slot of the running pass.
        /// @param[in] reason Diagnostic message describing the configuration error.
        CXXGPR_NOINLINE void fail_config(ErrorSlot& err, const char* reason)
        {
            err.emplace(config_err(reason));
        }

        /// @brief Runs a non-recursive step in a stack frame of its own and stores its value or its error.
        /// @details The recursive encoders and decoders hand every step that builds a std::expected, a JSON object,
        /// or an error message to this function, so that those temporaries occupy the stack once instead of once
        /// per nesting level.
        /// @tparam Out Destination type assignable from the value of the step.
        /// @tparam Step Callable returning a std::expected whose value is assignable to Out.
        /// @param[out] out Destination of the value of the step on success; left unchanged on failure.
        /// @param[out] err Error slot that receives the error of the step on failure.
        /// @param[in] step Non-recursive step to run.
        template <typename Out, typename Step>
        CXXGPR_NOINLINE void produce_into(Out& out, ErrorSlot& err, const Step& step)
        {
            auto result = step();
            if (!result.has_value())
            {
                err.emplace(std::move(result).error());
                return;
            }
            out = std::move(*result);
        }

        /// @brief Encodes the state text of a custom kernel or transform as the JSON value stored under "state".
        /// @details Text that parses as a JSON value other than a string is stored as that value, and the loader
        /// re-serializes it. Every other text is stored verbatim as a JSON string: text that does not parse, and
        /// text that is itself a JSON string literal, whose quotes would otherwise be lost because the loader hands
        /// any string value back as it is. Text stored as a string must be valid UTF-8, since config.json is.
        /// @param[in] text State text returned by persist_state_json().
        /// @return JSON value to store, or GprError when the text cannot be written into config.json.
        /// @retval GprErrorCode::PersistFailed When a stored JSON value is nested deeper than the loader accepts
        /// (Config) or text that has to be stored as a string is not valid UTF-8 (NotPersistable).
        [[nodiscard]] std::expected<nlohmann::json, GprError> encode_custom_state(const std::string& text)
        {
            // Non-throwing parse: text that is not JSON yields a discarded value. The parser is not recursive.
            auto parsed = nlohmann::json::parse(text, nullptr, false);
            if (!parsed.is_discarded() && !parsed.is_string())
            {
                // Bound the nesting of a value stored as JSON before dump() recurses into it once per level; the
                // loader would reject the deeper document anyway.
                if (auto depth_ok = check_json_max_depth(text, MAX_JSON_NESTING_DEPTH, PersistErrorKind::Config);
                    !depth_ok.has_value())
                {
                    return std::unexpected(std::move(depth_ok).error());
                }
                return parsed;
            }
            if (!is_valid_utf8(text))
            {
                return std::unexpected(
                    GprError::persist_failed(
                        PersistErrorKind::NotPersistable,
                        "custom state is neither JSON nor valid UTF-8 text and cannot be stored in config.json"
                    )
                );
            }
            return nlohmann::json(text);
        }

        /// @brief Checks, without recursion, that the object and array nesting of value does not exceed max_depth.
        /// @details Counts the levels that check_json_max_depth() counts in the serialized text of value, where the
        /// root container is level 1, so that a document is refused before dump() recurses into it once per level.
        /// @param[in] value JSON document to inspect.
        /// @param[in] max_depth Maximum permissible nesting depth of objects and arrays.
        /// @return Empty expected when the nesting stays within max_depth, or PersistFailed(Config) otherwise.
        /// @retval GprErrorCode::PersistFailed When objects and arrays are nested deeper than max_depth.
        [[nodiscard]] std::expected<void, GprError> check_value_max_depth(
            const nlohmann::json& value, std::size_t max_depth
        )
        {
            // Depth-first walk over an explicit stack of (node, level) pairs.
            std::vector<std::pair<const nlohmann::json*, std::size_t>> pending{
                {&value, 1U}
            };
            while (!pending.empty())
            {
                const auto [node, level] = pending.back();
                pending.pop_back();
                if (!node->is_structured())
                {
                    continue;
                }
                if (level > max_depth)
                {
                    return std::unexpected(config_err("JSON nesting depth exceeds limit"));
                }
                for (const auto& child : *node)
                {
                    pending.emplace_back(&child, level + 1U);
                }
            }
            return {};
        }

        /// @brief Reads an optional top-level string field of config.json.
        /// @param[in] root Parsed config.json root object.
        /// @param[in] key Field name.
        /// @param[in] fallback Value used when the field is absent.
        /// @return The field's string, fallback when absent, or PersistFailed(Config) when it is not a string.
        [[nodiscard]] std::expected<std::string, GprError> optional_string(
            const nlohmann::json& root, std::string_view key, std::string_view fallback
        )
        {
            const auto it = root.find(key);
            if (it == root.end())
            {
                return std::string(fallback);
            }
            if (!it->is_string())
            {
                return std::unexpected(config_err("config.json field " + std::string(key) + " must be a string"));
            }
            return it->get<std::string>();
        }

        /// @brief Serializes a bounded parameter value and its open interval (lo, hi) as JSON.
        /// @param[in] val Parameter value in constrained user units.
        /// @param[in] iv Validated open interval bounds (lo, hi).
        /// @return JSON object containing "value", "lo", and "hi".
        [[nodiscard]] nlohmann::json encode_bounded(double val, const Interval& iv)
        {
            return nlohmann::json{
                {"value",     val},
                {   "lo", iv.lo()},
                {   "hi", iv.hi()}
            };
        }

        /// @brief Parses and validates a bounded parameter JSON object against Interval::positive and BoundedParam.
        /// @param[in] j JSON object containing numeric "value", "lo", and "hi" fields.
        /// @return Validated (value, Interval) pair or GprError on schema or range error.
        /// @retval GprErrorCode::PersistFailed When required numeric fields are missing or malformed.
        /// @retval GprErrorCode::InvalidInterval When bounds or value fail open-interval validation.
        [[nodiscard]] std::expected<std::pair<double, Interval>, GprError> decode_bounded(const nlohmann::json& j)
        {
            if (!j.is_object() || !j.contains("value") || !j.contains("lo") || !j.contains("hi"))
            {
                return std::unexpected(config_err("bounded parameter must have value, lo, hi"));
            }
            if (!j["value"].is_number() || !j["lo"].is_number() || !j["hi"].is_number())
            {
                return std::unexpected(config_err("bounded parameter fields must be numbers"));
            }
            const double val = j["value"].get<double>();
            const double lo = j["lo"].get<double>();
            const double hi = j["hi"].get<double>();
            // Re-validate open positivity bounds and membership of val inside (lo, hi).
            auto iv = Interval::positive(lo, hi);
            if (!iv.has_value())
            {
                return std::unexpected(std::move(iv).error());
            }
            auto bp = BoundedParam::create(val, *iv);
            if (!bp.has_value())
            {
                return std::unexpected(std::move(bp).error());
            }
            return std::pair<double, Interval>{bp->value(), *iv};
        }

        /// @brief Maps MaternNu enum value to its canonical config.json string token.
        /// @param[in] nu Matern smoothness parameter enum.
        /// @return Canonical fraction string ("1/2", "3/2", or "5/2").
        [[nodiscard]] std::string_view nu_to_str(MaternNu nu) noexcept
        {
            switch (nu)
            {
                case MaternNu::OneHalf:
                    return "1/2";
                case MaternNu::ThreeHalves:
                    return "3/2";
                case MaternNu::FiveHalves:
                    return "5/2";
            }
            return "5/2";
        }

        /// @brief Parses canonical Matern smoothness string into MaternNu.
        /// @param[in] s Serialized smoothness string ("1/2", "3/2", or "5/2").
        /// @return Matching MaternNu enum value, or PersistFailed(Config) if unrecognized.
        /// @retval GprErrorCode::PersistFailed When s is not a supported Matern nu token.
        [[nodiscard]] std::expected<MaternNu, GprError> str_to_nu(std::string_view s)
        {
            if (s == "1/2")
            {
                return MaternNu::OneHalf;
            }
            if (s == "3/2")
            {
                return MaternNu::ThreeHalves;
            }
            if (s == "5/2")
            {
                return MaternNu::FiveHalves;
            }
            return std::unexpected(config_err("invalid matern nu string"));
        }

        /// @brief Serializes a single KernelLeaf variant into its tagged config.json representation.
        /// @details Encodes hyperparameters alongside open bounds for all 10 built-in leaves and validates
        /// persist_id and JSON state for CustomKernel leaves.
        /// @param[in] leaf Kernel leaf variant to serialize.
        /// @return Single-key JSON object representing the leaf, or GprError if custom leaf is not persistable.
        /// @retval GprErrorCode::PersistFailed When a custom kernel has an invalid/empty persist_id or state fails.
        [[nodiscard]] std::expected<nlohmann::json, GprError> encode_leaf(const KernelLeaf& leaf)
        {
            return std::visit(
                [](const auto& k) -> std::expected<nlohmann::json, GprError>
            {
                using K = std::decay_t<decltype(k)>;
                if constexpr (std::same_as<K, RbfKernel>)
                {
                    std::array<Interval, 1> ivs{Interval::default_positive()};
                    (void)k.bounds_into(ivs);
                    return nlohmann::json{
                        {"rbf", {{"length_scale", encode_bounded(k.lengthscale(), ivs[0])}}}
                    };
                }
                else if constexpr (std::same_as<K, RbfArdKernel>)
                {
                    const std::size_t np = k.num_params();
                    std::vector<Interval> ivs(np, Interval::default_positive());
                    (void)k.bounds_into(ivs);
                    const auto ls = k.lengthscales();
                    nlohmann::json arr = nlohmann::json::array();
                    for (std::size_t i = 0; i < np; ++i)
                    {
                        arr.push_back(encode_bounded(ls[i], ivs[i]));
                    }
                    return nlohmann::json{
                        {"rbf_ard", {{"length_scales", std::move(arr)}}}
                    };
                }
                else if constexpr (std::same_as<K, MaternKernel>)
                {
                    std::array<Interval, 1> ivs{Interval::default_positive()};
                    (void)k.bounds_into(ivs);
                    return nlohmann::json{
                        {"matern",
                         {{"nu", nu_to_str(k.nu())}, {"length_scale", encode_bounded(k.lengthscale(), ivs[0])}}}
                    };
                }
                else if constexpr (std::same_as<K, MaternArdKernel>)
                {
                    const std::size_t np = k.num_params();
                    std::vector<Interval> ivs(np, Interval::default_positive());
                    (void)k.bounds_into(ivs);
                    const auto ls = k.lengthscales();
                    nlohmann::json arr = nlohmann::json::array();
                    for (std::size_t i = 0; i < np; ++i)
                    {
                        arr.push_back(encode_bounded(ls[i], ivs[i]));
                    }
                    return nlohmann::json{
                        {"matern_ard", {{"nu", nu_to_str(k.nu())}, {"length_scales", std::move(arr)}}}
                    };
                }
                else if constexpr (std::same_as<K, PeriodicKernel>)
                {
                    std::array<Interval, 2> ivs{Interval::default_positive(), Interval::default_positive()};
                    (void)k.bounds_into(ivs);
                    return nlohmann::json{
                        {"periodic",
                         {{"length_scale", encode_bounded(k.lengthscale(), ivs[0])},
                                {"period", encode_bounded(k.period(), ivs[1])}}}
                    };
                }
                else if constexpr (std::same_as<K, RationalQuadraticKernel>)
                {
                    std::array<Interval, 2> ivs{Interval::default_positive(), Interval::default_positive()};
                    (void)k.bounds_into(ivs);
                    return nlohmann::json{
                        {"rq",
                         {{"length_scale", encode_bounded(k.lengthscale(), ivs[0])},
                                {"alpha", encode_bounded(k.alpha(), ivs[1])}}}
                    };
                }
                else if constexpr (std::same_as<K, RationalQuadraticArdKernel>)
                {
                    const std::size_t np = k.num_params();
                    std::vector<Interval> ivs(np, Interval::default_positive());
                    (void)k.bounds_into(ivs);
                    const auto ls = k.lengthscales();
                    nlohmann::json arr = nlohmann::json::array();
                    for (std::size_t i = 0; i + 1U < np; ++i)
                    {
                        arr.push_back(encode_bounded(ls[i], ivs[i]));
                    }
                    return nlohmann::json{
                        {"rq_ard",
                         {{"length_scales", std::move(arr)}, {"alpha", encode_bounded(k.alpha(), ivs[np - 1U])}}}
                    };
                }
                else if constexpr (std::same_as<K, ConstantKernel>)
                {
                    std::array<Interval, 1> ivs{Interval::default_positive()};
                    (void)k.bounds_into(ivs);
                    return nlohmann::json{
                        {"constant", {{"value", encode_bounded(k.constant(), ivs[0])}}}
                    };
                }
                else if constexpr (std::same_as<K, LinearKernel>)
                {
                    std::array<Interval, 1> ivs{Interval::default_positive()};
                    (void)k.bounds_into(ivs);
                    return nlohmann::json{
                        {"linear", {{"variance", encode_bounded(k.variance(), ivs[0])}}}
                    };
                }
                else if constexpr (std::same_as<K, WhiteKernel>)
                {
                    std::array<Interval, 1> ivs{Interval::default_positive()};
                    (void)k.bounds_into(ivs);
                    return nlohmann::json{
                        {"white", {{"variance", encode_bounded(k.variance(), ivs[0])}}}
                    };
                }
                else
                {
                    // Validate custom kernel persistence ID and parse JSON state cleanly if valid JSON.
                    const std::string_view pid = k.persist_id();
                    if (auto v = validate_persist_id(pid); !v.has_value())
                    {
                        if (pid.empty())
                        {
                            return std::unexpected(
                                GprError::persist_failed(
                                    PersistErrorKind::NotPersistable, "custom kernel has empty persist_id"
                                )
                            );
                        }
                        return std::unexpected(std::move(v).error());
                    }
                    auto st = k.persist_state_json();
                    if (!st.has_value())
                    {
                        return std::unexpected(std::move(st).error());
                    }
                    auto state = encode_custom_state(*st);
                    if (!state.has_value())
                    {
                        return std::unexpected(std::move(state).error());
                    }
                    return nlohmann::json{
                        {"custom", {{"persist_id", std::string(pid)}, {"state", std::move(*state)}}}
                    };
                }
            }, leaf
            );
        }

        /// @brief Builds the JSON object of a sum or product kernel node from its encoded children.
        /// @param[in] tag Node tag, "sum" or "product".
        /// @param[in] lhs Encoded left child, moved from.
        /// @param[in] rhs Encoded right child, moved from.
        /// @param[out] out Destination of the JSON object of the node.
        CXXGPR_NOINLINE void make_composite_kernel_json(
            const char* tag, nlohmann::json&& lhs, nlohmann::json&& rhs, nlohmann::json& out
        )
        {
            out = nlohmann::json{
                {tag, {{"left", std::move(lhs)}, {"right", std::move(rhs)}}}
            };
        }

        /// @brief Recursively encodes a KernelSpec tree into JSON while guarding against excessive depth.
        /// @details A nesting level keeps only its two encoded children on the stack: leaf encoding, object
        /// construction, and error construction run in frames of their own.
        /// @param[in] spec Kernel specification tree node to encode.
        /// @param[in] depth Current AST recursion depth bounded by k_max_nesting_depth.
        /// @param[out] out Destination of the JSON representation of spec.
        /// @param[out] err Error slot that receives the first failure: PersistFailed(Config) when nesting depth
        /// exceeds 128, or the error of a leaf that fails to serialize.
        void encode_kernel_node(const KernelSpec& spec, std::size_t depth, nlohmann::json& out, ErrorSlot& err)
        {
            if (depth > k_max_nesting_depth)
            {
                fail_config(err, "kernel nesting depth exceeds 128");
                return;
            }
            const auto& node = KernelSpecPersistAccess::node(spec);
            if (const auto* leaf = std::get_if<KernelLeaf>(&node))
            {
                produce_into(out, err, [leaf] { return encode_leaf(*leaf); });
                return;
            }
            // Encode binary SumNode and ProductNode subtrees recursively with depth + 1.
            auto children = KernelSpecPersistAccess::as_sum(node);
            const bool is_sum = children.first != nullptr;
            if (!is_sum)
            {
                children = KernelSpecPersistAccess::as_product(node);
            }
            nlohmann::json lhs;
            encode_kernel_node(*children.first, depth + 1U, lhs, err);
            if (err.has_value())
            {
                return;
            }
            nlohmann::json rhs;
            encode_kernel_node(*children.second, depth + 1U, rhs, err);
            if (err.has_value())
            {
                return;
            }
            make_composite_kernel_json(is_sum ? "sum" : "product", std::move(lhs), std::move(rhs), out);
        }

        /// @brief Decodes a kernel leaf node, that is, a node whose tag is neither "sum" nor "product".
        /// @param[in] key Tag of the node.
        /// @param[in] body JSON value stored under key.
        /// @param[in] registry Extension registry used to instantiate "custom" leaves.
        /// @return Reconstructed leaf KernelSpec on success, or GprError on schema, interval, or registry failure.
        /// @retval GprErrorCode::PersistFailed When the schema is invalid, the tag is unknown, or a custom ID is
        /// missing.
        [[nodiscard]] std::expected<KernelSpec, GprError> decode_kernel_leaf(
            const std::string& key, const nlohmann::json& body, const PersistRegistry& registry
        )
        {
            if (!body.is_object())
            {
                return std::unexpected(config_err("kernel leaf body must be an object"));
            }

            if (key == "rbf")
            {
                if (!body.contains("length_scale"))
                {
                    return std::unexpected(config_err("rbf missing length_scale"));
                }
                auto p = decode_bounded(body["length_scale"]);
                if (!p.has_value())
                {
                    return std::unexpected(std::move(p).error());
                }
                auto k = RbfKernel::create(p->first, p->second);
                if (!k.has_value())
                {
                    return std::unexpected(std::move(k).error());
                }
                return KernelSpec(*k);
            }
            if (key == "rbf_ard" || key == "matern_ard")
            {
                if (!body.contains("length_scales") || !body["length_scales"].is_array()
                    || body["length_scales"].empty())
                {
                    return std::unexpected(config_err("ARD kernel missing length_scales array"));
                }
                std::vector<double> vals;
                std::vector<Interval> ivs;
                for (const auto& elem : body["length_scales"])
                {
                    auto p = decode_bounded(elem);
                    if (!p.has_value())
                    {
                        return std::unexpected(std::move(p).error());
                    }
                    vals.push_back(p->first);
                    ivs.push_back(p->second);
                }
                if (key == "rbf_ard")
                {
                    auto k = RbfArdKernel::create(vals, ivs);
                    if (!k.has_value())
                    {
                        return std::unexpected(std::move(k).error());
                    }
                    return KernelSpec(*k);
                }
                if (!body.contains("nu") || !body["nu"].is_string())
                {
                    return std::unexpected(config_err("matern_ard missing nu"));
                }
                auto nu = str_to_nu(body["nu"].get<std::string>());
                if (!nu.has_value())
                {
                    return std::unexpected(std::move(nu).error());
                }
                auto k = MaternArdKernel::create(vals, *nu, ivs);
                if (!k.has_value())
                {
                    return std::unexpected(std::move(k).error());
                }
                return KernelSpec(*k);
            }
            if (key == "matern")
            {
                if (!body.contains("nu") || !body["nu"].is_string() || !body.contains("length_scale"))
                {
                    return std::unexpected(config_err("matern missing nu or length_scale"));
                }
                auto nu = str_to_nu(body["nu"].get<std::string>());
                if (!nu.has_value())
                {
                    return std::unexpected(std::move(nu).error());
                }
                auto p = decode_bounded(body["length_scale"]);
                if (!p.has_value())
                {
                    return std::unexpected(std::move(p).error());
                }
                auto k = MaternKernel::create(p->first, *nu, p->second);
                if (!k.has_value())
                {
                    return std::unexpected(std::move(k).error());
                }
                return KernelSpec(*k);
            }
            if (key == "periodic")
            {
                if (!body.contains("length_scale") || !body.contains("period"))
                {
                    return std::unexpected(config_err("periodic missing length_scale or period"));
                }
                auto ls = decode_bounded(body["length_scale"]);
                if (!ls.has_value())
                {
                    return std::unexpected(std::move(ls).error());
                }
                auto per = decode_bounded(body["period"]);
                if (!per.has_value())
                {
                    return std::unexpected(std::move(per).error());
                }
                auto k = PeriodicKernel::create(ls->first, per->first, ls->second, per->second);
                if (!k.has_value())
                {
                    return std::unexpected(std::move(k).error());
                }
                return KernelSpec(*k);
            }
            if (key == "rq")
            {
                if (!body.contains("length_scale") || !body.contains("alpha"))
                {
                    return std::unexpected(config_err("rq missing length_scale or alpha"));
                }
                auto ls = decode_bounded(body["length_scale"]);
                if (!ls.has_value())
                {
                    return std::unexpected(std::move(ls).error());
                }
                auto al = decode_bounded(body["alpha"]);
                if (!al.has_value())
                {
                    return std::unexpected(std::move(al).error());
                }
                auto k = RationalQuadraticKernel::create(ls->first, al->first, ls->second, al->second);
                if (!k.has_value())
                {
                    return std::unexpected(std::move(k).error());
                }
                return KernelSpec(*k);
            }
            if (key == "rq_ard")
            {
                if (!body.contains("length_scales") || !body["length_scales"].is_array()
                    || body["length_scales"].empty() || !body.contains("alpha"))
                {
                    return std::unexpected(config_err("rq_ard missing length_scales or alpha"));
                }
                std::vector<double> vals;
                std::vector<Interval> ivs;
                for (const auto& elem : body["length_scales"])
                {
                    auto p = decode_bounded(elem);
                    if (!p.has_value())
                    {
                        return std::unexpected(std::move(p).error());
                    }
                    vals.push_back(p->first);
                    ivs.push_back(p->second);
                }
                auto al = decode_bounded(body["alpha"]);
                if (!al.has_value())
                {
                    return std::unexpected(std::move(al).error());
                }
                auto k = RationalQuadraticArdKernel::create(vals, al->first, ivs, al->second);
                if (!k.has_value())
                {
                    return std::unexpected(std::move(k).error());
                }
                return KernelSpec(*k);
            }
            if (key == "constant")
            {
                if (!body.contains("value"))
                {
                    return std::unexpected(config_err("constant missing value"));
                }
                auto p = decode_bounded(body["value"]);
                if (!p.has_value())
                {
                    return std::unexpected(std::move(p).error());
                }
                auto k = ConstantKernel::create(p->first, p->second);
                if (!k.has_value())
                {
                    return std::unexpected(std::move(k).error());
                }
                return KernelSpec(*k);
            }
            if (key == "linear" || key == "white")
            {
                if (!body.contains("variance"))
                {
                    return std::unexpected(config_err("linear/white missing variance"));
                }
                auto p = decode_bounded(body["variance"]);
                if (!p.has_value())
                {
                    return std::unexpected(std::move(p).error());
                }
                if (key == "linear")
                {
                    auto k = LinearKernel::create(p->first, p->second);
                    if (!k.has_value())
                    {
                        return std::unexpected(std::move(k).error());
                    }
                    return KernelSpec(*k);
                }
                auto k = WhiteKernel::create(p->first, p->second);
                if (!k.has_value())
                {
                    return std::unexpected(std::move(k).error());
                }
                return KernelSpec(*k);
            }
            if (key == "custom")
            {
                if (!body.contains("persist_id") || !body["persist_id"].is_string() || !body.contains("state"))
                {
                    return std::unexpected(config_err("custom kernel missing persist_id or state"));
                }
                const std::string pid = body["persist_id"].get<std::string>();
                const std::string state_str =
                    body["state"].is_string() ? body["state"].get<std::string>() : body["state"].dump();
                auto ck = registry.create_kernel(pid, state_str);
                if (!ck.has_value())
                {
                    return std::unexpected(std::move(ck).error());
                }
                return KernelSpec(*ck);
            }

            return std::unexpected(config_err("unknown kernel node kind"));
        }

        /// @brief Checks the body of a sum or product node and returns its left and right child nodes.
        /// @param[in] body JSON value stored under the "sum" or "product" tag.
        /// @param[out] err Error slot that receives PersistFailed(Config) when a child is missing.
        /// @return Pointers to the left and right child nodes, or two null pointers when err was filled.
        [[nodiscard]] CXXGPR_NOINLINE std::pair<const nlohmann::json*, const nlohmann::json*> composite_kernel_children(
            const nlohmann::json& body, ErrorSlot& err
        )
        {
            if (!body.is_object() || !body.contains("left") || !body.contains("right"))
            {
                fail_config(err, "composite kernel node missing left/right");
                return {nullptr, nullptr};
            }
            return {&body["left"], &body["right"]};
        }

        /// @brief Replaces the decoded left child in lhs with its sum or product with the decoded right child.
        /// @param[in] sum True for a sum node, false for a product node.
        /// @param[in,out] lhs Decoded left child on entry, the composite kernel on return.
        /// @param[in] rhs Decoded right child, moved from.
        CXXGPR_NOINLINE void combine_kernel_children(
            bool sum, std::optional<KernelSpec>& lhs, std::optional<KernelSpec>&& rhs
        )
        {
            // The caller combines only after both children decoded, so both hold a value here.
            if (lhs.has_value() && rhs.has_value())
            {
                lhs = sum ? (std::move(*lhs) + std::move(*rhs)) : (std::move(*lhs) * std::move(*rhs));
            }
        }

        /// @brief Recursively decodes a kernel JSON node into a KernelSpec tree.
        /// @details A nesting level keeps only the decoded right child on the stack, the left child being decoded
        /// straight into out: leaf decoding, child lookup, composition, and error construction run in frames of
        /// their own.
        /// @param[in] j Single-key JSON object representing a leaf, sum, or product kernel node.
        /// @param[in] registry Extension registry used to instantiate "custom" leaves.
        /// @param[in] depth Current AST recursion depth bounded by k_max_nesting_depth.
        /// @param[out] out Destination of the reconstructed KernelSpec.
        /// @param[out] err Error slot that receives the first failure: PersistFailed when nesting exceeds 128, the
        /// schema is invalid, or a custom ID is missing, or the interval or registry error of a leaf.
        void decode_kernel_node(
            const nlohmann::json& j,
            const PersistRegistry& registry,
            std::size_t depth,
            std::optional<KernelSpec>& out,
            ErrorSlot& err
        )
        {
            if (depth > k_max_nesting_depth)
            {
                fail_config(err, "kernel nesting depth exceeds 128");
                return;
            }
            if (!j.is_object() || j.size() != 1U)
            {
                fail_config(err, "kernel node must be a single-key JSON object");
                return;
            }
            const auto it = j.begin();
            const std::string& key = it.key();
            const nlohmann::json& body = it.value();

            // Reconstruct binary sum and product compositions before leaf variants.
            if (key != "sum" && key != "product")
            {
                produce_into(out, err, [&] { return decode_kernel_leaf(key, body, registry); });
                return;
            }
            const auto [left, right] = composite_kernel_children(body, err);
            if (err.has_value())
            {
                return;
            }
            decode_kernel_node(*left, registry, depth + 1U, out, err);
            if (err.has_value())
            {
                return;
            }
            std::optional<KernelSpec> rhs;
            decode_kernel_node(*right, registry, depth + 1U, rhs, err);
            if (err.has_value())
            {
                return;
            }
            combine_kernel_children(key == "sum", out, std::move(rhs));
        }

        /// @brief Serializes a user-defined transform implementing persist_id() and persist_state_json().
        /// @tparam CustomTransformT Input or target transform interface type.
        /// @param[in] t Custom transform instance to serialize.
        /// @return JSON object {"custom": {"persist_id": ..., "state": ...}}, or GprError if non-persistable.
        /// @retval GprErrorCode::PersistFailed When persist_id() is empty/reserved or state export fails.
        template <typename CustomTransformT>
        [[nodiscard]] std::expected<nlohmann::json, GprError> encode_custom_transform(const CustomTransformT& t)
        {
            const std::string_view pid = t.persist_id();
            if (pid.empty())
            {
                return std::unexpected(
                    GprError::persist_failed(
                        PersistErrorKind::NotPersistable, "custom transform does not support persistence"
                    )
                );
            }
            if (auto v = validate_persist_id(pid); !v.has_value())
            {
                return std::unexpected(std::move(v).error());
            }
            auto st = t.persist_state_json();
            if (!st.has_value())
            {
                return std::unexpected(std::move(st).error());
            }
            auto state = encode_custom_state(*st);
            if (!state.has_value())
            {
                return std::unexpected(std::move(state).error());
            }
            return nlohmann::json{
                {"custom", {{"persist_id", std::string(pid)}, {"state", std::move(*state)}}}
            };
        }

        /// @brief Wraps the encoded children of a composite transform as {tag: {field: items}}.
        /// @param[in] tag Node tag, "pipeline" or "columnwise".
        /// @param[in] field Name of the child array, "steps" or "maps".
        /// @param[in] items Encoded children in order, moved from.
        /// @param[out] out Destination of the JSON object of the node.
        CXXGPR_NOINLINE void make_composite_transform_json(
            const char* tag, const char* field, nlohmann::json&& items, nlohmann::json& out
        )
        {
            out = nlohmann::json{
                {tag, {{field, std::move(items)}}}
            };
        }

        /// @brief Encodes a built-in leaf of the UnfittedInputTransform hierarchy.
        /// @param[in] t Unfitted input transform node to serialize.
        /// @param[out] out Destination of the JSON representation of t when it is a built-in leaf.
        /// @return True when t is an identity, standardize, or min-max transform and out holds its JSON.
        [[nodiscard]] CXXGPR_NOINLINE bool encode_unfitted_input_leaf(
            const transform::UnfittedInputTransform& t, nlohmann::json& out
        )
        {
            if (dynamic_cast<const transform::IdentityInput*>(&t) != nullptr)
            {
                out = nlohmann::json("identity");
                return true;
            }
            if (dynamic_cast<const transform::StandardizeInput*>(&t) != nullptr)
            {
                out = nlohmann::json{
                    {"standardize", nlohmann::json::object()}
                };
                return true;
            }
            if (const auto* mm = dynamic_cast<const transform::MinMaxInput*>(&t))
            {
                const auto [lo, hi] = mm->feature_range();
                out = nlohmann::json{
                    {"min_max", {{"range_lo", lo}, {"range_hi", hi}}}
                };
                return true;
            }
            return false;
        }

        /// @brief Recursively encodes an UnfittedInputTransform hierarchy into config.json format.
        /// @details A nesting level keeps only the child array and the current child on the stack: leaf and custom
        /// encoding, object construction, and error construction run in frames of their own.
        /// @param[in] t Unfitted input transform node to serialize.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @param[out] out Destination of the JSON representation of t.
        /// @param[out] err Error slot that receives the first failure: excessive depth or a non-persistable custom
        /// step.
        void encode_unfitted_input_node(
            const transform::UnfittedInputTransform& t, std::size_t depth, nlohmann::json& out, ErrorSlot& err
        )
        {
            if (depth > k_max_nesting_depth)
            {
                fail_config(err, "transform nesting depth exceeds 128");
                return;
            }
            if (encode_unfitted_input_leaf(t, out))
            {
                return;
            }
            const auto* pipe = dynamic_cast<const transform::Pipeline*>(&t);
            const auto* col = (pipe == nullptr) ? dynamic_cast<const transform::ColumnwiseInput*>(&t) : nullptr;
            if (pipe == nullptr && col == nullptr)
            {
                produce_into(out, err, [&t] { return encode_custom_transform(t); });
                return;
            }
            // Encode Pipeline steps and ColumnwiseInput maps in order with depth + 1.
            nlohmann::json items(nlohmann::json::value_t::array);
            for (const auto& child : (pipe != nullptr) ? pipe->steps() : col->maps())
            {
                nlohmann::json item;
                encode_unfitted_input_node(*child, depth + 1U, item, err);
                if (err.has_value())
                {
                    return;
                }
                items.push_back(std::move(item));
            }
            make_composite_transform_json(
                (pipe != nullptr) ? "pipeline" : "columnwise", (pipe != nullptr) ? "steps" : "maps", std::move(items),
                out
            );
        }

        /// @brief Encodes a built-in leaf of the FittedInputTransform hierarchy with its fitted statistics.
        /// @param[in] t Fitted input transform node to serialize.
        /// @param[out] out Destination of the JSON representation of t when it is a built-in leaf.
        /// @return True when t is an identity, standardize, or min-max transform and out holds its JSON.
        [[nodiscard]] CXXGPR_NOINLINE bool encode_fitted_input_leaf(
            const transform::FittedInputTransform& t, nlohmann::json& out
        )
        {
            if (dynamic_cast<const transform::FittedIdentityInput*>(&t) != nullptr)
            {
                out = nlohmann::json("identity");
                return true;
            }
            if (const auto* st = dynamic_cast<const transform::FittedStandardizeInput*>(&t))
            {
                out = nlohmann::json{
                    {"standardize", {{"mean", st->mean()}, {"std", st->std()}}}
                };
                return true;
            }
            if (const auto* mm = dynamic_cast<const transform::FittedMinMaxInput*>(&t))
            {
                const auto [lo, hi] = mm->feature_range();
                out = nlohmann::json{
                    {"min_max", {{"data_min", mm->min()}, {"data_max", mm->max()}, {"range_lo", lo}, {"range_hi", hi}}}
                };
                return true;
            }
            return false;
        }

        /// @brief Recursively encodes a FittedInputTransform hierarchy and its fitted statistics into JSON.
        /// @details A nesting level keeps only the child array and the current child on the stack: leaf and custom
        /// encoding, object construction, and error construction run in frames of their own.
        /// @param[in] t Fitted input transform node to serialize.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @param[out] out Destination of the JSON representation of t.
        /// @param[out] err Error slot that receives the first failure: excessive depth or a non-persistable custom
        /// step.
        void encode_fitted_input_node(
            const transform::FittedInputTransform& t, std::size_t depth, nlohmann::json& out, ErrorSlot& err
        )
        {
            if (depth > k_max_nesting_depth)
            {
                fail_config(err, "transform nesting depth exceeds 128");
                return;
            }
            if (encode_fitted_input_leaf(t, out))
            {
                return;
            }
            const auto* pipe = dynamic_cast<const transform::FittedPipeline*>(&t);
            const auto* col = (pipe == nullptr) ? dynamic_cast<const transform::FittedColumnwiseInput*>(&t) : nullptr;
            if (pipe == nullptr && col == nullptr)
            {
                produce_into(out, err, [&t] { return encode_custom_transform(t); });
                return;
            }
            // Encode FittedPipeline steps and FittedColumnwiseInput maps in order with depth + 1.
            nlohmann::json items(nlohmann::json::value_t::array);
            for (const auto& child : (pipe != nullptr) ? pipe->steps() : col->maps())
            {
                nlohmann::json item;
                encode_fitted_input_node(*child, depth + 1U, item, err);
                if (err.has_value())
                {
                    return;
                }
                items.push_back(std::move(item));
            }
            make_composite_transform_json(
                (pipe != nullptr) ? "pipeline" : "columnwise", (pipe != nullptr) ? "steps" : "maps", std::move(items),
                out
            );
        }

        /// @brief Encodes a built-in leaf of the UnfittedTargetTransform hierarchy.
        /// @param[in] t Unfitted target transform node to serialize.
        /// @param[out] out Destination of the JSON representation of t when it is a built-in leaf.
        /// @return True when t is an identity, standardize, or min-max transform and out holds its JSON.
        [[nodiscard]] CXXGPR_NOINLINE bool encode_unfitted_target_leaf(
            const transform::UnfittedTargetTransform& t, nlohmann::json& out
        )
        {
            if (dynamic_cast<const transform::IdentityTarget*>(&t) != nullptr)
            {
                out = nlohmann::json("identity");
                return true;
            }
            if (dynamic_cast<const transform::StandardizeTarget*>(&t) != nullptr)
            {
                out = nlohmann::json{
                    {"standardize", nlohmann::json::object()}
                };
                return true;
            }
            if (const auto* mm = dynamic_cast<const transform::MinMaxTarget*>(&t))
            {
                const auto [lo, hi] = mm->feature_range();
                out = nlohmann::json{
                    {"min_max", {{"range_lo", lo}, {"range_hi", hi}}}
                };
                return true;
            }
            return false;
        }

        /// @brief Recursively encodes an UnfittedTargetTransform hierarchy into config.json format.
        /// @details A nesting level keeps only the step array and the current step on the stack: leaf and custom
        /// encoding, object construction, and error construction run in frames of their own.
        /// @param[in] t Unfitted target transform node to serialize.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @param[out] out Destination of the JSON representation of t.
        /// @param[out] err Error slot that receives the first failure: excessive depth or a non-persistable custom
        /// step.
        void encode_unfitted_target_node(
            const transform::UnfittedTargetTransform& t, std::size_t depth, nlohmann::json& out, ErrorSlot& err
        )
        {
            if (depth > k_max_nesting_depth)
            {
                fail_config(err, "transform nesting depth exceeds 128");
                return;
            }
            if (encode_unfitted_target_leaf(t, out))
            {
                return;
            }
            const auto* pipe = dynamic_cast<const transform::TargetPipeline*>(&t);
            if (pipe == nullptr)
            {
                produce_into(out, err, [&t] { return encode_custom_transform(t); });
                return;
            }
            // Encode TargetPipeline steps in order with depth + 1.
            nlohmann::json items(nlohmann::json::value_t::array);
            for (const auto& child : pipe->steps())
            {
                nlohmann::json item;
                encode_unfitted_target_node(*child, depth + 1U, item, err);
                if (err.has_value())
                {
                    return;
                }
                items.push_back(std::move(item));
            }
            make_composite_transform_json("pipeline", "steps", std::move(items), out);
        }

        /// @brief Encodes a built-in leaf of the FittedTargetTransform hierarchy with its fitted statistics.
        /// @param[in] t Fitted target transform node to serialize.
        /// @param[out] out Destination of the JSON representation of t when it is a built-in leaf.
        /// @return True when t is an identity, standardize, or min-max transform and out holds its JSON.
        [[nodiscard]] CXXGPR_NOINLINE bool encode_fitted_target_leaf(
            const transform::FittedTargetTransform& t, nlohmann::json& out
        )
        {
            if (dynamic_cast<const transform::FittedIdentityTarget*>(&t) != nullptr)
            {
                out = nlohmann::json("identity");
                return true;
            }
            if (const auto* st = dynamic_cast<const transform::FittedStandardizeTarget*>(&t))
            {
                out = nlohmann::json{
                    {"standardize", {{"mean", st->mean()}, {"std", st->std()}}}
                };
                return true;
            }
            if (const auto* mm = dynamic_cast<const transform::FittedMinMaxTarget*>(&t))
            {
                const auto [lo, hi] = mm->feature_range();
                out = nlohmann::json{
                    {"min_max", {{"data_min", mm->min()}, {"data_max", mm->max()}, {"range_lo", lo}, {"range_hi", hi}}}
                };
                return true;
            }
            return false;
        }

        /// @brief Recursively encodes a FittedTargetTransform hierarchy and its fitted statistics into JSON.
        /// @details A nesting level keeps only the step array and the current step on the stack: leaf and custom
        /// encoding, object construction, and error construction run in frames of their own.
        /// @param[in] t Fitted target transform node to serialize.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @param[out] out Destination of the JSON representation of t.
        /// @param[out] err Error slot that receives the first failure: excessive depth or a non-persistable custom
        /// step.
        void encode_fitted_target_node(
            const transform::FittedTargetTransform& t, std::size_t depth, nlohmann::json& out, ErrorSlot& err
        )
        {
            if (depth > k_max_nesting_depth)
            {
                fail_config(err, "transform nesting depth exceeds 128");
                return;
            }
            if (encode_fitted_target_leaf(t, out))
            {
                return;
            }
            const auto* pipe = dynamic_cast<const transform::FittedTargetPipeline*>(&t);
            if (pipe == nullptr)
            {
                produce_into(out, err, [&t] { return encode_custom_transform(t); });
                return;
            }
            // Encode FittedTargetPipeline steps in order with depth + 1.
            nlohmann::json items(nlohmann::json::value_t::array);
            for (const auto& child : pipe->steps())
            {
                nlohmann::json item;
                encode_fitted_target_node(*child, depth + 1U, item, err);
                if (err.has_value())
                {
                    return;
                }
                items.push_back(std::move(item));
            }
            make_composite_transform_json("pipeline", "steps", std::move(items), out);
        }

        /// @brief Children of a composite transform node found by a non-recursive node decoder.
        struct CompositeItems
        {
            /// @brief JSON array of the child nodes, or null when the node is not a composite.
            const nlohmann::json* items = nullptr;
            /// @brief True for a "pipeline" node, false for a "columnwise" node.
            bool pipeline = false;
        };

        /// @brief Decodes an UnfittedInputTransform node other than a composite, or finds the children of a composite.
        /// @param[in] j JSON string ("identity") or single-key transform descriptor object.
        /// @param[in] registry Registry supplying factories for "custom" transforms.
        /// @param[out] composite Children of j when j is a valid "pipeline" or "columnwise" node; left null otherwise.
        /// @return Owned UnfittedInputTransform instance, a null pointer when j is a composite, or GprError on schema,
        /// range, or registry error.
        [[nodiscard]] std::expected<std::unique_ptr<transform::UnfittedInputTransform>, GprError>
            decode_unfitted_input_leaf(
                const nlohmann::json& j, const PersistRegistry& registry, CompositeItems& composite
            )
        {
            if (j.is_string() && j.get<std::string>() == "identity")
            {
                return std::make_unique<transform::IdentityInput>();
            }
            if (!j.is_object() || j.size() != 1U)
            {
                return std::unexpected(config_err("invalid unfitted input transform JSON"));
            }
            const std::string& key = j.begin().key();
            const nlohmann::json& body = j.begin().value();
            if (key == "standardize")
            {
                return std::make_unique<transform::StandardizeInput>();
            }
            if (key == "min_max")
            {
                if (!body.is_object() || !body.contains("range_lo") || !body.contains("range_hi"))
                {
                    return std::unexpected(config_err("min_max missing range_lo/range_hi"));
                }
                auto mm =
                    transform::MinMaxInput::create(body["range_lo"].get<double>(), body["range_hi"].get<double>());
                if (!mm.has_value())
                {
                    return std::unexpected(std::move(mm).error());
                }
                return std::make_unique<transform::MinMaxInput>(*mm);
            }
            if (key == "pipeline" || key == "columnwise")
            {
                // Check the child array of Pipeline ("steps") or ColumnwiseInput ("maps"); the caller decodes it.
                const std::string_view arr_key = (key == "pipeline") ? "steps" : "maps";
                if (!body.is_object() || !body.contains(arr_key) || !body[arr_key].is_array())
                {
                    return std::unexpected(config_err("composite transform missing array"));
                }
                composite = CompositeItems{.items = &body[arr_key], .pipeline = key == "pipeline"};
                return nullptr;
            }
            if (key == "custom")
            {
                if (!body.is_object() || !body.contains("persist_id") || !body["persist_id"].is_string()
                    || !body.contains("state"))
                {
                    return std::unexpected(config_err("custom transform missing persist_id/state"));
                }
                const std::string pid = body["persist_id"].get<std::string>();
                const std::string st =
                    body["state"].is_string() ? body["state"].get<std::string>() : body["state"].dump();
                return registry.create_unfitted_input(pid, st);
            }
            return std::unexpected(config_err("unknown unfitted input transform"));
        }

        /// @brief Builds a Pipeline or ColumnwiseInput from its decoded children.
        /// @param[in] pipeline True for a Pipeline, false for a ColumnwiseInput.
        /// @param[in] items Decoded children in order, moved from.
        /// @param[out] out Destination of the composite transform.
        CXXGPR_NOINLINE void make_unfitted_input_composite(
            bool pipeline,
            std::vector<std::unique_ptr<transform::UnfittedInputTransform>>&& items,
            std::unique_ptr<transform::UnfittedInputTransform>& out
        )
        {
            if (pipeline)
            {
                out = std::make_unique<transform::Pipeline>(std::move(items));
                return;
            }
            out = std::make_unique<transform::ColumnwiseInput>(std::move(items));
        }

        /// @brief Recursively decodes an UnfittedInputTransform hierarchy from JSON.
        /// @details A nesting level keeps only the decoded children on the stack: node decoding, composite
        /// construction, and error construction run in frames of their own.
        /// @param[in] j JSON string ("identity") or single-key transform descriptor object.
        /// @param[in] registry Registry supplying factories for "custom" transforms.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @param[out] out Destination of the owned UnfittedInputTransform instance.
        /// @param[out] err Error slot that receives the first schema, range, depth, or registry error.
        void decode_unfitted_input_node(
            const nlohmann::json& j,
            const PersistRegistry& registry,
            std::size_t depth,
            std::unique_ptr<transform::UnfittedInputTransform>& out,
            ErrorSlot& err
        )
        {
            if (depth > k_max_nesting_depth)
            {
                fail_config(err, "transform nesting depth exceeds 128");
                return;
            }
            CompositeItems composite;
            produce_into(out, err, [&] { return decode_unfitted_input_leaf(j, registry, composite); });
            if (err.has_value() || composite.items == nullptr)
            {
                return;
            }
            // Decode Pipeline ("steps") and ColumnwiseInput ("maps") children uniformly with depth + 1.
            std::vector<std::unique_ptr<transform::UnfittedInputTransform>> items;
            for (const auto& child : *composite.items)
            {
                std::unique_ptr<transform::UnfittedInputTransform> item;
                decode_unfitted_input_node(child, registry, depth + 1U, item, err);
                if (err.has_value())
                {
                    return;
                }
                items.push_back(std::move(item));
            }
            make_unfitted_input_composite(composite.pipeline, std::move(items), out);
        }

        /// @brief Decodes a FittedInputTransform node other than a composite, or finds the children of a composite.
        /// @param[in] j JSON string ("identity") or single-key fitted transform descriptor object.
        /// @param[in] registry Registry supplying factories for "custom" transforms.
        /// @param[out] composite Children of j when j is a valid "pipeline" or "columnwise" node; left null otherwise.
        /// @return Owned FittedInputTransform instance with validated statistics, a null pointer when j is a
        /// composite, or GprError on schema, parameter, or registry error.
        [[nodiscard]] std::expected<std::unique_ptr<transform::FittedInputTransform>, GprError>
            decode_fitted_input_leaf(
                const nlohmann::json& j, const PersistRegistry& registry, CompositeItems& composite
            )
        {
            if (j.is_string() && j.get<std::string>() == "identity")
            {
                return std::make_unique<transform::FittedIdentityInput>();
            }
            if (!j.is_object() || j.size() != 1U)
            {
                return std::unexpected(config_err("invalid fitted input transform JSON"));
            }
            const std::string& key = j.begin().key();
            const nlohmann::json& body = j.begin().value();
            if (key == "standardize")
            {
                if (!body.is_object() || !body.contains("mean") || !body.contains("std") || !body["mean"].is_array()
                    || !body["std"].is_array())
                {
                    return std::unexpected(config_err("fitted standardize missing mean/std"));
                }
                auto st = transform::FittedStandardizeInput::from_parts(
                    body["mean"].get<std::vector<double>>(), body["std"].get<std::vector<double>>()
                );
                if (!st.has_value())
                {
                    return std::unexpected(std::move(st).error());
                }
                return std::make_unique<transform::FittedStandardizeInput>(std::move(*st));
            }
            if (key == "min_max")
            {
                if (!body.is_object() || !body.contains("data_min") || !body.contains("data_max")
                    || !body.contains("range_lo") || !body.contains("range_hi"))
                {
                    return std::unexpected(config_err("fitted min_max missing fields"));
                }
                auto mm = transform::FittedMinMaxInput::from_parts(
                    body["data_min"].get<std::vector<double>>(), body["data_max"].get<std::vector<double>>(),
                    body["range_lo"].get<double>(), body["range_hi"].get<double>()
                );
                if (!mm.has_value())
                {
                    return std::unexpected(std::move(mm).error());
                }
                return std::make_unique<transform::FittedMinMaxInput>(std::move(*mm));
            }
            if (key == "pipeline" || key == "columnwise")
            {
                const std::string_view arr_key = (key == "pipeline") ? "steps" : "maps";
                if (!body.is_object() || !body.contains(arr_key) || !body[arr_key].is_array())
                {
                    return std::unexpected(config_err("fitted composite transform missing array"));
                }
                composite = CompositeItems{.items = &body[arr_key], .pipeline = key == "pipeline"};
                return nullptr;
            }
            if (key == "custom")
            {
                if (!body.is_object() || !body.contains("persist_id") || !body["persist_id"].is_string()
                    || !body.contains("state"))
                {
                    return std::unexpected(config_err("custom fitted input missing persist_id/state"));
                }
                const std::string pid = body["persist_id"].get<std::string>();
                const std::string st =
                    body["state"].is_string() ? body["state"].get<std::string>() : body["state"].dump();
                return registry.create_fitted_input(pid, st);
            }
            return std::unexpected(config_err("unknown fitted input transform"));
        }

        /// @brief Builds a FittedPipeline or FittedColumnwiseInput from its decoded children.
        /// @param[in] pipeline True for a FittedPipeline, false for a FittedColumnwiseInput.
        /// @param[in] items Decoded children in order, moved from.
        /// @param[out] out Destination of the composite transform.
        CXXGPR_NOINLINE void make_fitted_input_composite(
            bool pipeline,
            std::vector<std::unique_ptr<transform::FittedInputTransform>>&& items,
            std::unique_ptr<transform::FittedInputTransform>& out
        )
        {
            if (pipeline)
            {
                out = std::make_unique<transform::FittedPipeline>(std::move(items));
                return;
            }
            out = std::make_unique<transform::FittedColumnwiseInput>(std::move(items));
        }

        /// @brief Recursively decodes a FittedInputTransform hierarchy and validates restored statistics.
        /// @details A nesting level keeps only the decoded children on the stack: node decoding, composite
        /// construction, and error construction run in frames of their own.
        /// @param[in] j JSON string ("identity") or single-key fitted transform descriptor object.
        /// @param[in] registry Registry supplying factories for "custom" transforms.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @param[out] out Destination of the owned FittedInputTransform instance.
        /// @param[out] err Error slot that receives the first schema, parameter, depth, or registry error.
        void decode_fitted_input_node(
            const nlohmann::json& j,
            const PersistRegistry& registry,
            std::size_t depth,
            std::unique_ptr<transform::FittedInputTransform>& out,
            ErrorSlot& err
        )
        {
            if (depth > k_max_nesting_depth)
            {
                fail_config(err, "transform nesting depth exceeds 128");
                return;
            }
            CompositeItems composite;
            produce_into(out, err, [&] { return decode_fitted_input_leaf(j, registry, composite); });
            if (err.has_value() || composite.items == nullptr)
            {
                return;
            }
            // Decode FittedPipeline ("steps") and FittedColumnwiseInput ("maps") children uniformly with depth + 1.
            std::vector<std::unique_ptr<transform::FittedInputTransform>> items;
            for (const auto& child : *composite.items)
            {
                std::unique_ptr<transform::FittedInputTransform> item;
                decode_fitted_input_node(child, registry, depth + 1U, item, err);
                if (err.has_value())
                {
                    return;
                }
                items.push_back(std::move(item));
            }
            make_fitted_input_composite(composite.pipeline, std::move(items), out);
        }

        /// @brief Decodes an UnfittedTargetTransform node other than a pipeline, or finds the steps of a pipeline.
        /// @param[in] j JSON string ("identity") or single-key target transform descriptor object.
        /// @param[in] registry Registry supplying factories for "custom" target transforms.
        /// @param[out] composite Steps of j when j is a valid "pipeline" node; left null otherwise.
        /// @return Owned UnfittedTargetTransform instance, a null pointer when j is a pipeline, or GprError on schema,
        /// range, or registry error.
        [[nodiscard]] std::expected<std::unique_ptr<transform::UnfittedTargetTransform>, GprError>
            decode_unfitted_target_leaf(
                const nlohmann::json& j, const PersistRegistry& registry, CompositeItems& composite
            )
        {
            if (j.is_string() && j.get<std::string>() == "identity")
            {
                return std::make_unique<transform::IdentityTarget>();
            }
            if (!j.is_object() || j.size() != 1U)
            {
                return std::unexpected(config_err("invalid unfitted target transform JSON"));
            }
            const std::string& key = j.begin().key();
            const nlohmann::json& body = j.begin().value();
            if (key == "standardize")
            {
                return std::make_unique<transform::StandardizeTarget>();
            }
            if (key == "min_max")
            {
                if (!body.is_object() || !body.contains("range_lo") || !body.contains("range_hi"))
                {
                    return std::unexpected(config_err("min_max target missing range_lo/range_hi"));
                }
                auto mm =
                    transform::MinMaxTarget::create(body["range_lo"].get<double>(), body["range_hi"].get<double>());
                if (!mm.has_value())
                {
                    return std::unexpected(std::move(mm).error());
                }
                return std::make_unique<transform::MinMaxTarget>(*mm);
            }
            if (key == "pipeline")
            {
                if (!body.is_object() || !body.contains("steps") || !body["steps"].is_array())
                {
                    return std::unexpected(config_err("target pipeline missing steps"));
                }
                // Hand the steps to the caller, which decodes them.
                composite = CompositeItems{.items = &body["steps"], .pipeline = true};
                return nullptr;
            }
            if (key == "custom")
            {
                if (!body.is_object() || !body.contains("persist_id") || !body["persist_id"].is_string()
                    || !body.contains("state"))
                {
                    return std::unexpected(config_err("custom target missing persist_id/state"));
                }
                const std::string pid = body["persist_id"].get<std::string>();
                const std::string st =
                    body["state"].is_string() ? body["state"].get<std::string>() : body["state"].dump();
                return registry.create_unfitted_target(pid, st);
            }
            return std::unexpected(config_err("unknown unfitted target transform"));
        }

        /// @brief Builds a TargetPipeline from its decoded steps.
        /// @param[in] steps Decoded steps in order, moved from.
        /// @param[out] out Destination of the pipeline.
        CXXGPR_NOINLINE void make_unfitted_target_pipeline(
            std::vector<std::unique_ptr<transform::UnfittedTargetTransform>>&& steps,
            std::unique_ptr<transform::UnfittedTargetTransform>& out
        )
        {
            out = std::make_unique<transform::TargetPipeline>(std::move(steps));
        }

        /// @brief Recursively decodes an UnfittedTargetTransform hierarchy from JSON.
        /// @details A nesting level keeps only the decoded steps on the stack: node decoding, pipeline
        /// construction, and error construction run in frames of their own.
        /// @param[in] j JSON string ("identity") or single-key target transform descriptor object.
        /// @param[in] registry Registry supplying factories for "custom" target transforms.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @param[out] out Destination of the owned UnfittedTargetTransform instance.
        /// @param[out] err Error slot that receives the first schema, range, depth, or registry error.
        void decode_unfitted_target_node(
            const nlohmann::json& j,
            const PersistRegistry& registry,
            std::size_t depth,
            std::unique_ptr<transform::UnfittedTargetTransform>& out,
            ErrorSlot& err
        )
        {
            if (depth > k_max_nesting_depth)
            {
                fail_config(err, "transform nesting depth exceeds 128");
                return;
            }
            CompositeItems composite;
            produce_into(out, err, [&] { return decode_unfitted_target_leaf(j, registry, composite); });
            if (err.has_value() || composite.items == nullptr)
            {
                return;
            }
            // Decode TargetPipeline steps in order with depth + 1.
            std::vector<std::unique_ptr<transform::UnfittedTargetTransform>> steps;
            for (const auto& child : *composite.items)
            {
                std::unique_ptr<transform::UnfittedTargetTransform> step;
                decode_unfitted_target_node(child, registry, depth + 1U, step, err);
                if (err.has_value())
                {
                    return;
                }
                steps.push_back(std::move(step));
            }
            make_unfitted_target_pipeline(std::move(steps), out);
        }

        /// @brief Decodes a FittedTargetTransform node other than a pipeline, or finds the steps of a pipeline.
        /// @param[in] j JSON string ("identity") or single-key fitted target transform descriptor object.
        /// @param[in] registry Registry supplying factories for "custom" target transforms.
        /// @param[out] composite Steps of j when j is a valid "pipeline" node; left null otherwise.
        /// @return Owned FittedTargetTransform instance with validated statistics, a null pointer when j is a
        /// pipeline, or GprError on schema, parameter, or registry error.
        [[nodiscard]] std::expected<std::unique_ptr<transform::FittedTargetTransform>, GprError>
            decode_fitted_target_leaf(
                const nlohmann::json& j, const PersistRegistry& registry, CompositeItems& composite
            )
        {
            if (j.is_string() && j.get<std::string>() == "identity")
            {
                return std::make_unique<transform::FittedIdentityTarget>();
            }
            if (!j.is_object() || j.size() != 1U)
            {
                return std::unexpected(config_err("invalid fitted target transform JSON"));
            }
            const std::string& key = j.begin().key();
            const nlohmann::json& body = j.begin().value();
            if (key == "standardize")
            {
                if (!body.is_object() || !body.contains("mean") || !body.contains("std"))
                {
                    return std::unexpected(config_err("fitted standardize target missing mean/std"));
                }
                auto st = transform::FittedStandardizeTarget::from_parts(
                    body["mean"].get<double>(), body["std"].get<double>()
                );
                if (!st.has_value())
                {
                    return std::unexpected(std::move(st).error());
                }
                return std::make_unique<transform::FittedStandardizeTarget>(*st);
            }
            if (key == "min_max")
            {
                if (!body.is_object() || !body.contains("data_min") || !body.contains("data_max")
                    || !body.contains("range_lo") || !body.contains("range_hi"))
                {
                    return std::unexpected(config_err("fitted min_max target missing fields"));
                }
                auto mm = transform::FittedMinMaxTarget::from_parts(
                    body["data_min"].get<double>(), body["data_max"].get<double>(), body["range_lo"].get<double>(),
                    body["range_hi"].get<double>()
                );
                if (!mm.has_value())
                {
                    return std::unexpected(std::move(mm).error());
                }
                return std::make_unique<transform::FittedMinMaxTarget>(*mm);
            }
            if (key == "pipeline")
            {
                if (!body.is_object() || !body.contains("steps") || !body["steps"].is_array())
                {
                    return std::unexpected(config_err("fitted target pipeline missing steps"));
                }
                // Hand the steps to the caller, which decodes them.
                composite = CompositeItems{.items = &body["steps"], .pipeline = true};
                return nullptr;
            }
            if (key == "custom")
            {
                if (!body.is_object() || !body.contains("persist_id") || !body["persist_id"].is_string()
                    || !body.contains("state"))
                {
                    return std::unexpected(config_err("custom fitted target missing persist_id/state"));
                }
                const std::string pid = body["persist_id"].get<std::string>();
                const std::string st =
                    body["state"].is_string() ? body["state"].get<std::string>() : body["state"].dump();
                return registry.create_fitted_target(pid, st);
            }
            return std::unexpected(config_err("unknown fitted target transform"));
        }

        /// @brief Builds a FittedTargetPipeline from its decoded steps.
        /// @param[in] steps Decoded steps in order, moved from.
        /// @param[out] out Destination of the pipeline.
        CXXGPR_NOINLINE void make_fitted_target_pipeline(
            std::vector<std::unique_ptr<transform::FittedTargetTransform>>&& steps,
            std::unique_ptr<transform::FittedTargetTransform>& out
        )
        {
            out = std::make_unique<transform::FittedTargetPipeline>(std::move(steps));
        }

        /// @brief Recursively decodes a FittedTargetTransform hierarchy and validates restored statistics.
        /// @details A nesting level keeps only the decoded steps on the stack: node decoding, pipeline
        /// construction, and error construction run in frames of their own.
        /// @param[in] j JSON string ("identity") or single-key fitted target transform descriptor object.
        /// @param[in] registry Registry supplying factories for "custom" target transforms.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @param[out] out Destination of the owned FittedTargetTransform instance.
        /// @param[out] err Error slot that receives the first schema, parameter, depth, or registry error.
        void decode_fitted_target_node(
            const nlohmann::json& j,
            const PersistRegistry& registry,
            std::size_t depth,
            std::unique_ptr<transform::FittedTargetTransform>& out,
            ErrorSlot& err
        )
        {
            if (depth > k_max_nesting_depth)
            {
                fail_config(err, "transform nesting depth exceeds 128");
                return;
            }
            CompositeItems composite;
            produce_into(out, err, [&] { return decode_fitted_target_leaf(j, registry, composite); });
            if (err.has_value() || composite.items == nullptr)
            {
                return;
            }
            // Decode FittedTargetPipeline steps in order with depth + 1.
            std::vector<std::unique_ptr<transform::FittedTargetTransform>> steps;
            for (const auto& child : *composite.items)
            {
                std::unique_ptr<transform::FittedTargetTransform> step;
                decode_fitted_target_node(child, registry, depth + 1U, step, err);
                if (err.has_value())
                {
                    return;
                }
                steps.push_back(std::move(step));
            }
            make_fitted_target_pipeline(std::move(steps), out);
        }

        /// @brief Serializes a JitterPolicy (FixedJitter or AdaptiveJitter) into a tagged JSON object.
        /// @param[in] jitter Cholesky jitter stabilization policy.
        /// @return JSON representation of jitter.
        [[nodiscard]] nlohmann::json encode_jitter(const JitterPolicy& jitter)
        {
            if (const auto fx = jitter.as_fixed(); fx.has_value())
            {
                return nlohmann::json{
                    {"fixed", {{"jitter", fx->jitter()}}}
                };
            }
            const auto ad = jitter.as_adaptive();
            if (!ad.has_value())
            {
                return nlohmann::json{};
            }
            return nlohmann::json{
                {"adaptive",
                 {{"initial", ad->initial()}, {"multiplier", ad->multiplier()}, {"max_retries", ad->max_retries()},
                        {"max_jitter", ad->max_jitter()}}}
            };
        }

        /// @brief Parses and validates a JitterPolicy JSON descriptor.
        /// @param[in] j Single-key JSON object ("fixed" or "adaptive").
        /// @return Validated JitterPolicy, or GprError on invalid schema or parameter values.
        [[nodiscard]] std::expected<JitterPolicy, GprError> decode_jitter(const nlohmann::json& j)
        {
            if (!j.is_object() || j.size() != 1U)
            {
                return std::unexpected(config_err("jitter must be a single-key object"));
            }
            if (j.contains("fixed"))
            {
                const auto& b = j["fixed"];
                if (!b.is_object() || !b.contains("jitter") || !b["jitter"].is_number())
                {
                    return std::unexpected(config_err("fixed jitter missing numeric jitter"));
                }
                return JitterPolicy::fixed(b["jitter"].get<double>());
            }
            if (j.contains("adaptive"))
            {
                const auto& b = j["adaptive"];
                if (!b.is_object() || !b.contains("initial") || !b.contains("multiplier") || !b.contains("max_retries")
                    || !b.contains("max_jitter"))
                {
                    return std::unexpected(config_err("adaptive jitter missing required fields"));
                }
                return JitterPolicy::adaptive(
                    b["initial"].get<double>(), b["multiplier"].get<double>(), b["max_retries"].get<std::size_t>(),
                    b["max_jitter"].get<double>()
                );
            }
            return std::unexpected(config_err("unknown jitter policy"));
        }

        /// @brief Encodes a supplied-distance or coordinate leaf of a described DistanceKernel tree.
        /// @param[in] node Described leaf node.
        /// @param[out] out Destination of the JSON of the leaf.
        /// @param[out] err Error slot that receives the failure when the leaf is missing or not persistable.
        CXXGPR_NOINLINE void encode_distance_leaf(
            const cxxgpr::detail::DistanceNodeDesc& node, nlohmann::json& out, ErrorSlot& err
        )
        {
            if (!node.leaf.has_value())
            {
                fail_config(err, "distance kernel leaf is missing");
                return;
            }
            auto leaf = encode_leaf(*node.leaf);
            if (!leaf.has_value())
            {
                err.emplace(std::move(leaf).error());
                return;
            }
            if (node.kind == cxxgpr::detail::DistanceNodeDesc::Kind::Coord)
            {
                out = std::move(*leaf);
                return;
            }
            out = nlohmann::json{
                {"distance", {{"slot", node.slot}, {"leaf", std::move(*leaf)}}}
            };
        }

        /// @brief Recursively encodes a described DistanceKernel subtree into JSON.
        /// @details A nesting level keeps only its two encoded children on the stack: leaf encoding, object
        /// construction, and error construction run in frames of their own.
        /// @param[in] node Described subtree.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @param[out] out Destination of the JSON of the subtree.
        /// @param[out] err Error slot that receives the first failure: a leaf that is not persistable or nesting
        /// that is too deep.
        void encode_distance_node(
            const cxxgpr::detail::DistanceNodeDesc& node, std::size_t depth, nlohmann::json& out, ErrorSlot& err
        )
        {
            using Kind = cxxgpr::detail::DistanceNodeDesc::Kind;
            if (depth > k_max_nesting_depth)
            {
                fail_config(err, "kernel nesting depth exceeds 128");
                return;
            }
            if (node.kind == Kind::Supplied || node.kind == Kind::Coord)
            {
                encode_distance_leaf(node, out, err);
                return;
            }
            nlohmann::json lhs;
            encode_distance_node(*node.lhs, depth + 1U, lhs, err);
            if (err.has_value())
            {
                return;
            }
            nlohmann::json rhs;
            encode_distance_node(*node.rhs, depth + 1U, rhs, err);
            if (err.has_value())
            {
                return;
            }
            make_composite_kernel_json(node.kind == Kind::Sum ? "sum" : "product", std::move(lhs), std::move(rhs), out);
        }

        /// @brief Decodes the kernel JSON of a distance-tree leaf, which must be a single kernel leaf.
        /// @param[in] leaf_json Kernel JSON node of the leaf.
        /// @param[in] registry Registry used to recreate custom leaves.
        /// @return The kernel leaf, or GprError on schema or registry failure or when the node is a composite.
        [[nodiscard]] std::expected<KernelLeaf, GprError> decode_distance_leaf_kernel(
            const nlohmann::json& leaf_json, const PersistRegistry& registry
        )
        {
            std::optional<KernelSpec> spec;
            ErrorSlot err;
            decode_kernel_node(leaf_json, registry, 0U, spec, err);
            if (err.has_value())
            {
                return std::unexpected(std::move(*err));
            }
            const auto* leaf =
                spec.has_value() ? std::get_if<KernelLeaf>(&KernelSpecPersistAccess::node(*spec)) : nullptr;
            if (leaf == nullptr)
            {
                return std::unexpected(config_err("a distance leaf must be a single kernel leaf"));
            }
            return *leaf;
        }

        /// @brief Children of a sum or product node found by decode_distance_leaf().
        struct DistanceChildren
        {
            /// @brief Left child node, or null when the node is not a composite.
            const nlohmann::json* left = nullptr;
            /// @brief Right child node, or null when the node is not a composite.
            const nlohmann::json* right = nullptr;
            /// @brief True for a "sum" node, false for a "product" node.
            bool sum = false;
        };

        /// @brief Decodes a distance-tree node other than a composite, or finds the children of a composite.
        /// @param[in] j JSON node: sum, product, distance leaf, or coordinate leaf.
        /// @param[in] registry Registry used to recreate custom leaves.
        /// @param[out] children Children of j when j is a valid "sum" or "product" node; left null otherwise.
        /// @return Described leaf node, a null pointer when j is a composite, or GprError on schema or registry
        /// failure.
        [[nodiscard]] std::expected<std::shared_ptr<const cxxgpr::detail::DistanceNodeDesc>, GprError>
            decode_distance_leaf(const nlohmann::json& j, const PersistRegistry& registry, DistanceChildren& children)
        {
            using Kind = cxxgpr::detail::DistanceNodeDesc::Kind;
            if (!j.is_object() || j.size() != 1U)
            {
                return std::unexpected(config_err("kernel node must be a single-key JSON object"));
            }
            const auto it = j.begin();
            const std::string& key = it.key();
            const nlohmann::json& body = it.value();
            if (key == "sum" || key == "product")
            {
                if (!body.is_object() || !body.contains("left") || !body.contains("right"))
                {
                    return std::unexpected(config_err("composite kernel node missing left/right"));
                }
                // Hand the children to the caller, which decodes them.
                children = DistanceChildren{.left = &body["left"], .right = &body["right"], .sum = key == "sum"};
                return nullptr;
            }
            auto node = std::make_shared<cxxgpr::detail::DistanceNodeDesc>();
            if (key == "distance")
            {
                if (!body.is_object() || !body.contains("slot") || !body["slot"].is_number_unsigned()
                    || !body.contains("leaf"))
                {
                    return std::unexpected(config_err("distance node needs an unsigned slot and a leaf"));
                }
                auto leaf = decode_distance_leaf_kernel(body["leaf"], registry);
                if (!leaf.has_value())
                {
                    return std::unexpected(std::move(leaf).error());
                }
                node->kind = Kind::Supplied;
                node->slot = body["slot"].get<std::size_t>();
                node->leaf = std::move(*leaf);
                return node;
            }
            auto leaf = decode_distance_leaf_kernel(j, registry);
            if (!leaf.has_value())
            {
                return std::unexpected(std::move(leaf).error());
            }
            node->kind = Kind::Coord;
            node->leaf = std::move(*leaf);
            return node;
        }

        /// @brief Builds a described sum or product node from its decoded children.
        /// @param[in] sum True for a sum node, false for a product node.
        /// @param[in] lhs Decoded left child, moved from.
        /// @param[in] rhs Decoded right child, moved from.
        /// @param[out] out Destination of the composite node.
        CXXGPR_NOINLINE void make_distance_composite(
            bool sum,
            std::shared_ptr<const cxxgpr::detail::DistanceNodeDesc>&& lhs,
            std::shared_ptr<const cxxgpr::detail::DistanceNodeDesc>&& rhs,
            std::shared_ptr<const cxxgpr::detail::DistanceNodeDesc>& out
        )
        {
            using Kind = cxxgpr::detail::DistanceNodeDesc::Kind;
            auto node = std::make_shared<cxxgpr::detail::DistanceNodeDesc>();
            node->kind = sum ? Kind::Sum : Kind::Product;
            node->lhs = std::move(lhs);
            node->rhs = std::move(rhs);
            out = std::move(node);
        }

        /// @brief Recursively decodes one kernel JSON node into a described DistanceKernel subtree.
        /// @details A nesting level keeps only its two decoded children on the stack: node decoding, composite
        /// construction, and error construction run in frames of their own.
        /// @param[in] j JSON node: sum, product, distance leaf, or coordinate leaf.
        /// @param[in] registry Registry used to recreate custom leaves.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @param[out] out Destination of the described subtree.
        /// @param[out] err Error slot that receives the first schema, depth, or registry error.
        void decode_distance_node(
            const nlohmann::json& j,
            const PersistRegistry& registry,
            std::size_t depth,
            std::shared_ptr<const cxxgpr::detail::DistanceNodeDesc>& out,
            ErrorSlot& err
        )
        {
            if (depth > k_max_nesting_depth)
            {
                fail_config(err, "kernel nesting depth exceeds 128");
                return;
            }
            DistanceChildren children;
            produce_into(out, err, [&] { return decode_distance_leaf(j, registry, children); });
            if (err.has_value() || children.left == nullptr)
            {
                return;
            }
            std::shared_ptr<const cxxgpr::detail::DistanceNodeDesc> lhs;
            decode_distance_node(*children.left, registry, depth + 1U, lhs, err);
            if (err.has_value())
            {
                return;
            }
            std::shared_ptr<const cxxgpr::detail::DistanceNodeDesc> rhs;
            decode_distance_node(*children.right, registry, depth + 1U, rhs, err);
            if (err.has_value())
            {
                return;
            }
            make_distance_composite(children.sum, std::move(lhs), std::move(rhs), out);
        }

        /// @brief Parses the distance section and the distance kernel of a supplied-distance checkpoint.
        /// @param[in] root Parsed config.json root object.
        /// @param[in] registry Registry used to recreate custom leaves.
        /// @param[in,out] cfg Configuration receiving the distance kernel description and the mode.
        /// @return Empty expected on success, or GprError when the section is malformed or inconsistent.
        [[nodiscard]] std::expected<void, GprError> parse_distance_section(
            const nlohmann::json& root, const PersistRegistry& registry, ParsedConfig& cfg
        )
        {
            const auto& dj = root["distance"];
            if (!dj.is_object() || !dj.contains("points") || !dj["points"].is_string() || !dj.contains("slots")
                || !dj["slots"].is_array())
            {
                return std::unexpected(config_err("distance section needs points and slots"));
            }
            const std::string points = dj["points"].get<std::string>();
            if (points != "distance_only" && points != "with_points")
            {
                return std::unexpected(config_err("distance.points must be distance_only or with_points"));
            }
            cfg.distance_with_points = points == "with_points";
            if (cfg.distance_with_points != (cfg.d > 0U))
            {
                return std::unexpected(config_err("d must be positive exactly for a with_points model"));
            }
            // A distance-only model has no coordinates, so both coordinate transforms must be the identity.
            if (!cfg.distance_with_points
                && (!root.contains("x_unfitted") || root["x_unfitted"] != nlohmann::json("identity")
                    || !root.contains("x_transform") || root["x_transform"] != nlohmann::json("identity")))
            {
                return std::unexpected(
                    config_err("a distance_only model must have identity x_unfitted and x_transform")
                );
            }
            cxxgpr::detail::DistanceKernelDesc desc;
            for (const auto& s : dj["slots"])
            {
                if (!s.is_object() || !s.contains("kind") || !s["kind"].is_string())
                {
                    return std::unexpected(config_err("distance slot needs a kind"));
                }
                const std::string kind = s["kind"].get<std::string>();
                if (kind == "scalar")
                {
                    desc.slots.push_back(cxxgpr::detail::DistanceSlotDesc{.ard_dims = std::nullopt});
                }
                else if (kind == "ard" && s.contains("dims") && s["dims"].is_number_unsigned()
                    && s["dims"].get<std::size_t>() > 0U)
                {
                    desc.slots.push_back(cxxgpr::detail::DistanceSlotDesc{.ard_dims = s["dims"].get<std::size_t>()});
                }
                else
                {
                    return std::unexpected(config_err("distance slot kind must be scalar or ard with dims >= 1"));
                }
            }
            ErrorSlot err;
            decode_distance_node(root["kernel"], registry, 0U, desc.root, err);
            if (err.has_value())
            {
                return std::unexpected(std::move(*err));
            }
            cfg.distance_kernel = std::move(desc);
            return {};
        }

    }  // namespace

    /// @brief Serializes a SaveInput bundle into pretty-printed config.json text.
    /// @details Encodes kernel AST, likelihood with bounds, jitter policy, unfitted/fitted transforms,
    /// precision/residual flags, exponential math mode, and distance cache policy.
    /// @param[in] input Borrowed model configuration and transform references to serialize.
    /// @return Formatted JSON text on success, or GprError if any subtree or custom component fails encoding.
    /// @retval GprErrorCode::PersistFailed When custom kernels/transforms are non-persistable or depth exceeds 128.
    std::expected<std::string, GprError> serialize_config_json(const SaveInput& input)
    {
        // Encode the kernel and transform trees in order; each pass stops at its first error.
        ErrorSlot err;
        nlohmann::json k_json;
        encode_kernel_node(input.kernel_spec, 0U, k_json, err);
        if (err.has_value())
        {
            return std::unexpected(std::move(*err));
        }
        nlohmann::json x_unf;
        encode_unfitted_input_node(input.unfitted_input, 0U, x_unf, err);
        if (err.has_value())
        {
            return std::unexpected(std::move(*err));
        }
        nlohmann::json x_fit;
        encode_fitted_input_node(input.fitted_input, 0U, x_fit, err);
        if (err.has_value())
        {
            return std::unexpected(std::move(*err));
        }
        nlohmann::json y_unf;
        encode_unfitted_target_node(input.unfitted_target, 0U, y_unf, err);
        if (err.has_value())
        {
            return std::unexpected(std::move(*err));
        }
        nlohmann::json y_fit;
        encode_fitted_target_node(input.fitted_target, 0U, y_fit, err);
        if (err.has_value())
        {
            return std::unexpected(std::move(*err));
        }

        // Map PersistPrecisionKind into config.json "precision" and conditional "residual" strings.
        std::string prec_str = "double";
        std::string res_str = "promote_storage";
        switch (input.precision_kind)
        {
            case PersistPrecisionKind::Double:
                prec_str = "double";
                break;
            case PersistPrecisionKind::Single:
                prec_str = "single";
                break;
            case PersistPrecisionKind::MixedPromote:
                prec_str = "mixed";
                res_str = "promote_storage";
                break;
            case PersistPrecisionKind::MixedReevaluate:
                prec_str = "mixed";
                res_str = "reevaluate_kernel";
                break;
        }

        nlohmann::json root{
            {"format_version",                                     FORMAT_VERSION                              },
            {             "n",                                                                    input.n_train},
            {             "d",                                                                 input.n_features},
            {    "has_factor",                                                                 input.has_factor},
            {   "factor_kind",                                                                            "llt"},
            { "factor_jitter",                                                              input.factor_jitter},
            {     "precision",                                                                         prec_str},
            {          "math",           input.kernel_exp == KernelExp::FastApprox ? "fast_approx" : "accurate"},
            {"distance_cache",       input.distance_cache == DistanceCachePolicy::Uncached ? "never" : "always"},
            {        "kernel",                                                                std::move(k_json)},
            {    "likelihood",
             {{"noise_variance", encode_bounded(input.likelihood.noise_variance(), input.likelihood.bounds())}}},
            {        "jitter",                                                      encode_jitter(input.jitter)},
            {    "x_unfitted",                                                                 std::move(x_unf)},
            {   "x_transform",                                                                 std::move(x_fit)},
            {    "y_unfitted",                                                                 std::move(y_unf)},
            {   "y_transform",                                                                 std::move(y_fit)},
        };
        if (input.precision_kind == PersistPrecisionKind::MixedPromote
            || input.precision_kind == PersistPrecisionKind::MixedReevaluate)
        {
            root["residual"] = res_str;
        }
        // Refuse to write a document that the loader would reject for its nesting depth, before the recursive
        // serializer descends into it once per level.
        if (auto depth_ok = check_value_max_depth(root, MAX_JSON_NESTING_DEPTH); !depth_ok.has_value())
        {
            return std::unexpected(std::move(depth_ok).error());
        }
        return root.dump(2);
    }

    /// @brief Inserts a top-level `save_id` field into an already-serialized config.json document.
    /// @details Re-parses the already-validated config.json string without re-invoking custom `persist_state_json()`
    /// callbacks, attaches `save_id`, and returns the pretty-printed JSON text.
    /// @param[in] config_json Pretty-printed config.json text produced by serialize_config_json or
    /// serialize_distance_config_json.
    /// @param[in] save_id Computed pairing identifier to store as top-level `save_id`.
    /// @return Formatted JSON string containing `save_id`, or GprError if config_json is not a valid JSON object.
    /// @retval GprErrorCode::PersistFailed When config_json cannot be parsed as a JSON object.
    std::expected<std::string, GprError> add_save_id_to_config_json(
        std::string_view config_json, std::string_view save_id
    )
    {
        nlohmann::json root = nlohmann::json::parse(config_json, nullptr, false);
        if (root.is_discarded() || !root.is_object())
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Config, "failed to re-parse serialized config.json")
            );
        }
        root["save_id"] = std::string(save_id);
        return root.dump(2);
    }

    /// @brief Reads, depth-checks, parses, and validates config.json into a ParsedConfig struct.
    /// @details Enforces MAX_JSON_NESTING_DEPTH prior to JSON parsing, checks model/distance checkpoint
    /// compatibility, validates format_version and factor_kind == "llt", and deserializes kernels,
    /// likelihood, jitter, and transforms.
    /// @param[in] path Filesystem path to config.json.
    /// @param[in] registry Registry supplying custom kernel and transform deserialization factories.
    /// @param[in] distance True to parse a supplied-distance checkpoint, false for a coordinate checkpoint.
    /// @return Validated ParsedConfig on success, or GprError on I/O, JSON, version, or schema failure.
    /// @retval GprErrorCode::PersistFailed When file read, JSON depth/syntax, schema, model kind, or registry fails.
    /// @retval GprErrorCode::UnsupportedPersistVersion When format_version != FORMAT_VERSION.
    static std::expected<ParsedConfig, GprError> parse_config_impl(
        const std::filesystem::path& path, const PersistRegistry& registry, bool distance
    )
    {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs)
        {
            return std::unexpected(GprError::persist_failed(PersistErrorKind::Io, "failed to open config.json"));
        }
        std::ostringstream oss;
        oss << ifs.rdbuf();
        const std::string json_text = oss.str();
        // Guard against excessively nested JSON inputs before invoking nlohmann::json::parse.
        if (auto depth_ok = check_json_max_depth(json_text, MAX_JSON_NESTING_DEPTH, PersistErrorKind::Config);
            !depth_ok.has_value())
        {
            return std::unexpected(std::move(depth_ok).error());
        }
        nlohmann::json root;
        try
        {
            root = nlohmann::json::parse(json_text);
        }
        catch (const nlohmann::json::exception&)
        {
            return std::unexpected(config_err("invalid JSON syntax in config.json"));
        }
        if (!root.is_object())
        {
            return std::unexpected(config_err("config.json root must be a JSON object"));
        }

        // Reject non-Exact GPR checkpoints (e.g., sparse or distance-supplied models) cleanly with WrongModel.
        if (root.contains("model"))
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::WrongModel, "model is not an Exact GPR checkpoint")
            );
        }
        if (root.contains("distance") != distance)
        {
            return std::unexpected(
                GprError::persist_failed(
                    PersistErrorKind::WrongModel,
                    distance ? "checkpoint is not a supplied-distance model"
                             : "distance-supplied model requires LoadedDistanceGpr"
                )
            );
        }

        if (!root.contains("format_version") || !root["format_version"].is_number_integer())
        {
            return std::unexpected(config_err("missing or non-integer format_version"));
        }
        const auto ver_i64 = root["format_version"].get<std::int64_t>();
        if (std::cmp_not_equal(ver_i64, FORMAT_VERSION))
        {
            return std::unexpected(
                GprError::unsupported_persist_version(
                    static_cast<std::uint32_t>(std::max<std::int64_t>(0, ver_i64)), FORMAT_VERSION
                )
            );
        }

        if (!root.contains("factor_kind") || !root["factor_kind"].is_string())
        {
            return std::unexpected(config_err("missing or non-string factor_kind"));
        }
        if (root["factor_kind"].get<std::string>() != "llt")
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::WrongModel, "only factor_kind == \"llt\" is supported")
            );
        }

        if (!root.contains("n") || !root["n"].is_number_unsigned() || !root.contains("d")
            || !root["d"].is_number_unsigned() || !root.contains("has_factor") || !root["has_factor"].is_boolean())
        {
            return std::unexpected(config_err("missing or invalid n, d, or has_factor"));
        }

        ParsedConfig cfg;
        if (root.contains("save_id"))
        {
            if (!root["save_id"].is_string())
            {
                return std::unexpected(config_err("save_id must be a string"));
            }
            const auto& sid = root["save_id"].get_ref<const std::string&>();
            if (!is_valid_save_id(sid))
            {
                return std::unexpected(config_err("save_id must be 32 lowercase hexadecimal digits"));
            }
            cfg.save_id = sid;
        }
        cfg.n = root["n"].get<std::size_t>();
        cfg.d = root["d"].get<std::size_t>();
        if (cfg.n == 0U || (!distance && cfg.d == 0U))
        {
            return std::unexpected(config_err("n and d must be positive for coordinate Exact GPR"));
        }
        cfg.has_factor = root["has_factor"].get<bool>();

        if (root.contains("factor_jitter"))
        {
            if (!root["factor_jitter"].is_number())
            {
                return std::unexpected(config_err("factor_jitter must be numeric"));
            }
            cfg.factor_jitter = root["factor_jitter"].get<double>();
            if (!std::isfinite(cfg.factor_jitter) || cfg.factor_jitter < 0.0)
            {
                return std::unexpected(config_err("factor_jitter must be finite and non-negative"));
            }
        }

        // Decode precision and mixed-precision residual formula settings.
        const auto prec_opt = optional_string(root, "precision", "double");
        if (!prec_opt.has_value())
        {
            return std::unexpected(prec_opt.error());
        }
        const std::string& prec = *prec_opt;
        const auto res_opt = optional_string(root, "residual", "promote_storage");
        if (!res_opt.has_value())
        {
            return std::unexpected(res_opt.error());
        }
        const std::string& res = *res_opt;
        if (prec == "double")
        {
            cfg.precision_kind = PersistPrecisionKind::Double;
        }
        else if (prec == "single")
        {
            cfg.precision_kind = PersistPrecisionKind::Single;
        }
        else if (prec == "mixed")
        {
            if (res == "promote_storage")
            {
                cfg.precision_kind = PersistPrecisionKind::MixedPromote;
            }
            else if (res == "reevaluate_kernel")
            {
                cfg.precision_kind = PersistPrecisionKind::MixedReevaluate;
            }
            else
            {
                return std::unexpected(config_err("unknown residual strategy"));
            }
        }
        else
        {
            return std::unexpected(config_err("unknown precision kind"));
        }

        const auto math_s_opt = optional_string(root, "math", "accurate");
        if (!math_s_opt.has_value())
        {
            return std::unexpected(math_s_opt.error());
        }
        const std::string& math_s = *math_s_opt;
        if (math_s == "accurate")
        {
            cfg.kernel_exp = KernelExp::Accurate;
        }
        else if (math_s == "fast_approx")
        {
            cfg.kernel_exp = KernelExp::FastApprox;
        }
        else
        {
            return std::unexpected(config_err("unknown math mode"));
        }

        const auto dc_s_opt = optional_string(root, "distance_cache", "always");
        if (!dc_s_opt.has_value())
        {
            return std::unexpected(dc_s_opt.error());
        }
        const std::string& dc_s = *dc_s_opt;
        if (dc_s == "always")
        {
            cfg.distance_cache = DistanceCachePolicy::Cached;
        }
        else if (dc_s == "never")
        {
            cfg.distance_cache = DistanceCachePolicy::Uncached;
        }
        else
        {
            return std::unexpected(config_err("unknown distance_cache policy"));
        }

        if (!root.contains("kernel") || !root.contains("likelihood") || !root.contains("jitter")
            || !root.contains("x_unfitted") || !root.contains("x_transform") || !root.contains("y_unfitted")
            || !root.contains("y_transform"))
        {
            return std::unexpected(config_err("missing required configuration sections"));
        }

        // Parse nested kernel, likelihood, jitter, and transform subsections inside a JSON type-error guard.
        try
        {
            ErrorSlot err;
            if (distance)
            {
                if (auto ds = parse_distance_section(root, registry, cfg); !ds.has_value())
                {
                    return std::unexpected(std::move(ds).error());
                }
            }
            else
            {
                decode_kernel_node(root["kernel"], registry, 0U, cfg.kernel_spec, err);
                if (err.has_value())
                {
                    return std::unexpected(std::move(*err));
                }
            }

            const auto& lj = root["likelihood"];
            if (!lj.is_object() || !lj.contains("noise_variance"))
            {
                return std::unexpected(config_err("likelihood missing noise_variance"));
            }
            auto nv = decode_bounded(lj["noise_variance"]);
            if (!nv.has_value())
            {
                return std::unexpected(std::move(nv).error());
            }
            auto lik = GaussianLikelihood::create(nv->first, nv->second);
            if (!lik.has_value())
            {
                return std::unexpected(std::move(lik).error());
            }
            cfg.likelihood = *lik;

            auto jit = decode_jitter(root["jitter"]);
            if (!jit.has_value())
            {
                return std::unexpected(std::move(jit).error());
            }
            cfg.jitter = *jit;

            decode_unfitted_input_node(root["x_unfitted"], registry, 0U, cfg.x_unfitted, err);
            if (err.has_value())
            {
                return std::unexpected(std::move(*err));
            }

            decode_fitted_input_node(root["x_transform"], registry, 0U, cfg.x_transform, err);
            if (err.has_value())
            {
                return std::unexpected(std::move(*err));
            }

            decode_unfitted_target_node(root["y_unfitted"], registry, 0U, cfg.y_unfitted, err);
            if (err.has_value())
            {
                return std::unexpected(std::move(*err));
            }

            decode_fitted_target_node(root["y_transform"], registry, 0U, cfg.y_transform, err);
            if (err.has_value())
            {
                return std::unexpected(std::move(*err));
            }
        }
        catch (const nlohmann::json::exception&)
        {
            return std::unexpected(config_err("invalid field types in config.json"));
        }

        return cfg;
    }

    /// @brief Serializes a supplied-distance model into pretty-printed config.json text.
    /// @details Writes the same sections as a coordinate model, with the kernel tree carrying distance nodes and an
    /// extra distance section naming the mode and the slots.
    /// @param[in] input Borrowed model configuration and transform references to serialize.
    /// @param[in] kernel Structural description of the kernel of the model.
    /// @return Formatted JSON text, or GprError when a custom component is not persistable.
    /// @retval GprErrorCode::PersistFailed When a custom kernel or transform is non-persistable or nesting is too deep.
    std::expected<std::string, GprError> serialize_distance_config_json(
        const DistanceSaveInput& input, const cxxgpr::detail::DistanceKernelDesc& kernel
    )
    {
        // Encode the kernel and transform trees in order; each pass stops at its first error.
        ErrorSlot err;
        nlohmann::json k_json;
        encode_distance_node(*kernel.root, 0U, k_json, err);
        if (err.has_value())
        {
            return std::unexpected(std::move(*err));
        }
        nlohmann::json x_unf;
        encode_unfitted_input_node(input.unfitted_input, 0U, x_unf, err);
        if (err.has_value())
        {
            return std::unexpected(std::move(*err));
        }
        nlohmann::json x_fit_json("identity");
        if (input.fitted_input != nullptr)
        {
            encode_fitted_input_node(*input.fitted_input, 0U, x_fit_json, err);
            if (err.has_value())
            {
                return std::unexpected(std::move(*err));
            }
        }
        nlohmann::json y_unf;
        encode_unfitted_target_node(input.unfitted_target, 0U, y_unf, err);
        if (err.has_value())
        {
            return std::unexpected(std::move(*err));
        }
        nlohmann::json y_fit;
        encode_fitted_target_node(input.fitted_target, 0U, y_fit, err);
        if (err.has_value())
        {
            return std::unexpected(std::move(*err));
        }

        std::string prec_str = "double";
        std::string res_str;
        switch (input.precision_kind)
        {
            case PersistPrecisionKind::Double:
                break;
            case PersistPrecisionKind::Single:
                prec_str = "single";
                break;
            case PersistPrecisionKind::MixedPromote:
                prec_str = "mixed";
                res_str = "promote_storage";
                break;
            case PersistPrecisionKind::MixedReevaluate:
                prec_str = "mixed";
                res_str = "reevaluate_kernel";
                break;
        }

        nlohmann::json slots = nlohmann::json::array();
        for (const auto& slot : kernel.slots)
        {
            if (slot.ard_dims.has_value())
            {
                slots.push_back({
                    {"kind",          "ard"},
                    {"dims", *slot.ard_dims}
                });
            }
            else
            {
                slots.push_back({
                    {"kind", "scalar"}
                });
            }
        }

        nlohmann::json root{
            {"format_version",                                     FORMAT_VERSION                              },
            {             "n",                                                                    input.n_train},
            {             "d",                                                                 input.n_features},
            {    "has_factor",                                                                 input.has_factor},
            {   "factor_kind",                                                                            "llt"},
            { "factor_jitter",                                                              input.factor_jitter},
            {     "precision",                                                                         prec_str},
            {          "math",           input.kernel_exp == KernelExp::FastApprox ? "fast_approx" : "accurate"},
            {"distance_cache",                                                                         "always"},
            {        "kernel",                                                                std::move(k_json)},
            {    "likelihood",
             {{"noise_variance", encode_bounded(input.likelihood.noise_variance(), input.likelihood.bounds())}}},
            {        "jitter",                                                      encode_jitter(input.jitter)},
            {    "x_unfitted",                                                                 std::move(x_unf)},
            {   "x_transform",                                                            std::move(x_fit_json)},
            {    "y_unfitted",                                                                 std::move(y_unf)},
            {   "y_transform",                                                                 std::move(y_fit)},
            {      "distance",
             {{"points",
             std::holds_alternative<const DistanceKernel<WithPoints>*>(input.kernel) ? "with_points"
             : "distance_only"},
             {"slots", std::move(slots)}}                                                                      },
        };
        if (!res_str.empty())
        {
            root["residual"] = res_str;
        }

        // Refuse to write a document that the loader would reject for its nesting depth: every composite kernel
        // node adds two JSON levels, so a tree accepted by the node-depth bound can still exceed the JSON bound.
        // The check runs before the recursive serializer descends into the document once per level.
        if (auto depth_ok = check_value_max_depth(root, MAX_JSON_NESTING_DEPTH); !depth_ok.has_value())
        {
            return std::unexpected(std::move(depth_ok).error());
        }
        return root.dump(2);
    }

    /// @brief Reads, validates and parses a coordinate checkpoint config.json.
    /// @param[in] path Filesystem path to config.json.
    /// @param[in] registry Registry supplying custom kernel and transform deserialization factories.
    /// @return Validated ParsedConfig on success, or GprError on I/O, JSON, version, or schema failure.
    /// @retval GprErrorCode::PersistFailed When file read, JSON depth/syntax, schema, model kind, or registry fails.
    /// @retval GprErrorCode::UnsupportedPersistVersion When format_version != FORMAT_VERSION.
    std::expected<ParsedConfig, GprError> parse_config_json(
        const std::filesystem::path& path, const PersistRegistry& registry
    )
    {
        return parse_config_impl(path, registry, false);
    }

    /// @brief Reads, validates and parses a supplied-distance checkpoint config.json.
    /// @param[in] path Filesystem path to config.json.
    /// @param[in] registry Registry supplying custom kernel and transform deserialization factories.
    /// @return Validated ParsedConfig on success, or GprError on I/O, JSON, version, or schema failure.
    /// @retval GprErrorCode::PersistFailed When file read, JSON depth/syntax, schema, model kind, or registry fails.
    /// @retval GprErrorCode::UnsupportedPersistVersion When format_version != FORMAT_VERSION.
    std::expected<ParsedConfig, GprError> parse_distance_config_json(
        const std::filesystem::path& path, const PersistRegistry& registry
    )
    {
        return parse_config_impl(path, registry, true);
    }

}  // namespace cxxgpr::persist::detail
