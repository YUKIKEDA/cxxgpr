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
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "json_depth.hpp"

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

        /// @brief Recursively encodes a KernelSpec tree into JSON while guarding against excessive depth.
        /// @param[in] spec Kernel specification tree node to encode.
        /// @param[in] depth Current AST recursion depth bounded by k_max_nesting_depth.
        /// @return JSON representation of spec, or GprError if depth or leaf serialization fails.
        /// @retval GprErrorCode::PersistFailed When nesting depth exceeds 128 or a leaf fails to serialize.
        [[nodiscard]] std::expected<nlohmann::json, GprError> encode_kernel(const KernelSpec& spec, std::size_t depth)
        {
            if (depth > k_max_nesting_depth)
            {
                return std::unexpected(config_err("kernel nesting depth exceeds 128"));
            }
            const auto& node = KernelSpecPersistAccess::node(spec);
            if (const auto* leaf = std::get_if<KernelLeaf>(&node))
            {
                return encode_leaf(*leaf);
            }
            // Encode binary SumNode and ProductNode subtrees recursively with depth + 1.
            const auto [s_lhs, s_rhs] = KernelSpecPersistAccess::as_sum(node);
            if (s_lhs != nullptr)
            {
                auto l = encode_kernel(*s_lhs, depth + 1U);
                if (!l.has_value())
                {
                    return l;
                }
                auto r = encode_kernel(*s_rhs, depth + 1U);
                if (!r.has_value())
                {
                    return r;
                }
                return nlohmann::json{
                    {"sum", {{"left", std::move(*l)}, {"right", std::move(*r)}}}
                };
            }
            const auto [p_lhs, p_rhs] = KernelSpecPersistAccess::as_product(node);
            auto l = encode_kernel(*p_lhs, depth + 1U);
            if (!l.has_value())
            {
                return l;
            }
            auto r = encode_kernel(*p_rhs, depth + 1U);
            if (!r.has_value())
            {
                return r;
            }
            return nlohmann::json{
                {"product", {{"left", std::move(*l)}, {"right", std::move(*r)}}}
            };
        }

        /// @brief Recursively decodes a kernel JSON node into a KernelSpec tree.
        /// @param[in] j Single-key JSON object representing a leaf, sum, or product kernel node.
        /// @param[in] registry Extension registry used to instantiate "custom" leaves.
        /// @param[in] depth Current AST recursion depth bounded by k_max_nesting_depth.
        /// @return Reconstructed KernelSpec on success, or GprError on schema, interval, or registry failure.
        /// @retval GprErrorCode::PersistFailed When nesting exceeds 128, schema is invalid, or custom ID is missing.
        [[nodiscard]] std::expected<KernelSpec, GprError> decode_kernel(
            const nlohmann::json& j, const PersistRegistry& registry, std::size_t depth
        )
        {
            if (depth > k_max_nesting_depth)
            {
                return std::unexpected(config_err("kernel nesting depth exceeds 128"));
            }
            if (!j.is_object() || j.size() != 1U)
            {
                return std::unexpected(config_err("kernel node must be a single-key JSON object"));
            }
            const auto it = j.begin();
            const std::string& key = it.key();
            const nlohmann::json& body = it.value();

            // Reconstruct binary sum and product compositions before leaf variants.
            if (key == "sum" || key == "product")
            {
                if (!body.is_object() || !body.contains("left") || !body.contains("right"))
                {
                    return std::unexpected(config_err("composite kernel node missing left/right"));
                }
                auto lhs = decode_kernel(body["left"], registry, depth + 1U);
                if (!lhs.has_value())
                {
                    return lhs;
                }
                auto rhs = decode_kernel(body["right"], registry, depth + 1U);
                if (!rhs.has_value())
                {
                    return rhs;
                }
                return (key == "sum") ? (*lhs + *rhs) : (*lhs * *rhs);
            }

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

        /// @brief Recursively encodes an UnfittedInputTransform hierarchy into config.json format.
        /// @param[in] t Unfitted input transform node to serialize.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @return JSON representation of t, or GprError on excessive depth or non-persistable custom step.
        [[nodiscard]] std::expected<nlohmann::json, GprError> encode_unfitted_input(
            const transform::UnfittedInputTransform& t, std::size_t depth
        )
        {
            if (depth > k_max_nesting_depth)
            {
                return std::unexpected(config_err("transform nesting depth exceeds 128"));
            }
            if (dynamic_cast<const transform::IdentityInput*>(&t) != nullptr)
            {
                return nlohmann::json("identity");
            }
            if (dynamic_cast<const transform::StandardizeInput*>(&t) != nullptr)
            {
                return nlohmann::json{
                    {"standardize", nlohmann::json::object()}
                };
            }
            if (const auto* mm = dynamic_cast<const transform::MinMaxInput*>(&t))
            {
                const auto [lo, hi] = mm->feature_range();
                return nlohmann::json{
                    {"min_max", {{"range_lo", lo}, {"range_hi", hi}}}
                };
            }
            if (const auto* pipe = dynamic_cast<const transform::Pipeline*>(&t))
            {
                nlohmann::json steps = nlohmann::json::array();
                for (const auto& s : pipe->steps())
                {
                    auto enc = encode_unfitted_input(*s, depth + 1U);
                    if (!enc.has_value())
                    {
                        return enc;
                    }
                    steps.push_back(std::move(*enc));
                }
                return nlohmann::json{
                    {"pipeline", {{"steps", std::move(steps)}}}
                };
            }
            if (const auto* col = dynamic_cast<const transform::ColumnwiseInput*>(&t))
            {
                nlohmann::json maps = nlohmann::json::array();
                for (const auto& m : col->maps())
                {
                    auto enc = encode_unfitted_input(*m, depth + 1U);
                    if (!enc.has_value())
                    {
                        return enc;
                    }
                    maps.push_back(std::move(*enc));
                }
                return nlohmann::json{
                    {"columnwise", {{"maps", std::move(maps)}}}
                };
            }
            return encode_custom_transform(t);
        }

        /// @brief Recursively encodes a FittedInputTransform hierarchy and its fitted statistics into JSON.
        /// @param[in] t Fitted input transform node to serialize.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @return JSON representation of t, or GprError on excessive depth or non-persistable custom step.
        [[nodiscard]] std::expected<nlohmann::json, GprError> encode_fitted_input(
            const transform::FittedInputTransform& t, std::size_t depth
        )
        {
            if (depth > k_max_nesting_depth)
            {
                return std::unexpected(config_err("transform nesting depth exceeds 128"));
            }
            if (dynamic_cast<const transform::FittedIdentityInput*>(&t) != nullptr)
            {
                return nlohmann::json("identity");
            }
            if (const auto* st = dynamic_cast<const transform::FittedStandardizeInput*>(&t))
            {
                return nlohmann::json{
                    {"standardize", {{"mean", st->mean()}, {"std", st->std()}}}
                };
            }
            if (const auto* mm = dynamic_cast<const transform::FittedMinMaxInput*>(&t))
            {
                const auto [lo, hi] = mm->feature_range();
                return nlohmann::json{
                    {"min_max", {{"data_min", mm->min()}, {"data_max", mm->max()}, {"range_lo", lo}, {"range_hi", hi}}}
                };
            }
            if (const auto* pipe = dynamic_cast<const transform::FittedPipeline*>(&t))
            {
                nlohmann::json steps = nlohmann::json::array();
                for (const auto& s : pipe->steps())
                {
                    auto enc = encode_fitted_input(*s, depth + 1U);
                    if (!enc.has_value())
                    {
                        return enc;
                    }
                    steps.push_back(std::move(*enc));
                }
                return nlohmann::json{
                    {"pipeline", {{"steps", std::move(steps)}}}
                };
            }
            if (const auto* col = dynamic_cast<const transform::FittedColumnwiseInput*>(&t))
            {
                nlohmann::json maps = nlohmann::json::array();
                for (const auto& m : col->maps())
                {
                    auto enc = encode_fitted_input(*m, depth + 1U);
                    if (!enc.has_value())
                    {
                        return enc;
                    }
                    maps.push_back(std::move(*enc));
                }
                return nlohmann::json{
                    {"columnwise", {{"maps", std::move(maps)}}}
                };
            }
            return encode_custom_transform(t);
        }

        /// @brief Recursively encodes an UnfittedTargetTransform hierarchy into config.json format.
        /// @param[in] t Unfitted target transform node to serialize.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @return JSON representation of t, or GprError on excessive depth or non-persistable custom step.
        [[nodiscard]] std::expected<nlohmann::json, GprError> encode_unfitted_target(
            const transform::UnfittedTargetTransform& t, std::size_t depth
        )
        {
            if (depth > k_max_nesting_depth)
            {
                return std::unexpected(config_err("transform nesting depth exceeds 128"));
            }
            if (dynamic_cast<const transform::IdentityTarget*>(&t) != nullptr)
            {
                return nlohmann::json("identity");
            }
            if (dynamic_cast<const transform::StandardizeTarget*>(&t) != nullptr)
            {
                return nlohmann::json{
                    {"standardize", nlohmann::json::object()}
                };
            }
            if (const auto* mm = dynamic_cast<const transform::MinMaxTarget*>(&t))
            {
                const auto [lo, hi] = mm->feature_range();
                return nlohmann::json{
                    {"min_max", {{"range_lo", lo}, {"range_hi", hi}}}
                };
            }
            if (const auto* pipe = dynamic_cast<const transform::TargetPipeline*>(&t))
            {
                nlohmann::json steps = nlohmann::json::array();
                for (const auto& s : pipe->steps())
                {
                    auto enc = encode_unfitted_target(*s, depth + 1U);
                    if (!enc.has_value())
                    {
                        return enc;
                    }
                    steps.push_back(std::move(*enc));
                }
                return nlohmann::json{
                    {"pipeline", {{"steps", std::move(steps)}}}
                };
            }
            return encode_custom_transform(t);
        }

        /// @brief Recursively encodes a FittedTargetTransform hierarchy and its fitted statistics into JSON.
        /// @param[in] t Fitted target transform node to serialize.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @return JSON representation of t, or GprError on excessive depth or non-persistable custom step.
        [[nodiscard]] std::expected<nlohmann::json, GprError> encode_fitted_target(
            const transform::FittedTargetTransform& t, std::size_t depth
        )
        {
            if (depth > k_max_nesting_depth)
            {
                return std::unexpected(config_err("transform nesting depth exceeds 128"));
            }
            if (dynamic_cast<const transform::FittedIdentityTarget*>(&t) != nullptr)
            {
                return nlohmann::json("identity");
            }
            if (const auto* st = dynamic_cast<const transform::FittedStandardizeTarget*>(&t))
            {
                return nlohmann::json{
                    {"standardize", {{"mean", st->mean()}, {"std", st->std()}}}
                };
            }
            if (const auto* mm = dynamic_cast<const transform::FittedMinMaxTarget*>(&t))
            {
                const auto [lo, hi] = mm->feature_range();
                return nlohmann::json{
                    {"min_max", {{"data_min", mm->min()}, {"data_max", mm->max()}, {"range_lo", lo}, {"range_hi", hi}}}
                };
            }
            if (const auto* pipe = dynamic_cast<const transform::FittedTargetPipeline*>(&t))
            {
                nlohmann::json steps = nlohmann::json::array();
                for (const auto& s : pipe->steps())
                {
                    auto enc = encode_fitted_target(*s, depth + 1U);
                    if (!enc.has_value())
                    {
                        return enc;
                    }
                    steps.push_back(std::move(*enc));
                }
                return nlohmann::json{
                    {"pipeline", {{"steps", std::move(steps)}}}
                };
            }
            return encode_custom_transform(t);
        }

        /// @brief Recursively decodes an UnfittedInputTransform hierarchy from JSON.
        /// @param[in] j JSON string ("identity") or single-key transform descriptor object.
        /// @param[in] registry Registry supplying factories for "custom" transforms.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @return Owned UnfittedInputTransform instance, or GprError on schema, range, or registry error.
        [[nodiscard]] std::expected<std::unique_ptr<transform::UnfittedInputTransform>, GprError> decode_unfitted_input(
            const nlohmann::json& j, const PersistRegistry& registry, std::size_t depth
        )
        {
            if (depth > k_max_nesting_depth)
            {
                return std::unexpected(config_err("transform nesting depth exceeds 128"));
            }
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
                // Decode Pipeline ("steps") and ColumnwiseInput ("maps") uniformly with depth + 1.
                const std::string_view arr_key = (key == "pipeline") ? "steps" : "maps";
                if (!body.is_object() || !body.contains(arr_key) || !body[arr_key].is_array())
                {
                    return std::unexpected(config_err("composite transform missing array"));
                }
                std::vector<std::unique_ptr<transform::UnfittedInputTransform>> items;
                for (const auto& elem : body[arr_key])
                {
                    auto sub = decode_unfitted_input(elem, registry, depth + 1U);
                    if (!sub.has_value())
                    {
                        return std::unexpected(std::move(sub).error());
                    }
                    items.push_back(std::move(*sub));
                }
                if (key == "pipeline")
                {
                    return std::make_unique<transform::Pipeline>(std::move(items));
                }
                return std::make_unique<transform::ColumnwiseInput>(std::move(items));
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

        /// @brief Recursively decodes a FittedInputTransform hierarchy and validates restored statistics.
        /// @param[in] j JSON string ("identity") or single-key fitted transform descriptor object.
        /// @param[in] registry Registry supplying factories for "custom" transforms.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @return Owned FittedInputTransform instance, or GprError on schema, parameter, or registry error.
        [[nodiscard]] std::expected<std::unique_ptr<transform::FittedInputTransform>, GprError> decode_fitted_input(
            const nlohmann::json& j, const PersistRegistry& registry, std::size_t depth
        )
        {
            if (depth > k_max_nesting_depth)
            {
                return std::unexpected(config_err("transform nesting depth exceeds 128"));
            }
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
                std::vector<std::unique_ptr<transform::FittedInputTransform>> items;
                for (const auto& elem : body[arr_key])
                {
                    auto sub = decode_fitted_input(elem, registry, depth + 1U);
                    if (!sub.has_value())
                    {
                        return std::unexpected(std::move(sub).error());
                    }
                    items.push_back(std::move(*sub));
                }
                if (key == "pipeline")
                {
                    return std::make_unique<transform::FittedPipeline>(std::move(items));
                }
                return std::make_unique<transform::FittedColumnwiseInput>(std::move(items));
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

        /// @brief Recursively decodes an UnfittedTargetTransform hierarchy from JSON.
        /// @param[in] j JSON string ("identity") or single-key target transform descriptor object.
        /// @param[in] registry Registry supplying factories for "custom" target transforms.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @return Owned UnfittedTargetTransform instance, or GprError on schema, range, or registry error.
        [[nodiscard]] std::expected<std::unique_ptr<transform::UnfittedTargetTransform>, GprError>
            decode_unfitted_target(const nlohmann::json& j, const PersistRegistry& registry, std::size_t depth)
        {
            if (depth > k_max_nesting_depth)
            {
                return std::unexpected(config_err("transform nesting depth exceeds 128"));
            }
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
                std::vector<std::unique_ptr<transform::UnfittedTargetTransform>> steps;
                for (const auto& elem : body["steps"])
                {
                    auto sub = decode_unfitted_target(elem, registry, depth + 1U);
                    if (!sub.has_value())
                    {
                        return std::unexpected(std::move(sub).error());
                    }
                    steps.push_back(std::move(*sub));
                }
                return std::make_unique<transform::TargetPipeline>(std::move(steps));
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

        /// @brief Recursively decodes a FittedTargetTransform hierarchy and validates restored statistics.
        /// @param[in] j JSON string ("identity") or single-key fitted target transform descriptor object.
        /// @param[in] registry Registry supplying factories for "custom" target transforms.
        /// @param[in] depth Current recursion depth bounded by k_max_nesting_depth.
        /// @return Owned FittedTargetTransform instance, or GprError on schema, parameter, or registry error.
        [[nodiscard]] std::expected<std::unique_ptr<transform::FittedTargetTransform>, GprError> decode_fitted_target(
            const nlohmann::json& j, const PersistRegistry& registry, std::size_t depth
        )
        {
            if (depth > k_max_nesting_depth)
            {
                return std::unexpected(config_err("transform nesting depth exceeds 128"));
            }
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
                std::vector<std::unique_ptr<transform::FittedTargetTransform>> steps;
                for (const auto& elem : body["steps"])
                {
                    auto sub = decode_fitted_target(elem, registry, depth + 1U);
                    if (!sub.has_value())
                    {
                        return std::unexpected(std::move(sub).error());
                    }
                    steps.push_back(std::move(*sub));
                }
                return std::make_unique<transform::FittedTargetPipeline>(std::move(steps));
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

    }  // namespace

    /// @brief Serializes a SaveInput bundle into pretty-printed config.json text.
    /// @details Encodes kernel AST, likelihood with bounds, jitter policy, unfitted/fitted transforms,
    /// precision/residual flags, exponential math mode, and distance cache policy.
    /// @param[in] input Borrowed model configuration and transform references to serialize.
    /// @return Formatted JSON text on success, or GprError if any subtree or custom component fails encoding.
    /// @retval GprErrorCode::PersistFailed When custom kernels/transforms are non-persistable or depth exceeds 128.
    std::expected<std::string, GprError> serialize_config_json(const SaveInput& input)
    {
        auto k_json = encode_kernel(input.kernel_spec, 0U);
        if (!k_json.has_value())
        {
            return std::unexpected(std::move(k_json).error());
        }
        auto x_unf = encode_unfitted_input(input.unfitted_input, 0U);
        if (!x_unf.has_value())
        {
            return std::unexpected(std::move(x_unf).error());
        }
        auto x_fit = encode_fitted_input(input.fitted_input, 0U);
        if (!x_fit.has_value())
        {
            return std::unexpected(std::move(x_fit).error());
        }
        auto y_unf = encode_unfitted_target(input.unfitted_target, 0U);
        if (!y_unf.has_value())
        {
            return std::unexpected(std::move(y_unf).error());
        }
        auto y_fit = encode_fitted_target(input.fitted_target, 0U);
        if (!y_fit.has_value())
        {
            return std::unexpected(std::move(y_fit).error());
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
            {        "kernel",                                                               std::move(*k_json)},
            {    "likelihood",
             {{"noise_variance", encode_bounded(input.likelihood.noise_variance(), input.likelihood.bounds())}}},
            {        "jitter",                                                      encode_jitter(input.jitter)},
            {    "x_unfitted",                                                                std::move(*x_unf)},
            {   "x_transform",                                                                std::move(*x_fit)},
            {    "y_unfitted",                                                                std::move(*y_unf)},
            {   "y_transform",                                                                std::move(*y_fit)},
        };
        if (input.precision_kind == PersistPrecisionKind::MixedPromote
            || input.precision_kind == PersistPrecisionKind::MixedReevaluate)
        {
            root["residual"] = res_str;
        }

        // Refuse to write a document that the loader would reject for its nesting depth.
        std::string text = root.dump(2);
        if (auto depth_ok = check_json_max_depth(text, MAX_JSON_NESTING_DEPTH, PersistErrorKind::Config);
            !depth_ok.has_value())
        {
            return std::unexpected(std::move(depth_ok).error());
        }
        return text;
    }

    /// @brief Reads, depth-checks, parses, and validates config.json into a ParsedConfig struct.
    /// @details Enforces MAX_JSON_NESTING_DEPTH prior to JSON parsing, checks model/distance checkpoint
    /// compatibility, validates format_version and factor_kind == "llt", and deserializes kernels,
    /// likelihood, jitter, and transforms.
    /// @param[in] path Filesystem path to config.json.
    /// @param[in] registry Registry supplying custom kernel and transform deserialization factories.
    /// @return Validated ParsedConfig on success, or GprError on I/O, JSON, version, or schema failure.
    /// @retval GprErrorCode::PersistFailed When file read, JSON depth/syntax, schema, model kind, or registry fails.
    /// @retval GprErrorCode::UnsupportedPersistVersion When format_version != FORMAT_VERSION.
    std::expected<ParsedConfig, GprError> parse_config_json(
        const std::filesystem::path& path, const PersistRegistry& registry
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
        if (root.contains("distance"))
        {
            return std::unexpected(
                GprError::persist_failed(
                    PersistErrorKind::WrongModel, "distance-supplied model requires LoadedDistanceGpr"
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
        cfg.n = root["n"].get<std::size_t>();
        cfg.d = root["d"].get<std::size_t>();
        if (cfg.n == 0U || cfg.d == 0U)
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
            auto k_spec = decode_kernel(root["kernel"], registry, 0U);
            if (!k_spec.has_value())
            {
                return std::unexpected(std::move(k_spec).error());
            }
            cfg.kernel_spec = std::move(*k_spec);

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
            auto lik = GaussianLikelihood::create(nv->first);
            if (!lik.has_value())
            {
                return std::unexpected(std::move(lik).error());
            }
            auto lik_wb = lik->with_bounds(nv->second);
            if (!lik_wb.has_value())
            {
                return std::unexpected(std::move(lik_wb).error());
            }
            cfg.likelihood = *lik_wb;

            auto jit = decode_jitter(root["jitter"]);
            if (!jit.has_value())
            {
                return std::unexpected(std::move(jit).error());
            }
            cfg.jitter = *jit;

            auto xu = decode_unfitted_input(root["x_unfitted"], registry, 0U);
            if (!xu.has_value())
            {
                return std::unexpected(std::move(xu).error());
            }
            cfg.x_unfitted = std::move(*xu);

            auto xf = decode_fitted_input(root["x_transform"], registry, 0U);
            if (!xf.has_value())
            {
                return std::unexpected(std::move(xf).error());
            }
            cfg.x_transform = std::move(*xf);

            auto yu = decode_unfitted_target(root["y_unfitted"], registry, 0U);
            if (!yu.has_value())
            {
                return std::unexpected(std::move(yu).error());
            }
            cfg.y_unfitted = std::move(*yu);

            auto yf = decode_fitted_target(root["y_transform"], registry, 0U);
            if (!yf.has_value())
            {
                return std::unexpected(std::move(yf).error());
            }
            cfg.y_transform = std::move(*yf);
        }
        catch (const nlohmann::json::exception&)
        {
            return std::unexpected(config_err("invalid field types in config.json"));
        }

        return cfg;
    }

}  // namespace cxxgpr::persist::detail
