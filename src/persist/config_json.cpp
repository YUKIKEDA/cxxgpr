// SPDX-License-Identifier: Apache-2.0
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

    struct KernelSpecPersistAccess
    {
        [[nodiscard]] static const KernelSpec::NodeVariant& node(const KernelSpec& spec) noexcept { return spec.node_; }

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
    namespace
    {

        constexpr std::size_t k_max_nesting_depth = 128U;

        [[nodiscard]] GprError config_err(std::string reason)
        {
            return GprError::persist_failed(PersistErrorKind::Config, std::move(reason));
        }

        [[nodiscard]] nlohmann::json encode_bounded(double val, const Interval& iv)
        {
            return nlohmann::json{
                {"value",     val},
                {   "lo", iv.lo()},
                {   "hi", iv.hi()}
            };
        }

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
                    nlohmann::json parsed_state;
                    try
                    {
                        parsed_state = nlohmann::json::parse(*st);
                    }
                    catch (const nlohmann::json::exception&)
                    {
                        parsed_state = *st;
                    }
                    return nlohmann::json{
                        {"custom", {{"persist_id", std::string(pid)}, {"state", std::move(parsed_state)}}}
                    };
                }
            }, leaf
            );
        }

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
            nlohmann::json parsed;
            try
            {
                parsed = nlohmann::json::parse(*st);
            }
            catch (const nlohmann::json::exception&)
            {
                parsed = *st;
            }
            return nlohmann::json{
                {"custom", {{"persist_id", std::string(pid)}, {"state", std::move(parsed)}}}
            };
        }

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

        return root.dump(2);
    }

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

        const std::string prec = root.value("precision", std::string("double"));
        const std::string res = root.value("residual", std::string("promote_storage"));
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

        const std::string math_s = root.value("math", std::string("accurate"));
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

        const std::string dc_s = root.value("distance_cache", std::string("always"));
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
