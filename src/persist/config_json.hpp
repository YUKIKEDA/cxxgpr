// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/likelihood.hpp"
#include "cxxgpr/kernel/distance_cache.hpp"
#include "cxxgpr/kernel/exp.hpp"
#include "cxxgpr/kernel/spec.hpp"
#include "cxxgpr/linalg/jitter.hpp"
#include "cxxgpr/persist/persist.hpp"
#include "cxxgpr/precision/policy.hpp"
#include "cxxgpr/transform/input.hpp"
#include "cxxgpr/transform/target.hpp"

#include <cstddef>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace cxxgpr::persist::detail {

struct ParsedConfig {
    std::size_t n{0};
    std::size_t d{0};
    bool has_factor{false};
    double factor_jitter{0.0};
    PersistPrecisionKind precision_kind{PersistPrecisionKind::Double};
    KernelExp kernel_exp{KernelExp::Accurate};
    DistanceCachePolicy distance_cache{DistanceCachePolicy::Cached};
    std::optional<KernelSpec> kernel_spec;
    std::optional<GaussianLikelihood> likelihood;
    JitterPolicy jitter;
    std::unique_ptr<transform::UnfittedInputTransform> x_unfitted;
    std::unique_ptr<transform::FittedInputTransform> x_transform;
    std::unique_ptr<transform::UnfittedTargetTransform> y_unfitted;
    std::unique_ptr<transform::FittedTargetTransform> y_transform;
};

[[nodiscard]] std::expected<std::string, GprError> serialize_config_json(const SaveInput& input);

[[nodiscard]] std::expected<ParsedConfig, GprError>
parse_config_json(const std::filesystem::path& path, const PersistRegistry& registry);

} // namespace cxxgpr::persist::detail
