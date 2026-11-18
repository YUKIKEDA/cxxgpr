// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>

namespace cxxgpr
{

    /// Name and semantic version of a linked third-party dependency.
    struct DependencyVersion
    {
        std::string name;
        std::string version;
    };

    /// Build, toolchain, and dependency provenance for reproducibility and baseline checks.
    struct BuildInfo
    {
        std::string schema;
        std::string version;
        std::string git_commit;
        bool uncommitted_changes = false;
        std::string compiler;
        std::string platform;
        std::string platform_key;
        std::string configuration;
        std::string floating_point;
        std::vector<DependencyVersion> dependencies;
    };

    /// Returns structured build provenance for the compiled cxxgpr library.
    [[nodiscard]] BuildInfo build_info();

    /// Returns structured build provenance serialized as formatted JSON.
    [[nodiscard]] std::string build_info_json();

}  // namespace cxxgpr
