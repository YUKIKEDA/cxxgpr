// SPDX-License-Identifier: Apache-2.0
/// @file build_info.cpp
/// @brief Implementation of build provenance collection and JSON serialization.
/// @details Materializes compile-time build configuration macros and dependency version headers
/// into BuildInfo structs and formatted JSON strings.

#include "cxxgpr/core/build_info.hpp"

#include "cxxgpr/build_config.hpp"
#include "cxxgpr/git_revision.hpp"

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include <format>
#include <string>

namespace cxxgpr
{

    /// @brief Constructs a BuildInfo snapshot from generated build and dependency macros.
    /// @details Populates git metadata, normalized platform key, build configuration, and
    /// Eigen/nlohmann_json versions.
    /// @return Populated BuildInfo value.
    BuildInfo build_info()
    {
        // Stamp the schema version and compile-time library and git provenance fields.
        BuildInfo info;
        info.schema = "cxxgpr.build-info/1";
        info.version = CXXGPR_VERSION;
        info.git_commit = CXXGPR_GIT_COMMIT;
        info.uncommitted_changes = (CXXGPR_GIT_DIRTY != 0);
        info.compiler = CXXGPR_COMPILER;
        info.platform = CXXGPR_PLATFORM;
        // Combine OS, architecture, and compiler keys into the baseline lookup triplet.
        info.platform_key = std::format("{}-{}-{}", CXXGPR_OS_KEY, CXXGPR_ARCH_KEY, CXXGPR_COMPILER_KEY);
        info.configuration = CXXGPR_BUILD_CONFIGURATION;
        info.floating_point = CXXGPR_FLOATING_POINT;
        // Record exact major.minor.patch versions of linked dependencies.
        info.dependencies = {
            DependencyVersion{
                              .name = "eigen",
                              .version = std::format("{}.{}.{}",         EIGEN_MAJOR_VERSION,         EIGEN_MINOR_VERSION, EIGEN_PATCH_VERSION),
                              },
            DependencyVersion{
                              .name = "nlohmann_json",
                              .version = std::format(
                              "{}.{}.{}", NLOHMANN_JSON_VERSION_MAJOR, NLOHMANN_JSON_VERSION_MINOR,     NLOHMANN_JSON_VERSION_PATCH
     ), },
        };
        return info;
    }

    /// @brief Serializes the current build provenance as an indented JSON string.
    /// @details Maps BuildInfo fields and dependency entries into a JSON object terminated by a
    /// single newline.
    /// @return Two-space-indented JSON string representing build_info().
    std::string build_info_json()
    {
        const BuildInfo info = build_info();
        // Serialize each dependency entry into an explicit name/version JSON object.
        nlohmann::json deps = nlohmann::json::array();
        for (const auto& dep : info.dependencies)
        {
            deps.push_back({
                {   "name",    dep.name},
                {"version", dep.version}
            });
        }
        // Assemble top-level provenance keys and emit two-space-indented JSON with trailing LF.
        const nlohmann::json doc = {
            {             "schema",              info.schema},
            {            "version",             info.version},
            {         "git_commit",          info.git_commit},
            {"uncommitted_changes", info.uncommitted_changes},
            {           "compiler",            info.compiler},
            {           "platform",            info.platform},
            {       "platform_key",        info.platform_key},
            {      "configuration",       info.configuration},
            {     "floating_point",      info.floating_point},
            {       "dependencies",                     deps},
        };
        return doc.dump(2) + '\n';
    }

}  // namespace cxxgpr
