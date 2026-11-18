// SPDX-License-Identifier: Apache-2.0
/// @file build_info.hpp
/// @brief Structured build provenance and dependency metadata types and accessors.
/// @details Exposes toolchain, git revision, floating-point configuration, and linked dependency
/// versions for reproducibility tracking and baseline verification.

#pragma once

#include <string>
#include <vector>

/// @brief Top-level namespace for the cxxgpr Gaussian process regression library.
namespace cxxgpr
{

    /// @brief Name and semantic version of a linked third-party dependency.
    /// @details Populated at compile time from dependency header or package version macros.
    struct DependencyVersion
    {
        /// @brief Canonical package identifier of the linked dependency.
        std::string name;
        /// @brief Semantic version string formatted as major.minor.patch.
        std::string version;
    };

    /// @brief Build, toolchain, and dependency provenance for reproducibility and baseline checks.
    /// @details Captures schema identifier, library version, git state, compiler and platform keys,
    /// build configuration, floating-point mode, and linked third-party dependency versions.
    struct BuildInfo
    {
        /// @brief Provenance payload schema version identifier (for example, "cxxgpr.build-info/1").
        std::string schema;
        /// @brief Compiled cxxgpr library semantic version string.
        std::string version;
        /// @brief Git commit hash recorded at configure/build time.
        std::string git_commit;
        /// @brief True if the working tree had uncommitted changes when built.
        bool uncommitted_changes = false;
        /// @brief Human-readable compiler identification string.
        std::string compiler;
        /// @brief Target operating system and architecture description.
        std::string platform;
        /// @brief Normalized os-arch-compiler triplet used to key baseline artifacts.
        std::string platform_key;
        /// @brief CMake build configuration name (such as Debug or Release).
        std::string configuration;
        /// @brief Active floating-point precision and contraction configuration description.
        std::string floating_point;
        /// @brief Ordered list of linked third-party dependency versions.
        std::vector<DependencyVersion> dependencies;
    };

    /// @brief Returns structured build provenance for the compiled cxxgpr library.
    /// @details Assembles compile-time version, git status, platform, and dependency macros into a
    /// BuildInfo snapshot.
    /// @return Populated BuildInfo instance describing the current library build.
    [[nodiscard]] BuildInfo build_info();

    /// @brief Returns structured build provenance serialized as formatted JSON.
    /// @details Converts the output of build_info() into a two-space-indented JSON object with a
    /// trailing newline.
    /// @return Pretty-printed JSON representation of build_info().
    [[nodiscard]] std::string build_info_json();

}  // namespace cxxgpr
