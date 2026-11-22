// SPDX-License-Identifier: Apache-2.0
// Spec: QA-06, REQ-11, DES-03
/// @file build_info_test.cpp
/// @brief Unit tests for build provenance metadata and zero-allocation audit instrumentation.
/// @details Verifies that cxxgpr::build_info() and cxxgpr::build_info_json() match pinned dependency
/// versions from vcpkg.json and that NoAllocationScope accurately catches heap allocations and Eigen
/// runtime malloc state transitions.

#include "cxxgpr/core/build_info.hpp"

#include <Eigen/Core>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "allocation_audit.hpp"

namespace
{

    /// @brief Reads pinned dependency versions from the repository vcpkg.json overrides array.
    /// @details Parses the manifest at path and maps each override package name to its version string.
    /// @param[in] path Null-terminated filesystem path to the vcpkg.json manifest file.
    /// @return Map from vcpkg override package name to pinned version string, or empty if unreadable.
    std::map<std::string, std::string> read_manifest_overrides(const char* path)
    {
        std::ifstream stream(path);
        if (!stream.is_open())
        {
            return {};
        }
        const nlohmann::json manifest = nlohmann::json::parse(stream);
        std::map<std::string, std::string> result;
        // Extract version pins whether specified under "version" or "version-semver".
        for (const auto& entry : manifest.at("overrides"))
        {
            const auto name = entry.at("name").get<std::string>();
            if (entry.contains("version"))
            {
                result.emplace(name, entry.at("version").get<std::string>());
            }
            else if (entry.contains("version-semver"))
            {
                result.emplace(name, entry.at("version-semver").get<std::string>());
            }
        }
        return result;
    }

    /// @brief Verifies build_info() metadata completeness, vcpkg pin parity, and JSON schema consistency.
    /// @details Checks that provenance fields are non-empty, reported Eigen and nlohmann_json versions
    /// match vcpkg.json overrides, and build_info_json() roundtrips the same fields.
    TEST(BuildInfoTest, MatchesManifestPinsAndJsonSchema)
    {
        // Confirm structured provenance fields and schema tag are populated.
        const auto info = cxxgpr::build_info();
        EXPECT_EQ(info.schema, "cxxgpr.build-info/1");
        EXPECT_FALSE(info.version.empty());
        EXPECT_FALSE(info.git_commit.empty());
        EXPECT_FALSE(info.compiler.empty());
        EXPECT_FALSE(info.platform.empty());
        EXPECT_FALSE(info.platform_key.empty());
        EXPECT_FALSE(info.configuration.empty());
        EXPECT_FALSE(info.floating_point.empty());

        // Cross-check dependency versions against the repository vcpkg.json manifest pins.
        const auto overrides = read_manifest_overrides(CXXGPR_MANIFEST_PATH);
        ASSERT_FALSE(overrides.empty());

        std::map<std::string, std::string> actual_deps;
        for (const auto& dep : info.dependencies)
        {
            actual_deps.emplace(dep.name, dep.version);
        }
        ASSERT_TRUE(actual_deps.contains("eigen"));
        ASSERT_TRUE(actual_deps.contains("nlohmann_json"));
        EXPECT_EQ(actual_deps.at("eigen"), overrides.at("eigen3"));
        EXPECT_EQ(actual_deps.at("nlohmann_json"), overrides.at("nlohmann-json"));

        // Verify the JSON serialization preserves every structured provenance attribute.
        const std::string json_text = cxxgpr::build_info_json();
        const nlohmann::json parsed = nlohmann::json::parse(json_text);
        EXPECT_EQ(parsed.at("schema").get<std::string>(), info.schema);
        EXPECT_EQ(parsed.at("version").get<std::string>(), info.version);
        EXPECT_EQ(parsed.at("git_commit").get<std::string>(), info.git_commit);
        EXPECT_EQ(parsed.at("uncommitted_changes").get<bool>(), info.uncommitted_changes);
        EXPECT_EQ(parsed.at("platform_key").get<std::string>(), info.platform_key);
        EXPECT_EQ(parsed.at("configuration").get<std::string>(), info.configuration);
    }

    /// @brief Verifies NoAllocationScope permits stack-allocated Eigen arithmetic and restores malloc state.
    /// @details Confirms AuditMode::CppAndEigen disables Eigen runtime malloc inside the scope, observes
    /// zero tracked allocations for fixed-size matrix multiplication, and restores malloc on scope exit.
    TEST(AllocationAuditTest, FixedSizeEigenWorkAllocatesZero)
    {
        Eigen::Matrix2d a;
        a << 1.0, 0.2, 0.2, 1.5;
        Eigen::Matrix2d b;
        b << 2.0, -0.1, 0.3, 0.8;
        Eigen::Matrix2d c = Eigen::Matrix2d::Zero();

        // Eigen runtime malloc must start enabled and be disabled only within the audit scope.
        EXPECT_TRUE(Eigen::internal::is_malloc_allowed());
        {
            cxxgpr::test_support::NoAllocationScope scope(cxxgpr::test_support::AuditMode::CppAndEigen);
            EXPECT_FALSE(Eigen::internal::is_malloc_allowed());
            c.noalias() = a * b;
            // Pin c via volatile sink so dead-store elimination cannot elide the multiply.
            cxxgpr::test_support::touch_for_test(c.data());
            EXPECT_EQ(scope.delta_allocations(), 0U);
        }
        // Scope destructor must restore Eigen's prior runtime malloc state.
        EXPECT_TRUE(Eigen::internal::is_malloc_allowed());
        EXPECT_NEAR(c(0, 0), 2.06, 1e-15);
    }

    /// @brief Negative control verifying NoAllocationScope detects dynamic C++ heap allocations.
    /// @details Allocates a std::vector inside AuditMode::CppOnly and checks that delta_allocations() >= 1,
    /// and verifies AssertZeroOnExit emits a non-fatal failure on scope destruction even without explicit checks.
    TEST(AllocationAuditTest, DynamicVectorAllocationIsDetected)
    {
        {
            cxxgpr::test_support::NoAllocationScope scope(
                cxxgpr::test_support::AuditMode::CppOnly, cxxgpr::test_support::AuditExpectation::Manual
            );
            const std::vector<double> dynamic_buf(64, 1.25);
            // Prevent the compiler from eliding the heap-allocated vector before delta_allocations().
            cxxgpr::test_support::touch_for_test(dynamic_buf.data());
            EXPECT_GE(scope.delta_allocations(), 1U);
        }

        EXPECT_NONFATAL_FAILURE(
            {
                const cxxgpr::test_support::NoAllocationScope scope(cxxgpr::test_support::AuditMode::CppOnly);
                const std::vector<double> dynamic_buf(64, 1.25);
                cxxgpr::test_support::touch_for_test(dynamic_buf.data());
            },
            "Expected zero heap allocations inside NoAllocationScope"
        );
    }

}  // namespace
