// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/build_info.hpp"

#include "allocation_audit.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <map>
#include <string>
#include <vector>

// Spec: QA-06, REQ-11, DES-03
namespace {

std::map<std::string, std::string> read_manifest_overrides(const char* path) {
    std::ifstream stream(path);
    if (!stream.is_open()) {
        return {};
    }
    const nlohmann::json manifest = nlohmann::json::parse(stream);
    std::map<std::string, std::string> result;
    for (const auto& entry : manifest.at("overrides")) {
        const auto name = entry.at("name").get<std::string>();
        if (entry.contains("version")) {
            result.emplace(name, entry.at("version").get<std::string>());
        } else if (entry.contains("version-semver")) {
            result.emplace(name, entry.at("version-semver").get<std::string>());
        }
    }
    return result;
}

TEST(BuildInfoTest, MatchesManifestPinsAndJsonSchema) {
    const auto info = cxxgpr::build_info();
    EXPECT_EQ(info.schema, "cxxgpr.build-info/1");
    EXPECT_FALSE(info.version.empty());
    EXPECT_FALSE(info.git_commit.empty());
    EXPECT_FALSE(info.compiler.empty());
    EXPECT_FALSE(info.platform.empty());
    EXPECT_FALSE(info.platform_key.empty());
    EXPECT_FALSE(info.configuration.empty());
    EXPECT_FALSE(info.floating_point.empty());

    const auto overrides = read_manifest_overrides(CXXGPR_MANIFEST_PATH);
    ASSERT_FALSE(overrides.empty());

    std::map<std::string, std::string> actual_deps;
    for (const auto& dep : info.dependencies) {
        actual_deps.emplace(dep.name, dep.version);
    }
    ASSERT_TRUE(actual_deps.contains("eigen"));
    ASSERT_TRUE(actual_deps.contains("nlohmann_json"));
    EXPECT_EQ(actual_deps.at("eigen"), overrides.at("eigen3"));
    EXPECT_EQ(actual_deps.at("nlohmann_json"), overrides.at("nlohmann-json"));

    const std::string json_text = cxxgpr::build_info_json();
    const nlohmann::json parsed = nlohmann::json::parse(json_text);
    EXPECT_EQ(parsed.at("schema").get<std::string>(), info.schema);
    EXPECT_EQ(parsed.at("version").get<std::string>(), info.version);
    EXPECT_EQ(parsed.at("git_commit").get<std::string>(), info.git_commit);
    EXPECT_EQ(parsed.at("uncommitted_changes").get<bool>(), info.uncommitted_changes);
    EXPECT_EQ(parsed.at("platform_key").get<std::string>(), info.platform_key);
    EXPECT_EQ(parsed.at("configuration").get<std::string>(), info.configuration);
}

TEST(AllocationAuditTest, FixedSizeEigenWorkAllocatesZero) {
    Eigen::Matrix2d a;
    a << 1.0, 0.2, 0.2, 1.5;
    Eigen::Matrix2d b;
    b << 2.0, -0.1, 0.3, 0.8;
    Eigen::Matrix2d c = Eigen::Matrix2d::Zero();

    cxxgpr::test_support::NoAllocationScope scope(cxxgpr::test_support::AuditMode::CppAndEigen);
    c.noalias() = a * b;
    cxxgpr::test_support::touch_for_test(c.data());
    EXPECT_EQ(scope.delta_allocations(), 0U);
    EXPECT_NEAR(c(0, 0), 2.06, 1e-15);
}

TEST(AllocationAuditTest, DynamicVectorAllocationIsDetected) {
    cxxgpr::test_support::NoAllocationScope scope(cxxgpr::test_support::AuditMode::CppOnly);
    const std::vector<double> dynamic_buf(64, 1.25);
    cxxgpr::test_support::touch_for_test(dynamic_buf.data());
    EXPECT_GE(scope.delta_allocations(), 1U);
}

} // namespace
