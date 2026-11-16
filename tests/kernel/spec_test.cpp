// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-01, REQ-02, REQ-04, DES-06

#include "cxxgpr/kernel/compiled.hpp"
#include "cxxgpr/kernel/spec.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>
#include <vector>

namespace {

TEST(KernelSpecTest, AssignsDepthFirstBindingsAndClassifiesCoordModes) {
    const auto c = cxxgpr::ConstantKernel::create(2.0);
    const auto rbf = cxxgpr::RbfKernel::create(1.5);
    const auto white = cxxgpr::WhiteKernel::create(0.01);
    const auto linear = cxxgpr::LinearKernel::create(0.5);
    ASSERT_TRUE(c.has_value());
    ASSERT_TRUE(rbf.has_value());
    ASSERT_TRUE(white.has_value());
    ASSERT_TRUE(linear.has_value());

    EXPECT_EQ(cxxgpr::KernelSpec(*c).coord_mode(), cxxgpr::CoordMode::Either);
    EXPECT_EQ((cxxgpr::KernelSpec(*c) * cxxgpr::KernelSpec(*rbf)).coord_mode(),
              cxxgpr::CoordMode::Dist);
    EXPECT_EQ((cxxgpr::KernelSpec(*c) + cxxgpr::KernelSpec(*linear)).coord_mode(),
              cxxgpr::CoordMode::Points);

    cxxgpr::KernelSpec tree = (cxxgpr::KernelSpec(*c) * cxxgpr::KernelSpec(*rbf)) +
                              (cxxgpr::KernelSpec(*white) + cxxgpr::KernelSpec(*linear));
    EXPECT_EQ(tree.coord_mode(), cxxgpr::CoordMode::Mixed);
    EXPECT_EQ(tree.num_params(), 4U);

    const auto bindings = tree.parameter_bindings();
    ASSERT_EQ(bindings.size(), 4U);
    EXPECT_EQ(bindings[0], (cxxgpr::ParameterBinding{.index = 0, .leaf_id = 0, .local_index = 0}));
    EXPECT_EQ(bindings[1], (cxxgpr::ParameterBinding{.index = 1, .leaf_id = 1, .local_index = 0}));
    EXPECT_EQ(bindings[2], (cxxgpr::ParameterBinding{.index = 2, .leaf_id = 2, .local_index = 0}));
    EXPECT_EQ(bindings[3], (cxxgpr::ParameterBinding{.index = 3, .leaf_id = 3, .local_index = 0}));

    const auto compiled = tree.compile<double>();
    EXPECT_EQ(compiled.num_params(), 4U);
    EXPECT_EQ(compiled.num_leaves(), 4U);
    EXPECT_EQ(compiled.coord_mode(), cxxgpr::CoordMode::Mixed);
    EXPECT_EQ(compiled.parameter_bindings(), bindings);
    EXPECT_GE(compiled.nested_depth(), 1U);
}

TEST(KernelSpecTest, SetParamsIsAtomicAcrossSpecAndCompiled) {
    const auto rbf = cxxgpr::RbfKernel::create(1.0);
    const auto per = cxxgpr::PeriodicKernel::create(2.0, 3.0);
    ASSERT_TRUE(rbf.has_value());
    ASSERT_TRUE(per.has_value());

    cxxgpr::KernelSpec spec = cxxgpr::KernelSpec(*rbf) + cxxgpr::KernelSpec(*per);
    auto compiled = spec.compile<double>();

    std::array<double, 3> orig{};
    ASSERT_TRUE(spec.get_params(orig).has_value());

    // Fail on the last parameter; earlier leaves must stay unchanged.
    const std::array<double, 3> bad{std::log(5.0), std::log(6.0), std::log(1e6)};
    EXPECT_EQ(spec.set_params(bad).error().code(), cxxgpr::GprErrorCode::InvalidInterval);
    EXPECT_EQ(compiled.set_params(bad).error().code(), cxxgpr::GprErrorCode::InvalidInterval);

    std::array<double, 3> after_spec{};
    std::array<double, 3> after_comp{};
    ASSERT_TRUE(spec.get_params(after_spec).has_value());
    ASSERT_TRUE(compiled.get_params(after_comp).has_value());
    for (std::size_t i = 0; i < 3U; ++i) {
        EXPECT_DOUBLE_EQ(after_spec[i], orig[i]);
        EXPECT_DOUBLE_EQ(after_comp[i], orig[i]);
    }

    const std::array<double, 3> good{std::numbers::ln2, std::log(3.0), std::log(4.0)};
    ASSERT_TRUE(spec.set_params(good).has_value());
    ASSERT_TRUE(compiled.set_params(good).has_value());
    std::vector<cxxgpr::Interval> bounds(3, cxxgpr::Interval::default_positive());
    ASSERT_TRUE(compiled.bounds_into(bounds).has_value());
    EXPECT_EQ(bounds[0], cxxgpr::Interval::default_positive());
}

} // namespace
