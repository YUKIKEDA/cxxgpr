// SPDX-License-Identifier: Apache-2.0
/// @file kernel.hpp
/// @brief Umbrella header aggregating all public kernel subsystem headers.
/// @details Includes built-in kernel leaf types, expression-tree specifications, compiled
/// execution-layer evaluators, distance caches, supplied distance slots, exponential evaluation
/// backends, and custom user-defined kernel interfaces.
#pragma once

/// @defgroup cxxgpr_kernel Kernel Expressions and Evaluators
/// @brief Built-in kernels, expression composition, distance caches, and custom kernel extensions.
/// @{
#include "cxxgpr/kernel/compiled.hpp"
#include "cxxgpr/kernel/custom.hpp"
#include "cxxgpr/kernel/distance.hpp"
#include "cxxgpr/kernel/distance_cache.hpp"
#include "cxxgpr/kernel/exp.hpp"
#include "cxxgpr/kernel/leaves.hpp"
#include "cxxgpr/kernel/spec.hpp"
/// @}
