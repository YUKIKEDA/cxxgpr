// SPDX-License-Identifier: Apache-2.0
/// @file workspace.hpp
/// @brief Umbrella header aggregating all public Exact GPR workspace headers.
/// @details Includes buffer containers (buffers.hpp), posterior inference and LOO predictors
/// (inference.hpp), incremental per-leaf Gram cache (leaf_cache.hpp), and marginal likelihood /
/// derivative evaluators (mll.hpp).
#pragma once

#include "cxxgpr/workspace/buffers.hpp"
#include "cxxgpr/workspace/inference.hpp"
#include "cxxgpr/workspace/leaf_cache.hpp"
#include "cxxgpr/workspace/mll.hpp"
