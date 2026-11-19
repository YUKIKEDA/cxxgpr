// SPDX-License-Identifier: Apache-2.0
/// @file optimizer.hpp
/// @brief Umbrella header aggregating all public cxxgpr hyperparameter optimizer headers.
/// @details Includes log-space bounds transforms, objective hierarchy, Fixed, FastSimulatedAnnealing,
/// Lbfgs, NelderMead, and TrustRegion solvers.
#pragma once

/// @defgroup cxxgpr_optimizer Hyperparameter Optimizers
/// @brief Log-space bounds transforms, objective hierarchy, and local/global optimization solvers.
/// @{
#include "cxxgpr/optimizer/bounds.hpp"
#include "cxxgpr/optimizer/fixed.hpp"
#include "cxxgpr/optimizer/fsa.hpp"
#include "cxxgpr/optimizer/lbfgs.hpp"
#include "cxxgpr/optimizer/nelder_mead.hpp"
#include "cxxgpr/optimizer/objective.hpp"
#include "cxxgpr/optimizer/trust_region.hpp"
/// @}
