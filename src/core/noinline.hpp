// SPDX-License-Identifier: Apache-2.0
/// @file noinline.hpp
/// @brief Portable attribute that keeps a function body out of the stack frames of its callers.
/// @details Recursive tree walks (the kernel specification walkers and the config.json encoders and decoders) run
/// every non-recursive step, such as leaf work, object construction, and error construction, in a function marked
/// CXXGPR_NOINLINE. The locals and temporaries of such a step then occupy the stack once, on top of the deepest
/// frame, instead of once per nesting level of the recursion that calls it.

#pragma once

#if defined(_MSC_VER)
/// @brief Forbids the compiler from inlining the function it precedes into its callers.
#    define CXXGPR_NOINLINE __declspec(noinline)
#else
/// @brief Forbids the compiler from inlining the function it precedes into its callers.
#    define CXXGPR_NOINLINE [[gnu::noinline]]
#endif
