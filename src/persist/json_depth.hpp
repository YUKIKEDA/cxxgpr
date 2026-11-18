// SPDX-License-Identifier: Apache-2.0
/// @file json_depth.hpp
/// @brief Iterative JSON nesting depth guard prior to recursive parser invocation.
/// @details Scans raw JSON text linearly while tracking string-literal escaping so deeply nested
/// payloads are rejected deterministically without risking parser stack overflow.

#pragma once

#include "cxxgpr/core/error.hpp"

#include <cstddef>
#include <expected>
#include <string_view>

/// @brief Internal persistence helpers for Exact GPR serialization and loading.
namespace cxxgpr::persist::detail
{

    /// @brief Maximum allowed JSON object and array nesting depth for persisted config and tensor headers.
    inline constexpr std::size_t MAX_JSON_NESTING_DEPTH = 128U;

    /// @brief Iteratively checks that JSON object and array nesting does not exceed max_depth before parsing.
    /// @details Performs a single linear pass over text while ignoring braces and brackets inside string
    /// literals, avoiding stack exhaustion on adversarial or corrupted JSON inputs.
    /// @param text Raw JSON text to inspect.
    /// @param max_depth Maximum permissible nesting depth of '{' and '[' delimiters.
    /// @param kind Persistence error subcategory reported if max_depth is exceeded.
    /// @return Empty expected when nesting stays within max_depth, or a PersistFailed error otherwise.
    /// @retval GprErrorCode::PersistFailed When structural nesting exceeds max_depth.
    [[nodiscard]] inline std::expected<void, GprError> check_json_max_depth(
        std::string_view text, std::size_t max_depth, PersistErrorKind kind
    )
    {
        std::size_t depth = 0U;
        bool in_string = false;
        bool escaped = false;
        for (const char ch : text)
        {
            // Track string-literal state and backslash escapes so delimiters inside strings are ignored.
            if (in_string)
            {
                if (escaped)
                {
                    escaped = false;
                }
                else if (ch == '\\')
                {
                    escaped = true;
                }
                else if (ch == '"')
                {
                    in_string = false;
                }
                continue;
            }
            if (ch == '"')
            {
                in_string = true;
            }
            else if (ch == '{' || ch == '[')
            {
                // Fail immediately once structural nesting depth surpasses max_depth.
                ++depth;
                if (depth > max_depth)
                {
                    return std::unexpected(GprError::persist_failed(kind, "JSON nesting depth exceeds limit"));
                }
            }
            else if (ch == '}' || ch == ']')
            {
                // Clamp decrements at zero so malformed closing delimiters are left to the JSON parser.
                if (depth > 0U)
                {
                    --depth;
                }
            }
        }
        return {};
    }

}  // namespace cxxgpr::persist::detail
