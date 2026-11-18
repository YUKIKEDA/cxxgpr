// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"

#include <cstddef>
#include <expected>
#include <string_view>

namespace cxxgpr::persist::detail
{

    inline constexpr std::size_t MAX_JSON_NESTING_DEPTH = 128U;

    /// Iteratively checks that JSON object/array nesting does not exceed max_depth before parsing.
    [[nodiscard]] inline std::expected<void, GprError> check_json_max_depth(
        std::string_view text, std::size_t max_depth, PersistErrorKind kind
    )
    {
        std::size_t depth = 0U;
        bool in_string = false;
        bool escaped = false;
        for (const char ch : text)
        {
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
                ++depth;
                if (depth > max_depth)
                {
                    return std::unexpected(GprError::persist_failed(kind, "JSON nesting depth exceeds limit"));
                }
            }
            else if (ch == '}' || ch == ']')
            {
                if (depth > 0U)
                {
                    --depth;
                }
            }
        }
        return {};
    }

}  // namespace cxxgpr::persist::detail
