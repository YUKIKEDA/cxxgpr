// SPDX-License-Identifier: Apache-2.0

/// @file checkpoint_io.hpp
/// @brief Library-private atomic writer shared by the coordinate and supplied-distance checkpoints.

#pragma once

#include "cxxgpr/core/error.hpp"

#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "safetensors.hpp"

/// @brief Library-internal helpers of the persistence layer.
namespace cxxgpr::persist::detail
{

    /// @brief Writes config.json and model.safetensors through staged temporary files and renames them into place.
    /// @param[in] dir Destination directory that already exists.
    /// @param[in] cfg_str Serialized config.json text.
    /// @param[in] tensors Tensors of model.safetensors.
    /// @param[in] save_id Optional pairing identifier written to model.safetensors `__metadata__`.
    /// @return Empty expected on success, or GprError when writing or renaming fails (temporary files are removed).
    /// @retval GprErrorCode::PersistFailed When a file cannot be written or renamed.
    [[nodiscard]] std::expected<void, GprError> write_checkpoint_files(
        const std::filesystem::path& dir,
        const std::string& cfg_str,
        std::span<const TensorWriteView> tensors,
        std::optional<std::string_view> save_id = std::nullopt
    );

}  // namespace cxxgpr::persist::detail
