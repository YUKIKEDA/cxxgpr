// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cxxgpr::persist::detail
{

    enum class TensorDtype : std::uint8_t
    {
        F64,
        F32,
    };

    struct TensorWriteView
    {
        std::string_view name;
        TensorDtype dtype{TensorDtype::F64};
        std::vector<std::size_t> shape;
        std::span<const std::byte> bytes;
    };

    struct TensorRecord
    {
        TensorDtype dtype{TensorDtype::F64};
        std::vector<std::size_t> shape;
        std::size_t byte_offset{0};
        std::size_t byte_length{0};
    };

    class MappedFile
    {
    public:
        MappedFile() noexcept = default;
        MappedFile(const MappedFile&) = delete;
        MappedFile& operator=(const MappedFile&) = delete;
        MappedFile(MappedFile&& other) noexcept;
        MappedFile& operator=(MappedFile&& other) noexcept;
        ~MappedFile();

        [[nodiscard]] static std::expected<MappedFile, GprError> open_read(const std::filesystem::path& path);

        [[nodiscard]] const std::byte* data() const noexcept { return data_; }

        [[nodiscard]] std::size_t size() const noexcept { return size_; }

    private:
        void close() noexcept;

        const std::byte* data_{nullptr};
        std::size_t size_{0};
#ifdef _WIN32
        void* file_handle_{nullptr};
        void* mapping_handle_{nullptr};
#else
        int fd_{-1};
#endif
    };

    class MappedSafetensors
    {
    public:
        MappedSafetensors() = default;

        [[nodiscard]] static std::expected<std::shared_ptr<const MappedSafetensors>, GprError> open(
            const std::filesystem::path& path
        );

        [[nodiscard]] bool contains(std::string_view name) const;

        [[nodiscard]] std::expected<std::span<const double>, GprError> read_f64(
            std::string_view name, std::span<const std::size_t> expected_shape
        ) const;

        [[nodiscard]] std::expected<std::span<const float>, GprError> read_f32(
            std::string_view name, std::span<const std::size_t> expected_shape
        ) const;

    private:
        MappedFile file_;
        std::size_t data_base_offset_{0};
        std::unordered_map<std::string, TensorRecord> records_;
    };

    [[nodiscard]] std::expected<void, GprError> write_safetensors_file(
        const std::filesystem::path& path, std::span<const TensorWriteView> tensors
    );

}  // namespace cxxgpr::persist::detail
