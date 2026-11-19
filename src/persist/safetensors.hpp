// SPDX-License-Identifier: Apache-2.0
/// @file safetensors.hpp
/// @brief Internal memory-mapped Safetensors reader and aligned binary writer.
/// @details Provides OS-backed read-only file mapping (CreateFileMappingW on Windows, mmap on POSIX),
/// Safetensors header parsing with 8-byte alignment and bounds checks, zero-copy F64/F32 tensor views,
/// and aligned Safetensors file serialization.

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

/// @brief Internal persistence helpers for Exact GPR serialization and loading.
namespace cxxgpr::persist::detail
{

    /// @brief Floating-point tensor element types supported in model.safetensors.
    /// @details Restricted to little-endian IEEE 754 binary64 (F64) and binary32 (F32).
    enum class TensorDtype : std::uint8_t
    {
        /// @brief 64-bit IEEE 754 floating-point tensor ("F64").
        F64,
        /// @brief 32-bit IEEE 754 floating-point tensor ("F32").
        F32,
    };

    /// @brief Non-owning tensor slice descriptor passed to write_safetensors_file().
    /// @details Describes tensor name, scalar dtype, shape dimensions, and raw little-endian bytes.
    struct TensorWriteView
    {
        /// @brief Tensor key name stored in the Safetensors JSON header.
        std::string_view name;
        /// @brief Element scalar type (F64 or F32).
        TensorDtype dtype{TensorDtype::F64};
        /// @brief Dimension extents of the tensor.
        std::vector<std::size_t> shape;
        /// @brief Contiguous raw little-endian tensor bytes matching shape and dtype.
        std::span<const std::byte> bytes;
    };

    /// @brief Parsed metadata record for a single tensor inside a mapped Safetensors file.
    /// @details Stores validated dtype, shape, relative byte offset from data_base_offset_, and byte length.
    struct TensorRecord
    {
        /// @brief Validated element scalar type.
        TensorDtype dtype{TensorDtype::F64};
        /// @brief Validated dimension extents.
        std::vector<std::size_t> shape;
        /// @brief Byte offset relative to the start of the Safetensors data section.
        std::size_t byte_offset{0};
        /// @brief Exact byte length verified against shape element product and dtype size.
        std::size_t byte_length{0};
    };

    /// @brief RAII wrapper owning a read-only memory-mapped file region.
    /// @details Uses CreateFileW/CreateFileMappingW/MapViewOfFile on Windows and open/mmap on POSIX.
    class MappedFile
    {
    public:
        /// @brief Constructs an empty, unmapped MappedFile.
        MappedFile() noexcept = default;

        /// @brief Deleted copy constructor because OS mapping handles have unique ownership.
        MappedFile(const MappedFile&) = delete;

        /// @brief Deleted copy assignment because OS mapping handles have unique ownership.
        /// @return Reference to this MappedFile.
        MappedFile& operator=(const MappedFile&) = delete;

        /// @brief Move-constructs a MappedFile by transferring OS mapping ownership.
        /// @param[in] other Source MappedFile whose handles are reset to null/invalid.
        MappedFile(MappedFile&& other) noexcept;

        /// @brief Move-assigns a MappedFile after closing any currently held mapping.
        /// @param[in] other Source MappedFile whose handles are reset to null/invalid.
        /// @return Reference to this MappedFile.
        MappedFile& operator=(MappedFile&& other) noexcept;

        /// @brief Unmaps the file view and closes all underlying OS file and mapping handles.
        ~MappedFile();

        /// @brief Opens path and maps its entire contents read-only into virtual memory.
        /// @details Requires the file to be at least 8 bytes long so the Safetensors header prefix fits.
        /// @param[in] path Filesystem path to the Safetensors binary file.
        /// @return MappedFile owning the read-only mapping, or a PersistFailed error on I/O or size failure.
        /// @retval GprErrorCode::PersistFailed When opening, statting, or memory-mapping the file fails.
        [[nodiscard]] static std::expected<MappedFile, GprError> open_read(const std::filesystem::path& path);

        /// @brief Returns a pointer to the first byte of the read-only mapped region.
        /// @return Pointer to mapped bytes, or nullptr when unmapped.
        [[nodiscard]] const std::byte* data() const noexcept { return data_; }

        /// @brief Returns the total mapped byte count.
        /// @return File size in bytes, or 0 when unmapped.
        [[nodiscard]] std::size_t size() const noexcept { return size_; }

    private:
        /// @brief Releases any active memory mapping and closes OS file descriptors or handles.
        void close() noexcept;

        /// @brief Base pointer to the read-only mapped memory view.
        const std::byte* data_{nullptr};
        /// @brief Total mapped file size in bytes.
        std::size_t size_{0};
#ifdef _WIN32
        /// @brief Native Win32 file HANDLE opened for shared read access.
        void* file_handle_{nullptr};
        /// @brief Native Win32 read-only file mapping HANDLE.
        void* mapping_handle_{nullptr};
#else
        /// @brief POSIX file descriptor opened read-only.
        int fd_{-1};
#endif
    };

    /// @brief Validated read-only Safetensors archive providing aligned, finite tensor views.
    /// @details Holds a MappedFile alive via std::shared_ptr so callers can retain zero-copy F64 spans
    /// safely for the lifetime of LoadedGpr.
    class MappedSafetensors
    {
    public:
        /// @brief Constructs an empty MappedSafetensors container for factory initialization.
        MappedSafetensors() = default;

        /// @brief Maps and validates a Safetensors file header and tensor index table.
        /// @details Checks the 8-byte header size prefix, 8-byte alignment of data_base_offset_, JSON depth
        /// limits, tensor shapes (with overflow checking), and non-overlapping bounds inside the file.
        /// @param[in] path Filesystem path to model.safetensors.
        /// @return Shared handle to the validated MappedSafetensors archive, or GprError on failure.
        /// @retval GprErrorCode::PersistFailed When file mapping, JSON parsing, alignment, or bounds fail.
        /// @retval GprErrorCode::SizeOverflow When tensor shape element multiplication overflows std::size_t.
        [[nodiscard]] static std::expected<std::shared_ptr<const MappedSafetensors>, GprError> open(
            const std::filesystem::path& path
        );

        /// @brief Returns true if a tensor named name exists in the parsed header index.
        /// @param[in] name Tensor key name to look up.
        /// @return True when name is present in records_.
        [[nodiscard]] bool contains(std::string_view name) const;

        /// @brief Returns a zero-copy F64 span into the mapped file after validating shape, alignment, and finiteness.
        /// @details Verifies tensor presence, TensorDtype::F64, exact shape equality, 8-byte pointer alignment,
        /// and finiteness of every stored double element.
        /// @param[in] name Tensor key name to retrieve.
        /// @param[in] expected_shape Expected tensor dimension extents.
        /// @return Read-only span of double values backed directly by the memory mapping, or GprError.
        /// @retval GprErrorCode::PersistFailed When tensor is missing, mismatched, misaligned, or non-finite.
        [[nodiscard]] std::expected<std::span<const double>, GprError> read_f64(
            std::string_view name, std::span<const std::size_t> expected_shape
        ) const;

        /// @brief Returns a zero-copy F32 span into the mapped file after validating shape, alignment, and finiteness.
        /// @details Verifies tensor presence, TensorDtype::F32, exact shape equality, 4-byte pointer alignment,
        /// and finiteness of every stored float element.
        /// @param[in] name Tensor key name to retrieve.
        /// @param[in] expected_shape Expected tensor dimension extents.
        /// @return Read-only span of float values backed directly by the memory mapping, or GprError.
        /// @retval GprErrorCode::PersistFailed When tensor is missing, mismatched, misaligned, or non-finite.
        [[nodiscard]] std::expected<std::span<const float>, GprError> read_f32(
            std::string_view name, std::span<const std::size_t> expected_shape
        ) const;

    private:
        /// @brief Underlying read-only memory-mapped file keeping all tensor spans valid.
        MappedFile file_;
        /// @brief Byte offset where tensor binary payloads begin (8 + padded header length).
        std::size_t data_base_offset_{0};
        /// @brief Map from tensor name to validated TensorRecord metadata.
        std::unordered_map<std::string, TensorRecord> records_;
    };

    /// @brief Writes a Safetensors file with an 8-byte aligned JSON header and 8-byte aligned tensor payloads.
    /// @details Pads the JSON header with ASCII spaces and inter-tensor gaps with zero bytes so every
    /// stored tensor begins on an 8-byte boundary suitable for zero-copy mmap access.
    /// @param[in] path Destination path (typically a temporary staging file before atomic rename).
    /// @param[in] tensors Sequence of tensor descriptors and raw byte spans to serialize.
    /// @return Empty expected on success, or PersistFailed(Io) on stream write failure.
    /// @retval GprErrorCode::PersistFailed When opening or writing the Safetensors file fails.
    [[nodiscard]] std::expected<void, GprError> write_safetensors_file(
        const std::filesystem::path& path, std::span<const TensorWriteView> tensors
    );

}  // namespace cxxgpr::persist::detail
