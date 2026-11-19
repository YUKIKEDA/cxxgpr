// SPDX-License-Identifier: Apache-2.0
/// @file safetensors.cpp
/// @brief Implementation of read-only memory-mapped Safetensors reader and aligned binary writer.
/// @details Implements platform-specific file mapping (Win32 CreateFileMappingW / POSIX mmap),
/// strict header alignment and bounds validation, finite F64/F32 tensor views, and 8-byte aligned
/// Safetensors file emission.

#include "safetensors.hpp"

#include "cxxgpr/core/checked.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <utility>

#include "json_depth.hpp"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace cxxgpr::persist::detail
{
    namespace
    {

        static_assert(
            std::endian::native == std::endian::little, "safetensors persistence requires little-endian byte order"
        );

        /// @brief Rounds v upward to the nearest 8-byte boundary.
        /// @param[in] v Input byte size or offset.
        /// @return Smallest multiple of 8 greater than or equal to v.
        [[nodiscard]] constexpr std::size_t align_up_8(std::size_t v) noexcept
        {
            return (v + 7U) & ~std::size_t{7U};
        }

        /// @brief Returns element byte width for TensorDtype.
        /// @param[in] dt Tensor scalar type tag.
        /// @return 8 for F64, 4 for F32.
        [[nodiscard]] constexpr std::size_t dtype_size(TensorDtype dt) noexcept
        {
            return (dt == TensorDtype::F64) ? sizeof(double) : sizeof(float);
        }

        /// @brief Maps TensorDtype to its canonical Safetensors JSON dtype string.
        /// @param[in] dt Tensor scalar type tag.
        /// @return "F64" or "F32" string view.
        [[nodiscard]] std::string_view dtype_name(TensorDtype dt) noexcept
        {
            return (dt == TensorDtype::F64) ? "F64" : "F32";
        }

    }  // namespace

    /// @brief Move-constructs a MappedFile while clearing source handles and pointers.
    /// @param[in] other Source MappedFile whose handles are transferred and reset.
    MappedFile::MappedFile(MappedFile&& other) noexcept :
        data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0U))
#ifdef _WIN32
        ,
        file_handle_(std::exchange(other.file_handle_, nullptr)),
        mapping_handle_(std::exchange(other.mapping_handle_, nullptr))
#else
        ,
        fd_(std::exchange(other.fd_, -1))
#endif
    {
    }

    /// @brief Move-assigns a MappedFile after releasing any existing mapping.
    /// @param[in] other Source MappedFile whose handles are transferred and reset.
    /// @return Reference to this MappedFile.
    MappedFile& MappedFile::operator=(MappedFile&& other) noexcept
    {
        if (this != &other)
        {
            // Release currently mapped view and OS handles before taking ownership from other.
            close();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0U);
#ifdef _WIN32
            file_handle_ = std::exchange(other.file_handle_, nullptr);
            mapping_handle_ = std::exchange(other.mapping_handle_, nullptr);
#else
            fd_ = std::exchange(other.fd_, -1);
#endif
        }
        return *this;
    }

    /// @brief Unmaps the read-only memory view and closes underlying OS handles.
    MappedFile::~MappedFile()
    {
        close();
    }

    /// @brief Releases any active OS memory mapping and closes file handles or descriptors.
    void MappedFile::close() noexcept
    {
        // Unmap view first, then close mapping and file handles in reverse acquisition order.
#ifdef _WIN32
        if (data_ != nullptr)
        {
            UnmapViewOfFile(data_);
            data_ = nullptr;
        }
        if (mapping_handle_ != nullptr)
        {
            CloseHandle(static_cast<HANDLE>(mapping_handle_));
            mapping_handle_ = nullptr;
        }
        if (file_handle_ != nullptr)
        {
            CloseHandle(static_cast<HANDLE>(file_handle_));
            file_handle_ = nullptr;
        }
#else
        if (data_ != nullptr && size_ > 0U)
        {
            munmap(const_cast<std::byte*>(data_), size_);
            data_ = nullptr;
        }
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
#endif
        size_ = 0U;
    }

    /// @brief Opens path read-only and maps the entire file into process memory.
    /// @details Uses Win32 CreateFileW / CreateFileMappingW / MapViewOfFile on Windows and POSIX
    /// open / fstat / mmap(PROT_READ, MAP_PRIVATE) on Unix-like systems. Rejects files smaller than 8 bytes.
    /// @param[in] path Filesystem path to model.safetensors.
    /// @return MappedFile owning the read-only memory mapping, or PersistFailed(Io/Tensor) on failure.
    /// @retval GprErrorCode::PersistFailed When opening, statting, size checking, or mapping fails.
    std::expected<MappedFile, GprError> MappedFile::open_read(const std::filesystem::path& path)
    {
        MappedFile out;
#ifdef _WIN32
        // Open shared read-only file handle and establish PAGE_READONLY file mapping on Windows.
        const std::wstring wpath = path.wstring();
        HANDLE fh = CreateFileW(
            wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr
        );
        if (fh == INVALID_HANDLE_VALUE)
        {
            return std::unexpected(GprError::persist_failed(PersistErrorKind::Io, "failed to open safetensors file"));
        }
        LARGE_INTEGER file_size{};
        if (GetFileSizeEx(fh, &file_size) == 0 || file_size.QuadPart < 8)
        {
            CloseHandle(fh);
            return std::unexpected(GprError::persist_failed(PersistErrorKind::Tensor, "safetensors file is too small"));
        }
        HANDLE mh = CreateFileMappingW(fh, nullptr, PAGE_READONLY, 0U, 0U, nullptr);
        if (mh == nullptr)
        {
            CloseHandle(fh);
            return std::unexpected(GprError::persist_failed(PersistErrorKind::Io, "CreateFileMappingW failed"));
        }
        const void* view = MapViewOfFile(mh, FILE_MAP_READ, 0U, 0U, 0U);
        if (view == nullptr)
        {
            CloseHandle(mh);
            CloseHandle(fh);
            return std::unexpected(GprError::persist_failed(PersistErrorKind::Io, "MapViewOfFile failed"));
        }
        out.file_handle_ = fh;
        out.mapping_handle_ = mh;
        out.data_ = static_cast<const std::byte*>(view);
        out.size_ = static_cast<std::size_t>(file_size.QuadPart);
#else
        // Open read-only descriptor and map file read-only private on POSIX platforms.
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
        {
            return std::unexpected(GprError::persist_failed(PersistErrorKind::Io, "failed to open safetensors file"));
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0 || st.st_size < 8)
        {
            ::close(fd);
            return std::unexpected(GprError::persist_failed(PersistErrorKind::Tensor, "safetensors file is too small"));
        }
        const auto fsize = static_cast<std::size_t>(st.st_size);
        void* mapped = ::mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED)
        {
            ::close(fd);
            return std::unexpected(GprError::persist_failed(PersistErrorKind::Io, "mmap failed"));
        }
        out.fd_ = fd;
        out.data_ = static_cast<const std::byte*>(mapped);
        out.size_ = fsize;
#endif
        return out;
    }

    /// @brief Maps and validates a Safetensors file, indexing all F64/F32 tensor records.
    /// @details Verifies the 8-byte header size prefix, 8-byte alignment of data_base, JSON nesting depth
    /// via MAX_JSON_NESTING_DEPTH, overflow-checked shape products, and bounds/alignment of data_offsets.
    /// @param[in] path Filesystem path to model.safetensors.
    /// @return Shared handle to MappedSafetensors, or GprError on I/O, header, or bounds error.
    /// @retval GprErrorCode::PersistFailed When header size, alignment, JSON, dtype, or offsets are invalid.
    /// @retval GprErrorCode::SizeOverflow When tensor shape element product or byte size overflows std::size_t.
    std::expected<std::shared_ptr<const MappedSafetensors>, GprError> MappedSafetensors::open(
        const std::filesystem::path& path
    )
    {
        auto file = MappedFile::open_read(path);
        if (!file.has_value())
        {
            return std::unexpected(std::move(file).error());
        }
        if (file->size() < 8U)
        {
            return std::unexpected(GprError::persist_failed(PersistErrorKind::Tensor, "safetensors header truncated"));
        }

        // Read little-endian uint64 header byte length and verify 8-byte alignment of the data section.
        std::uint64_t header_len_u64 = 0U;
        std::memcpy(&header_len_u64, file->data(), sizeof(std::uint64_t));
        if (header_len_u64 > file->size() - 8U)
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Tensor, "safetensors header length exceeds file size")
            );
        }
        const auto header_len = static_cast<std::size_t>(header_len_u64);
        const std::size_t data_base = 8U + header_len;
        if ((data_base % 8U) != 0U)
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Tensor, "safetensors data section is not 8-byte aligned")
            );
        }

        // Guard against deeply nested header JSON before invoking nlohmann::json::parse.
        const std::string_view header_sv(reinterpret_cast<const char*>(file->data() + 8U), header_len);
        if (auto depth_ok = check_json_max_depth(header_sv, MAX_JSON_NESTING_DEPTH, PersistErrorKind::Tensor);
            !depth_ok.has_value())
        {
            return std::unexpected(std::move(depth_ok).error());
        }
        nlohmann::json header_json;
        try
        {
            header_json = nlohmann::json::parse(header_sv);
        }
        catch (const nlohmann::json::exception&)
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Tensor, "invalid JSON in safetensors header")
            );
        }
        if (!header_json.is_object())
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Tensor, "safetensors header must be a JSON object")
            );
        }

        const std::size_t data_section_bytes = file->size() - data_base;
        std::unordered_map<std::string, TensorRecord> records;

        // Validate every tensor descriptor entry for dtype, overflow-safe shape, alignment, and byte bounds.
        for (const auto& [name, meta] : header_json.items())
        {
            if (name == "__metadata__")
            {
                continue;
            }
            if (!meta.is_object() || !meta.contains("dtype") || !meta.contains("shape")
                || !meta.contains("data_offsets"))
            {
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Tensor, "missing required tensor metadata fields")
                );
            }
            if (!meta["dtype"].is_string() || !meta["shape"].is_array() || !meta["data_offsets"].is_array()
                || meta["data_offsets"].size() != 2U)
            {
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Tensor, "invalid tensor metadata types")
                );
            }

            TensorRecord rec;
            const std::string dt_str = meta["dtype"].get<std::string>();
            if (dt_str == "F64")
            {
                rec.dtype = TensorDtype::F64;
            }
            else if (dt_str == "F32")
            {
                rec.dtype = TensorDtype::F32;
            }
            else
            {
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Tensor, "unsupported safetensors dtype")
                );
            }

            std::size_t elem_count = 1U;
            for (const auto& dim_val : meta["shape"])
            {
                if (!dim_val.is_number_unsigned())
                {
                    return std::unexpected(
                        GprError::persist_failed(
                            PersistErrorKind::Tensor, "tensor shape entries must be unsigned integers"
                        )
                    );
                }
                const auto dim_sz = dim_val.get<std::size_t>();
                auto prod = checked_mul(elem_count, dim_sz);
                if (!prod.has_value())
                {
                    return std::unexpected(std::move(prod).error());
                }
                elem_count = *prod;
                rec.shape.push_back(dim_sz);
            }

            const auto& off = meta["data_offsets"];
            if (!off[0].is_number_unsigned() || !off[1].is_number_unsigned())
            {
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Tensor, "data_offsets must be unsigned integers")
                );
            }
            const auto off_begin = off[0].get<std::size_t>();
            const auto off_end = off[1].get<std::size_t>();
            if (off_end < off_begin || off_end > data_section_bytes)
            {
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Tensor, "tensor data_offsets out of bounds")
                );
            }
            // Enforce 8-byte alignment for every tensor start offset so mmap zero-copy F64 reads are valid.
            if (((data_base + off_begin) % 8U) != 0U)
            {
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Tensor, "tensor data is not 8-byte aligned")
                );
            }
            auto expected_bytes = checked_mul(elem_count, dtype_size(rec.dtype));
            if (!expected_bytes.has_value() || (off_end - off_begin) != *expected_bytes)
            {
                return std::unexpected(
                    GprError::persist_failed(
                        PersistErrorKind::Tensor, "tensor byte length does not match shape and dtype"
                    )
                );
            }
            rec.byte_offset = off_begin;
            rec.byte_length = off_end - off_begin;
            records.emplace(name, std::move(rec));
        }

        auto result = std::make_shared<MappedSafetensors>();
        result->file_ = std::move(*file);
        result->data_base_offset_ = data_base;
        result->records_ = std::move(records);
        return result;
    }

    /// @brief Checks whether a tensor named name exists in the Safetensors header index.
    /// @param[in] name Tensor key name to query.
    /// @return True when name is present in records_.
    bool MappedSafetensors::contains(std::string_view name) const
    {
        return records_.contains(std::string(name));
    }

    /// @brief Returns a validated, zero-copy F64 span directly referencing mapped memory.
    /// @details Checks tensor existence, dtype F64, shape equality, 8-byte alignment, and finiteness of all elements.
    /// @param[in] name Tensor key name to look up.
    /// @param[in] expected_shape Expected tensor dimension extents.
    /// @return Read-only span of double values backed by the mmap view, or PersistFailed(Tensor).
    /// @retval GprErrorCode::PersistFailed When tensor is missing, mismatched, misaligned, or contains NaN/Inf.
    std::expected<std::span<const double>, GprError> MappedSafetensors::read_f64(
        std::string_view name, std::span<const std::size_t> expected_shape
    ) const
    {
        const auto it = records_.find(std::string(name));
        if (it == records_.end())
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Tensor, "missing tensor in model.safetensors")
            );
        }
        const TensorRecord& rec = it->second;
        if (rec.dtype != TensorDtype::F64)
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Tensor, "tensor dtype mismatch (expected F64)")
            );
        }
        if (!std::ranges::equal(rec.shape, expected_shape))
        {
            return std::unexpected(GprError::persist_failed(PersistErrorKind::Tensor, "tensor shape mismatch"));
        }
        // Verify runtime pointer alignment before creating a typed std::span<const double> view.
        const std::byte* ptr = file_.data() + data_base_offset_ + rec.byte_offset;
        if ((reinterpret_cast<std::uintptr_t>(ptr) % alignof(double)) != 0U)
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Tensor, "F64 tensor pointer is not 8-byte aligned")
            );
        }
        const auto count = rec.byte_length / sizeof(double);
        const std::span<const double> view(reinterpret_cast<const double*>(ptr), count);
        // Reject any NaN or +/-Inf scalar before exposing the zero-copy view to inference routines.
        for (const double v : view)
        {
            if (!std::isfinite(v))
            {
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Tensor, "non-finite value in F64 tensor")
                );
            }
        }
        return view;
    }

    /// @brief Returns a validated, zero-copy F32 span directly referencing mapped memory.
    /// @details Checks tensor existence, dtype F32, shape equality, 4-byte alignment, and finiteness of all elements.
    /// @param[in] name Tensor key name to look up.
    /// @param[in] expected_shape Expected tensor dimension extents.
    /// @return Read-only span of float values backed by the mmap view, or PersistFailed(Tensor).
    /// @retval GprErrorCode::PersistFailed When tensor is missing, mismatched, misaligned, or contains NaN/Inf.
    std::expected<std::span<const float>, GprError> MappedSafetensors::read_f32(
        std::string_view name, std::span<const std::size_t> expected_shape
    ) const
    {
        const auto it = records_.find(std::string(name));
        if (it == records_.end())
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Tensor, "missing tensor in model.safetensors")
            );
        }
        const TensorRecord& rec = it->second;
        if (rec.dtype != TensorDtype::F32)
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Tensor, "tensor dtype mismatch (expected F32)")
            );
        }
        if (!std::ranges::equal(rec.shape, expected_shape))
        {
            return std::unexpected(GprError::persist_failed(PersistErrorKind::Tensor, "tensor shape mismatch"));
        }
        // Verify runtime pointer alignment before creating a typed std::span<const float> view.
        const std::byte* ptr = file_.data() + data_base_offset_ + rec.byte_offset;
        if ((reinterpret_cast<std::uintptr_t>(ptr) % alignof(float)) != 0U)
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Tensor, "F32 tensor pointer is not aligned")
            );
        }
        const auto count = rec.byte_length / sizeof(float);
        const std::span<const float> view(reinterpret_cast<const float*>(ptr), count);
        for (const float v : view)
        {
            if (!std::isfinite(v))
            {
                return std::unexpected(
                    GprError::persist_failed(PersistErrorKind::Tensor, "non-finite value in F32 tensor")
                );
            }
        }
        return view;
    }

    /// @brief Serializes a sequence of tensors into an 8-byte aligned Safetensors binary file.
    /// @details Computes 8-byte aligned data offsets, pads the JSON header with spaces so data_base is a
    /// multiple of 8, writes the uint64 header length prefix and header JSON, and writes each tensor payload
    /// preceded by zero-byte alignment padding.
    /// @param[in] path Destination binary file path.
    /// @param[in] tensors Tensor descriptors and little-endian byte spans to write.
    /// @return Empty expected on success, or PersistFailed(Io) on stream open or write failure.
    /// @retval GprErrorCode::PersistFailed When opening, writing, or flushing path fails.
    std::expected<void, GprError> write_safetensors_file(
        const std::filesystem::path& path, std::span<const TensorWriteView> tensors
    )
    {
        // Plan 8-byte aligned offsets for every tensor and construct the JSON header map.
        nlohmann::json header = nlohmann::json::object();
        std::size_t cursor = 0U;
        std::vector<std::pair<std::size_t, std::size_t>> offsets;
        offsets.reserve(tensors.size());

        for (const auto& t : tensors)
        {
            const std::size_t begin = align_up_8(cursor);
            const std::size_t end = begin + t.bytes.size();
            offsets.emplace_back(begin, end);
            cursor = end;

            nlohmann::json meta;
            meta["dtype"] = dtype_name(t.dtype);
            meta["shape"] = t.shape;
            meta["data_offsets"] = nlohmann::json::array({begin, end});
            header[std::string(t.name)] = std::move(meta);
        }

        // Pad JSON header string with spaces to keep (8 + header_len) 8-byte aligned.
        std::string header_str = header.dump();
        const std::size_t padded_header_len = align_up_8(header_str.size());
        header_str.append(padded_header_len - header_str.size(), ' ');
        const auto header_len_u64 = static_cast<std::uint64_t>(header_str.size());

        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        if (!ofs)
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Io, "failed to open temporary safetensors file for writing")
            );
        }
        ofs.write(reinterpret_cast<const char*>(&header_len_u64), sizeof(header_len_u64));
        ofs.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));

        // Emit alignment zero-padding followed by raw tensor bytes for each entry.
        std::size_t written = 0U;
        static constexpr std::array<char, 8> k_zero_pad{};
        for (std::size_t idx = 0; idx < tensors.size(); ++idx)
        {
            const std::size_t pad = offsets[idx].first - written;
            if (pad > 0U)
            {
                ofs.write(k_zero_pad.data(), static_cast<std::streamsize>(pad));
                written += pad;
            }
            const auto& bytes = tensors[idx].bytes;
            if (!bytes.empty())
            {
                ofs.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                written += bytes.size();
            }
        }
        ofs.flush();
        if (!ofs)
        {
            return std::unexpected(
                GprError::persist_failed(PersistErrorKind::Io, "failed while writing safetensors file")
            );
        }
        return {};
    }

}  // namespace cxxgpr::persist::detail
