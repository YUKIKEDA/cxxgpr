// SPDX-License-Identifier: Apache-2.0
/// @file save_id.cpp
/// @brief FIPS 180-4 SHA-256 implementation and pairing identifier helpers.
/// @details Computes deterministic 128-bit truncated SHA-256 pairing identifiers from checkpoint content
/// without external cryptographic dependencies.

#include "persist/save_id.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace cxxgpr::persist::detail
{

    namespace
    {

        /// @brief Standalone FIPS 180-4 SHA-256 incremental hasher.
        class Sha256 final
        {
        public:
            /// @brief Initializes the SHA-256 state with standard IV constants.
            Sha256() noexcept = default;

            /// @brief Absorbs a raw byte span into the SHA-256 state.
            /// @param[in] data Bytes to hash.
            void update(std::span<const std::uint8_t> data) noexcept
            {
                const std::size_t n = data.size();
                total_bytes_ += static_cast<std::uint64_t>(n);

                std::size_t offset = 0U;
                if (buffer_len_ > 0U)
                {
                    const std::size_t take = std::min(64U - buffer_len_, n);
                    std::memcpy(buffer_.data() + buffer_len_, data.data(), take);
                    buffer_len_ += take;
                    offset += take;
                    if (buffer_len_ == 64U)
                    {
                        compress(buffer_.data());
                        buffer_len_ = 0U;
                    }
                }
                while (offset + 64U <= n)
                {
                    compress(data.data() + offset);
                    offset += 64U;
                }
                if (offset < n)
                {
                    buffer_len_ = n - offset;
                    std::memcpy(buffer_.data(), data.data() + offset, buffer_len_);
                }
            }

            /// @brief Absorbs a 64-bit unsigned integer encoded in little-endian byte order.
            /// @param[in] value Integer value to encode and hash.
            void update_le64(std::uint64_t value) noexcept
            {
                std::array<std::uint8_t, 8> le{};
                for (std::size_t i = 0U; i < 8U; ++i)
                {
                    le[i] = static_cast<std::uint8_t>((value >> (8U * i)) & 0xFFU);
                }
                update(le);
            }

            /// @brief Absorbs an arbitrary byte-convertible view into the SHA-256 state.
            /// @param[in] text Character sequence to hash.
            void update_str(std::string_view text) noexcept
            {
                update(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
            }

            /// @brief Absorbs an arbitrary std::byte span into the SHA-256 state.
            /// @param[in] bytes Raw byte span to hash.
            void update_bytes(std::span<const std::byte> bytes) noexcept
            {
                update(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
            }

            /// @brief Finalizes padding and returns the 32-byte SHA-256 digest.
            /// @return Completed 256-bit digest in big-endian byte order.
            [[nodiscard]] std::array<std::uint8_t, 32> finish() noexcept
            {
                const std::uint64_t bit_len = total_bytes_ * 8ULL;
                const std::uint8_t pad_one = 0x80U;
                update(std::span<const std::uint8_t>(&pad_one, 1U));

                static constexpr std::array<std::uint8_t, 64> k_zeros{};
                if (buffer_len_ > 56U)
                {
                    update(std::span<const std::uint8_t>(k_zeros.data(), 64U - buffer_len_));
                }
                update(std::span<const std::uint8_t>(k_zeros.data(), 56U - buffer_len_));

                std::array<std::uint8_t, 8> be_len{};
                for (std::size_t i = 0U; i < 8U; ++i)
                {
                    be_len[7U - i] = static_cast<std::uint8_t>((bit_len >> (8U * i)) & 0xFFU);
                }
                update(be_len);

                std::array<std::uint8_t, 32> digest{};
                for (std::size_t i = 0U; i < 8U; ++i)
                {
                    digest[(4U * i) + 0U] = static_cast<std::uint8_t>((state_[i] >> 24U) & 0xFFU);
                    digest[(4U * i) + 1U] = static_cast<std::uint8_t>((state_[i] >> 16U) & 0xFFU);
                    digest[(4U * i) + 2U] = static_cast<std::uint8_t>((state_[i] >> 8U) & 0xFFU);
                    digest[(4U * i) + 3U] = static_cast<std::uint8_t>(state_[i] & 0xFFU);
                }
                return digest;
            }

        private:
            /// @brief Compresses one 64-byte message block into the SHA-256 state.
            /// @param[in] block Pointer to 64 contiguous input bytes.
            void compress(const std::uint8_t* block) noexcept
            {
                static constexpr std::array<std::uint32_t, 64> k_round{
                    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
                    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
                    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
                    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
                    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
                    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
                    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
                    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
                    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
                    0xc67178f2U,
                };

                std::array<std::uint32_t, 64> w{};
                for (std::size_t i = 0U; i < 16U; ++i)
                {
                    w[i] = (static_cast<std::uint32_t>(block[(4U * i) + 0U]) << 24U)
                        | (static_cast<std::uint32_t>(block[(4U * i) + 1U]) << 16U)
                        | (static_cast<std::uint32_t>(block[(4U * i) + 2U]) << 8U)
                        | static_cast<std::uint32_t>(block[(4U * i) + 3U]);
                }
                for (std::size_t i = 16U; i < 64U; ++i)
                {
                    const std::uint32_t s0 =
                        std::rotr(w[i - 15U], 7) ^ std::rotr(w[i - 15U], 18) ^ (w[i - 15U] >> 3U);
                    const std::uint32_t s1 =
                        std::rotr(w[i - 2U], 17) ^ std::rotr(w[i - 2U], 19) ^ (w[i - 2U] >> 10U);
                    w[i] = w[i - 16U] + s0 + w[i - 7U] + s1;
                }

                std::uint32_t a = state_[0];
                std::uint32_t b = state_[1];
                std::uint32_t c = state_[2];
                std::uint32_t d = state_[3];
                std::uint32_t e = state_[4];
                std::uint32_t f = state_[5];
                std::uint32_t g = state_[6];
                std::uint32_t h = state_[7];

                for (std::size_t i = 0U; i < 64U; ++i)
                {
                    const std::uint32_t s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
                    const std::uint32_t ch = (e & f) ^ ((~e) & g);
                    const std::uint32_t temp1 = h + s1 + ch + k_round[i] + w[i];
                    const std::uint32_t s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
                    const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                    const std::uint32_t temp2 = s0 + maj;

                    h = g;
                    g = f;
                    f = e;
                    e = d + temp1;
                    d = c;
                    c = b;
                    b = a;
                    a = temp1 + temp2;
                }

                state_[0] += a;
                state_[1] += b;
                state_[2] += c;
                state_[3] += d;
                state_[4] += e;
                state_[5] += f;
                state_[6] += g;
                state_[7] += h;
            }

            /// @brief Working 256-bit hash state initialized to FIPS 180-4 SHA-256 initial values.
            std::array<std::uint32_t, 8> state_{
                0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
            };
            /// @brief Partial 64-byte input block buffer.
            std::array<std::uint8_t, 64> buffer_{};
            /// @brief Number of buffered bytes currently held in buffer_.
            std::size_t buffer_len_{0U};
            /// @brief Total number of input bytes absorbed so far.
            std::uint64_t total_bytes_{0ULL};
        };

    }  // namespace

    /// @brief Returns true if text is a valid 32-lowercase-hex-digit pairing identifier.
    /// @param[in] text Candidate identifier string.
    /// @return True iff text has length 32 and every character is in `[0-9a-f]`.
    bool is_valid_save_id(std::string_view text) noexcept
    {
        if (text.size() != 32U)
        {
            return false;
        }
        for (const char ch : text)
        {
            const bool is_digit = (ch >= '0' && ch <= '9');
            const bool is_lower_hex = (ch >= 'a' && ch <= 'f');
            if (!is_digit && !is_lower_hex)
            {
                return false;
            }
        }
        return true;
    }

    /// @brief Computes the pairing identifier for a checkpoint from config.json (without save_id) and tensors.
    /// @details Hashes the domain tag `cxxgpr.save_id.v1\0`, length-prefixed config JSON bytes, and length-prefixed
    /// tensor descriptors and raw payloads in storage offset order using SHA-256, returning the first 16 bytes
    /// as 32 lowercase hexadecimal digits.
    /// @param[in] config_json_without_id Serialized config.json bytes without the `save_id` top-level key.
    /// @param[in] tensors Ordered tensor descriptors and raw payloads in model.safetensors offset order.
    /// @return 32-character lowercase hexadecimal pairing identifier.
    std::string compute_save_id(std::string_view config_json_without_id, std::span<const TensorWriteView> tensors)
    {
        Sha256 hasher;
        // 1. 17 ASCII bytes `cxxgpr.save_id.v1` followed by one 0x00 byte.
        static constexpr std::array<std::uint8_t, 18> k_domain{
            'c', 'x', 'x', 'g', 'p', 'r', '.', 's', 'a', 'v', 'e', '_', 'i', 'd', '.', 'v', '1', 0x00U,
        };
        hasher.update(k_domain);

        // 2. le64(|C|) and the bytes C (config.json without save_id).
        hasher.update_le64(static_cast<std::uint64_t>(config_json_without_id.size()));
        hasher.update_str(config_json_without_id);

        // 3. le64(k) and each tensor descriptor + raw payload in offset order.
        hasher.update_le64(static_cast<std::uint64_t>(tensors.size()));
        for (const auto& t : tensors)
        {
            hasher.update_le64(static_cast<std::uint64_t>(t.name.size()));
            hasher.update_str(t.name);

            const std::string_view dt_str = (t.dtype == TensorDtype::F64) ? "F64" : "F32";
            hasher.update_le64(static_cast<std::uint64_t>(dt_str.size()));
            hasher.update_str(dt_str);

            hasher.update_le64(static_cast<std::uint64_t>(t.shape.size()));
            for (const std::size_t dim : t.shape)
            {
                hasher.update_le64(static_cast<std::uint64_t>(dim));
            }

            hasher.update_le64(static_cast<std::uint64_t>(t.bytes.size()));
            hasher.update_bytes(t.bytes);
        }

        const auto digest = hasher.finish();
        static constexpr char k_hex[] = "0123456789abcdef";
        std::string out(32U, '0');
        for (std::size_t i = 0U; i < 16U; ++i)
        {
            out[(2U * i) + 0U] = k_hex[(digest[i] >> 4U) & 0x0FU];
            out[(2U * i) + 1U] = k_hex[digest[i] & 0x0FU];
        }
        return out;
    }

    /// @brief Verifies that config.json and model.safetensors pairing identifiers agree on load.
    /// @details Succeeds when both identifiers are absent (legacy checkpoints) or both are present and equal.
    /// Rejects mismatched or one-sided identifiers with PersistFailed(Tensor).
    /// @param[in] config_id Optional validated `save_id` read from config.json.
    /// @param[in] tensor_id Optional validated `cxxgpr.save_id` read from model.safetensors `__metadata__`.
    /// @return Empty expected on agreement, or PersistFailed(Tensor) when files come from different saves.
    /// @retval GprErrorCode::PersistFailed When pairing identifiers differ or only one file carries an identifier.
    std::expected<void, GprError> check_save_id_pairing(
        const std::optional<std::string>& config_id, const std::optional<std::string>& tensor_id
    )
    {
        if (!config_id.has_value() && !tensor_id.has_value())
        {
            return {};
        }
        if (config_id.has_value() && tensor_id.has_value() && *config_id == *tensor_id)
        {
            return {};
        }
        return std::unexpected(
            GprError::persist_failed(
                PersistErrorKind::Tensor, "config.json and model.safetensors come from different saves"
            )
        );
    }

}  // namespace cxxgpr::persist::detail
