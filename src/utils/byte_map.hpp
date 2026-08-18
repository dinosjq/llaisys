#pragma once

#include "llaisys.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaisys::utils {

using ByteMap = std::unordered_map<std::string, std::vector<std::uint8_t>>;

// C API (char* keys + opaque byte vals) → ByteMap used by Model::set_meta.
inline ByteMap make_byte_map(const char *const *keys, const void *const *vals, const size_t *nbytes, size_t n) {
    ByteMap out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (keys == nullptr || vals == nullptr || nbytes == nullptr || keys[i] == nullptr || vals[i] == nullptr) {
            throw std::invalid_argument("byte map entry is null");
        }
        const auto *bytes = static_cast<const std::uint8_t *>(vals[i]);
        out.emplace(keys[i], std::vector<std::uint8_t>(bytes, bytes + nbytes[i]));
    }
    return out;
}

// Homogeneous map: each vals[i] points at a T; copy via reinterpret_cast.
template <typename T>
inline std::unordered_map<std::string, T> make_typed_map(const char *const *keys, const void *const *vals, size_t n) {
    std::unordered_map<std::string, T> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (keys == nullptr || vals == nullptr || keys[i] == nullptr || vals[i] == nullptr) {
            throw std::invalid_argument("typed map entry is null");
        }
        out.emplace(keys[i], *reinterpret_cast<const T *>(vals[i]));
    }
    return out;
}

inline const std::vector<std::uint8_t> &require_key(const ByteMap &kv, const std::string &key) {
    auto it = kv.find(key);
    if (it == kv.end()) {
        throw std::invalid_argument("missing meta key: " + key);
    }
    return it->second;
}

template <typename T>
inline T as(const std::vector<std::uint8_t> &bytes) {
    if (bytes.size() != sizeof(T)) {
        throw std::invalid_argument("byte blob size mismatch for typed read");
    }
    return *reinterpret_cast<const T *>(bytes.data());
}

template <typename T>
inline T require_as(const ByteMap &kv, const std::string &key) {
    return as<T>(require_key(kv, key));
}

template <typename T>
inline T get_or_as(const ByteMap &kv, const std::string &key, T fallback) {
    auto it = kv.find(key);
    if (it == kv.end()) {
        return fallback;
    }
    return as<T>(it->second);
}

inline std::string as_string(const std::vector<std::uint8_t> &bytes) {
    return std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

inline std::string require_string(const ByteMap &kv, const std::string &key) {
    return as_string(require_key(kv, key));
}

inline std::string get_or_string(const ByteMap &kv, const std::string &key, const std::string &fallback) {
    auto it = kv.find(key);
    if (it == kv.end()) {
        return fallback;
    }
    return as_string(it->second);
}

template <typename T>
inline void put_as(ByteMap &kv, const std::string &key, const T &value) {
    const auto *p = reinterpret_cast<const std::uint8_t *>(&value);
    kv[key] = std::vector<std::uint8_t>(p, p + sizeof(T));
}

inline void put_string(ByteMap &kv, const std::string &key, const std::string &value) {
    kv[key] = std::vector<std::uint8_t>(value.begin(), value.end());
}

inline llaisysDataType_t parse_torch_dtype(const std::string &s) {
    std::string lower = s;
    for (char &c : lower) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    if (lower.find("bfloat") != std::string::npos || lower == "bf16") {
        return LLAISYS_DTYPE_BF16;
    }
    if (lower.find("float16") != std::string::npos || lower == "fp16" || lower == "half") {
        return LLAISYS_DTYPE_F16;
    }
    if (lower.find("float32") != std::string::npos || lower == "fp32" || lower == "float") {
        return LLAISYS_DTYPE_F32;
    }
    throw std::invalid_argument("unsupported torch_dtype: " + s);
}

} // namespace llaisys::utils
