#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstdint>
#include <complex>
#include <sstream>
#include <cstdint>
#include <numeric>
#include <filesystem>
#include <cstddef>

namespace detail
{
// Numpy type map, see
// https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#arrays-dtypes-constructing
// And another function to convert given std datatype to numpy representation.
template <typename t>
struct MapType
    : std::false_type
{
};

template <>
struct MapType<bool>
    : std::true_type
{
    static constexpr const char* const typeSpecifier = "?";
};

template <>
struct MapType<char>
    : std::true_type
{
    static constexpr const char* const typeSpecifier = "b";
};

template <>
struct MapType<std::byte>
    : std::true_type
{
    static constexpr const char* const typeSpecifier = "B";
};

template <>
struct MapType<float>
    : std::true_type
{
    static constexpr const char* const typeSpecifier = "f4";
};

template <>
struct MapType<double>
    : std::true_type
{
    static constexpr const char* const typeSpecifier = "f8";
};

template <>
struct MapType<uint8_t>
    : std::true_type
{
    static constexpr const char* const typeSpecifier = "u1";
};

template <>
struct MapType<uint16_t>
    : std::true_type
{
    static constexpr const char* const typeSpecifier = "u2";
};

template <>
struct MapType<uint32_t>
    : std::true_type
{
    static constexpr const char* const typeSpecifier = "u4";
};

template <>
struct MapType<uint64_t>
    : std::true_type
{
    static constexpr const char* const typeSpecifier = "u8";
};

template <>
struct MapType<int16_t>
    : std::true_type
{
    static constexpr const char* const typeSpecifier = "i2";
};

template <>
struct MapType<int32_t>
    : std::true_type
{
    static constexpr const char* const typeSpecifier = "i4";
};

template <>
struct MapType<int64_t>
    : std::true_type
{
    static constexpr const char* const typeSpecifier = "i8";
};

template <>
struct MapType<std::complex<float> >
    : std::true_type
{
    static constexpr const char* const typeSpecifier = "c";
};

// see
// https://docs.scipy.org/doc/numpy-1.14.2/neps/npy-format.html#format-specification-version-2-0
template <typename T, int columns>
class NpyHeader
{
public:
    explicit NpyHeader(const std::array<size_t, columns>& shape)
        : m_headerDict(buildHeaderDictionary(shape))
    {
    }

    NpyHeader() = delete;

    template <typename T, int columns>
    friend std::ostream& operator<<(
        std::ostream& out, const NpyHeader<T, columns>&);

private:
    static constexpr std::array<uint8_t, 8> preEmble = { 0x93, 'N', 'U', 'M',
                                                         'P', 'Y',
                                                         0x02, // format version 2.0 (2 bytes)
                                                         0x00 };

    std::string m_headerDict;
    // TODO: replace me with std::endian in c++ 20
    constexpr char endianTest()
    {
        unsigned char x[] = { 1, 0 };
        short y = *(short*)x;
        return y == 1 ? '<' : '>';
    }

    std::string buildHeaderDictionary(const std::array<size_t, columns>& shape)
    {
        const auto endian = endianTest();

        static_assert(MapType<T>::value,
                      "Must specialize map_type for given element type T");

        constexpr auto mappedType = MapType<T>::typeSpecifier;

        std::stringstream stream;
        stream << "{'descr': '";
        stream << endian << mappedType;
        stream << "', 'fortran_order': False, 'shape': (";
        stream << std::to_string(shape[0]);
        for (size_t i = 1; i < shape.size(); i++)
        {
            stream << ",";
            stream << std::to_string(shape[i]);
        }
        if (shape.size() == 1)
        {
            stream << ",";
        }

        stream << "), }";

        return stream.str();
    }
};

template <typename T, int columns>
std::ostream& operator<<(std::ostream& out, const NpyHeader<T, columns>& header)
{
    // From the numpy docs:
    // The next HEADER_LEN bytes form the header data describing the arrays
    // format. It is an ASCII string which contains a Python literal
    // expression of a dictionary. It is terminated by a newline (n) and
    // padded with spaces (x20) to make the total length of the magic
    // string (preemble) + 4 + HEADER_LEN be evenly divisible by 16 for
    // alignment purposes.

    const auto totalHeaderLengthNoPadding =
        header.preEmble.size() + sizeof(uint32_t) // header len size
        + header.m_headerDict.size() + 1; // +1 = newline

    for (const auto& c : header.preEmble)
    {
        out << c;
    }
    const auto paddingRequired = 16 - (totalHeaderLengthNoPadding % 16);
    const auto headerLength =
        uint32_t(header.m_headerDict.size() + paddingRequired + 1);

    // HEADER_LEN, must be 4 bytes, as described as a change for
    // format 2.0 from version 1.0
    std::array<char, sizeof(uint32_t)> buffer;
    std::memcpy(buffer.data(), &headerLength, buffer.size());

    out.write(buffer.data(), buffer.size());
    out << header.m_headerDict;

    for (auto i = 0u; i < paddingRequired; ++i)
    {
        out << ' ';
    }
    out << '\n';

    return out;
}

std::unique_ptr<std::ostream> openForWriting(const std::filesystem::path& path);

template <typename T, int columns, typename Stream>
void writeNpyHeader(Stream& stream, const std::array<size_t, columns>& shape)
{
    using namespace ::detail;

    NpyHeader<T, columns> header(shape);
    stream << header;
}
} // namespace detail

template <int columns, typename Stream, typename Iterator>
void writeNpy(Stream& stream, Iterator begin, Iterator end,
               const std::array<size_t, columns>& shape)
{
    using T = typename std::iterator_traits<Iterator>::value_type;

    ::detail::writeNpyHeader<T, columns, Stream>(stream, shape);

    if constexpr (std::is_same_v<typename std::iterator_traits<
                                     Iterator>::iterator_category,
                                 std::random_access_iterator_tag>)
    {
        [[maybe_unused]] const size_t expectedElements = std::accumulate(
            shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());

        const auto bytes = ptrdiff_t(end - begin) * sizeof(T);

        std::vector<char> buffer(bytes, 0);
        std::memcpy(buffer.data(), &*begin, bytes);
        stream.write(buffer.data(), bytes);
    }
    else
    {
        auto it = begin;

        std::array<char, sizeof(T)> buffer;

        while (it != end)
        {
            std::memcpy(buffer.data(), &*it, sizeof(T));
            stream.write(buffer.data(), buffer.size());
            it++;
        }
    }
}

template <typename T, int columns, typename Stream>
void writeNpy(Stream& stream, const std::vector<T>& data,
               const std::array<size_t, columns>& shape)
{
    writeNpy(stream, data.cbegin(), data.cend(), shape);
}

template <typename T, int columns>
void writeNpy(const std::filesystem::path& path, const std::vector<T>& data,
               const std::array<size_t, columns>& shape)
{
    writeNpy<T, columns>(*::detail::openForWriting(path), data, shape);
}

template <int columns, typename Iterator>
void writeNpy(const std::filesystem::path& path, Iterator begin, Iterator end,
               const std::array<size_t, columns>& shape)
{
    writeNpy(*::detail::openForWriting(path), begin, end, shape);
}

template <typename T>
void writeNpy(const std::filesystem::path& path, const std::vector<T>& data)
{
    writeNpy<T, 1>(*::detail::openForWriting(path), data, { data.size() });
}
