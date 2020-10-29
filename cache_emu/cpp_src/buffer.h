#ifndef BUFFER_H
#define BUFFER_H

#include <vector>

//Buffer定义，用于C与Python之间的信息传递
struct IntBuffer
{
    int32_t *data;
    size_t size;
};


struct FloatBuffer
{
    float *data;
    size_t size;
};

inline
IntBuffer from_std_vector(std::vector<int> &v)
{
    IntBuffer buf{};
    buf.data = v.data();
    buf.size = v.size();
    return buf;
}

inline
FloatBuffer from_std_vector(std::vector<float> &v)
{
    FloatBuffer buf{};
    buf.data = v.data();
    buf.size = v.size();
    return buf;
}

inline
IntBuffer from_memory(int32_t *data, size_t size)
{
    IntBuffer buf{};
    buf.data = data;
    buf.size = size;
    return buf;
}

inline
FloatBuffer from_memory(float *data, size_t size)
{
    FloatBuffer buf{};
    buf.data = data;
    buf.size = size;
    return buf;
}

inline
void copy_to_std_vector(IntBuffer &buf, std::vector<int> &v)
{
    v.resize(buf.size);

    for (int i = 0; i < buf.size; ++i) {
        v[i] = buf.data[i];
    }
}

inline
void copy_to_std_vector(FloatBuffer &buf, std::vector<float> &v)
{
    v.resize(buf.size);

    for (int i = 0; i < buf.size; ++i) {
        v[i] = buf.data[i];
    }
}


template<typename T>
inline std::string buffer_to_string(const T &buf)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < buf.size; i++) {
        oss << buf.data[i] << " ";
    }
    oss << "]";
    return oss.str();
}

#endif
