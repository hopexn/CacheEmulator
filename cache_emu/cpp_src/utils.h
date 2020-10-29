#pragma once

//C headers
#include <cmath>
#include <cassert>

//C++ headers
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <string>
#include <sstream>
#include <unordered_map>

//缓存元素类型
typedef int32_t ContentType;
typedef int32_t TimestampType;
typedef float FeatureType;

//NoneType记为-1
const ContentType NoneContentType = -1;
const TimestampType NoneTimestampType = -1;

#define EPS 1e-6            //精度
#define MAX_CONTENTS 1e7    //最大ID值

struct Request
{
    ContentType content_id = NoneContentType;
    TimestampType timestamp = 0;
};

struct Triple
{
    size_t first, second, third;
};

typedef std::vector<ContentType> ContentVector;
typedef std::set<ContentType> ContentSet;
typedef std::unordered_map<ContentType, size_t> ContentFreqMap;
typedef std::vector<float> FloatVector;
typedef std::vector<int32_t> IntVector;


template<typename T>
inline void copy_to_std_vector(T *data, size_t size, std::vector<T> &v)
{
    for (size_t i = 0; i < size; ++i) {
        v.push_back(data[i]);
    }
}

template<typename T>
inline void copy_to_std_set(T *data, size_t size, std::set<T> &v)
{
    for (size_t i = 0; i < size; ++i) {
        v.insert(data[i]);
    }
}

template<typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> v)
{
    os << "[";
    for (auto &e: v) { os << e << ' '; }
    os << "]";
    return os;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, std::set<T> v)
{
    os << "(";
    for (auto &e: v) { os << e << ' '; }
    os << ")";
    return os;
}

/**
 * DEBUG: 调试等级
 * 0: 不启用调试
 * 1: 启用assert
 * 2：启用verbose模式
 */
#define DEBUG 0
#define ASSERT(bool_expr) if(DEBUG > 0){assert(bool_expr);}
#define VERBOSE (DEBUG > 1)
