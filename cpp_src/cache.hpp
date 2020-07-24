#pragma once

#include <vector>
#include <unordered_map>
#include <ostream>

using namespace std;

#include "utils.h"

class Cache
{
private:
    //缓存内容
    ElementVector entries;
    //缓存内容集合，用于检测缓存命中与否
    unordered_map<ElementType, size_t> pos_map;
    unordered_map<ElementType, size_t> freq_map;

    //buf
    FloatVector freq_buf;

    inline void check_idx(size_t idx)
    {
        ASSERT(idx < this->capacity() && "Index is out of the capacity of cache!");
    }

public:
    explicit Cache(int capacity)
    {
        entries.resize(capacity);
        for (auto &e : entries) {
            e = NoneType;
        }
    }

    void reset()
    {
        if (DEBUG) {
            cout << "Cache reset." << endl;
        }

        for (auto &e : entries) {
            e = NoneType;
        }
        pos_map.clear();
        freq_map.clear();
    }

    //获取所有缓存内容
    inline ElementVector *get_contents()
    {
        return &entries;
    }

    //获取某一个元素的频率
    inline float get_frequency(ElementType e)
    {
        float freq = 0;
        auto it = this->freq_map.find(e);
        if (it != this->freq_map.end()) {
            freq = it->second;
        }
        return freq;
    }

    //获取每个内容的命中次数
    inline FloatVector *get_frequencies(const ElementVector *elements)
    {
        freq_buf.resize(0);

        for (auto &e: *elements) {
            auto freq = this->get_frequency(e);
            freq_buf.push_back(freq);
        }

        return &freq_buf;
    }


    //清楚统计的内容频率
    inline void clear_frequencies()
    {
        this->freq_map.clear();
    }

    //缓存中内容数量
    inline size_t size()
    {
        return this->pos_map.size();
    }

    //缓存的最大容量
    inline size_t capacity()
    {
        return this->entries.size();
    }

    //缓存是否已满
    inline bool full()
    {
        return this->size() >= this->capacity();
    }

    //获取缓存某一位置上的内容
    inline ElementType get(size_t idx)
    {
        this->check_idx(idx);
        return this->entries[idx];
    }

    //将内容放置在某处
    inline void set(size_t idx, ElementType e)
    {
        ASSERT((this->pos_map.find(e) == this->pos_map.end()) && "Error: content is already in the cache!");

        this->check_idx(idx);
        auto e_old = this->entries[idx];
        auto it = this->pos_map.find(e_old);
        if (it != this->pos_map.end()) {
            this->pos_map.erase(it);
        }

        this->entries[idx] = e;
        this->pos_map[e] = idx;
    }

    //根据元素查找它在缓存中的位置，-1表示该元素不在缓存中
    inline int find(ElementType e)
    {
        auto it = this->pos_map.find(e);
        if (it != this->pos_map.end()) {
            return it->second;
        }
        else {
            return -1;
        }
    }

    //检测内容是否在缓存中，同时更新内容频率
    inline bool hit_test(ElementType e)
    {
        if (this->freq_map.find(e) == this->freq_map.end()) {
            freq_map[e] = 0;
        }
        freq_map[e]++;

        auto idx = this->find(e);
        return idx != -1;
    }

    //使用新的内容替换老的内容
    inline void replace(ElementType e_new, ElementType e_old)
    {
        if (VERBOSE) {
            cout << "cache.replace: " << e_new << ", " << e_old << endl;
        }

        int idx;
        if (e_old == NoneType)
            idx = this->size();
        else
            idx = this->find(e_old);

        ASSERT(idx != -1 && "Can't find origin element!");

        this->set(idx, e_new);
    }

    friend ostream &operator<<(ostream &os, Cache &cache)
    {
        for (auto &e: cache.entries) {
            auto freq = cache.get_frequency(e);
            os << e << " " << freq << endl;
        }
        return os;
    }
};

