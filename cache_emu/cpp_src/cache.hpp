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
    ContentVector contents;
    //保存返回的频率值
    FloatVector freq_ret;

    //缓存内容集合，用于检测缓存命中与否
    unordered_map<ContentType, size_t> pos_map;
    unordered_map<ContentType, size_t> freq_map;

    inline void check_idx(size_t idx)
    {
        ASSERT(idx < this->capacity() && "Index is out of the capacity of cache!");
    }

public:
    explicit Cache(size_t _capacity)
            : contents(_capacity, NoneContentType), freq_ret(_capacity, 0) {}

    void reset()
    {
        if (VERBOSE) {
            cout << "Cache reset." << endl;
        }

        for (auto &e : contents) {
            e = NoneContentType;
        }
        pos_map.clear();
        freq_map.clear();
    }

    //获取所有缓存内容
    inline ContentVector *get_contents()
    {
        return &contents;
    }

    //获取某一个元素的频率
    inline float get_frequency(ContentType e)
    {
        float freq = 0;
        auto it = this->freq_map.find(e);
        if (it != this->freq_map.end()) {
            freq = it->second;
        }
        return freq;
    }

    //获取每个内容的命中次数
    inline FloatVector *get_frequencies(const ContentVector *elements)
    {
        freq_ret.resize(0);

        for (auto &e: *elements) {
            auto freq = this->get_frequency(e);
            freq_ret.push_back(freq);
        }

        return &freq_ret;
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
        return this->contents.size();
    }

    //缓存是否已满
    inline bool full()
    {
        return this->size() >= this->capacity();
    }

    //获取缓存某一位置上的内容
    inline ContentType get(size_t idx)
    {
        this->check_idx(idx);
        return this->contents[idx];
    }

    //将内容放置在某处
    inline void set(size_t idx, ContentType e)
    {
        ASSERT((this->pos_map.find(e) == this->pos_map.end()) && "Error: content is already in the cache!");

        this->check_idx(idx);
        auto e_old = this->contents[idx];
        auto it = this->pos_map.find(e_old);
        if (it != this->pos_map.end()) {
            this->pos_map.erase(it);
        }

        this->contents[idx] = e;
        this->pos_map[e] = idx;
    }

    //根据元素查找它在缓存中的位置，-1表示该元素不在缓存中
    inline int find(ContentType e)
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
    inline bool hit_test(ContentType e)
    {
        freq_map[e]++;
        auto idx = this->find(e);
        return idx != -1;
    }

    //使用新的内容替换老的内容
    inline void replace(ContentType e_new, ContentType e_old)
    {
        if (VERBOSE) {
            cout << "cache.replace: " << e_new << ", " << e_old << endl;
        }

        int idx;
        if (e_old == NoneContentType)
            idx = this->size();
        else
            idx = this->find(e_old);

        ASSERT(idx != -1 && "Can't find origin content!");

        this->set(idx, e_new);
    }
};
