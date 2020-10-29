#include <iostream>
#include <set>

using namespace std;

#include "utils.h"
#include "cache.hpp"
#include "request.hpp"
#include "feature.hpp"

class CacheEmu
{
protected:
    int capacity;

    Cache cache;
    FeatureManager feature_manager;
    RequestLoader *loader = nullptr;

    //用于记录已经处理的请求数以及其中命中的次数
    int request_cnt = 0, hit_cnt = 0;
    int episode_request_cnt = 0, episode_hit_cnt = 0;
    FloatVector episode_hit_rates;

    //当前步数以及当前回合
    int i_slice = 0, i_episode = 0;

    //用于保存待替换与替换目标内容
    ContentSet s_buf_old, s_buf_new;

    //缓冲区，用于保存用于返回的结果
    ContentVector step_buf;
    ContentVector candidate_buf;
    FloatVector candidate_frequency_buf;

public:
    explicit CacheEmu(int capacity, RequestLoader *loader)
            : cache(capacity), feature_manager()
    {
        this->capacity = capacity;
        this->loader = loader;
    }

    void reset()
    {
        if (DEBUG) {
            cout << "CacheEmu reset." << endl;
        }

        i_slice = 0;
        i_episode = 0;

        request_cnt = 0;
        hit_cnt = 0;
        episode_request_cnt = 0;
        episode_hit_cnt = 0;

        this->cache.reset();
        this->feature_manager.reset();

        candidate_buf.resize(0);
        for (auto e: *cache.get_contents()) {
            candidate_buf.push_back(e);
        }

        candidate_frequency_buf.resize(0);
        for (auto e: *this->cache.get_frequencies(&candidate_buf)) {
            candidate_frequency_buf.push_back(e);
        }
        this->cache.clear_frequencies();
    }


    //使用id特征
    void use_id_feature()
    {
        this->feature_manager.add_feature_extractor(new IdFeatureExtractor());
    }

    //使用lfu特征
    void use_lfu_feature()
    {
        //this->feature_manager.add_feature_extractor(new LfuFeatureExtractor());
        this->feature_manager.add_feature_extractor(new OgdLfuFeatureExtractor(this->capacity));
    }

    //使用lru特征
    void use_lru_feature()
    {
        //this->feature_manager.add_feature_extractor(new LruFeatureExtractor());
        this->feature_manager.add_feature_extractor(new OgdLruFeatureExtractor(this->capacity));
    }

    //使用ogd_optimal特征
    void use_ogd_opt_feature()
    {
        this->feature_manager.add_feature_extractor(new OgdOptimalFeatureExtractor(this->capacity));
    }

    //使用带滑动窗口的LFU特征
    void use_swlfu_feature(size_t history_sw_len)
    {
        this->feature_manager.add_feature_extractor(new SWLfuFeatureExtractor(history_sw_len, this->loader));
    }

    //返回特征维度大小
    size_t feature_dims()
    {
        return this->feature_manager.feature_dims;
    }

    //获取特征
    inline Feature get_features(ContentVector &v)
    {
        return this->feature_manager.get_features(v);
    }

    //更新缓存内容
    inline void update_cache(ContentType *es, size_t size)
    {
        s_buf_old.clear();
        s_buf_new.clear();

        auto cache_contents = this->cache.get_contents();
        copy_to_std_set(cache_contents->data(), size, s_buf_old);
        copy_to_std_set(es, size, s_buf_new);

        s_buf_old.erase(NoneContentType);
        s_buf_new.erase(NoneContentType);

        for (size_t i = 0; i < size; i++) {
            if (es[i] != NoneContentType && this->cache.find(es[i]) != -1) {
                s_buf_old.erase(es[i]);
                s_buf_new.erase(es[i]);
            }
        }

        if (VERBOSE) {
            cout << "update_cache: " << s_buf_old << "," << s_buf_new << endl;
        }

        auto it1 = s_buf_old.begin();
        auto it2 = s_buf_new.begin();
        while (it1 != s_buf_old.end() && it2 != s_buf_new.end()) {
            cache.replace(*it2, *it1);
            it1++;
            it2++;
        }

        while (it2 != s_buf_new.end()) {
            cache.replace(*it2, NoneContentType);
            it2++;
        }
    }

    //获取总时间片数
    inline std::size_t get_num_slices()
    {
        return this->loader->get_num_slices();
    }

    //获取当前时间片
    inline std::size_t get_i_slice() const
    {
        return this->i_slice;
    }

    //获取平均命中率
    inline float get_mean_hit_rate()
    {
        float mean_hit_rate = (float) hit_cnt / (request_cnt + EPS);
        return mean_hit_rate;
    }

    //是否已处理完所有请求
    inline bool finished()
    {
        return this->i_slice >= this->loader->get_num_slices();
    }

    //获取当前的回合数
    inline size_t get_i_episode()
    {
        return this->i_episode;
    }

    //获取当前缓存中的内容
    ContentVector *get_cache_contents()
    {
        auto contents = this->cache.get_contents();
        if (VERBOSE) {
            cout << "cache_contents: " << *contents << endl;
        }
        return contents;
    }

    //获取候选内容：当前缓存内容+发生miss的内容
    ContentVector *get_candidates()
    {
        if (VERBOSE) {
            cout << "candidates: " << candidate_buf << endl;
        }
        return &candidate_buf;
    }

    //获取候选目标的频率
    inline FloatVector *get_candidate_frequencies()
    {
        if (VERBOSE) {
            cout << "candidate_frequencies: " << candidate_frequency_buf << endl;
        }
        return &candidate_frequency_buf;
    }

    //获取当前步处理的内容
    inline ContentVector *get_step_elements()
    {
        return &step_buf;
    }

    //回合结束时执行的操作
    float on_episode_end()
    {
        auto episode_hit_rate = (float) episode_hit_cnt / (episode_request_cnt + EPS);
        episode_hit_rates.push_back(episode_hit_rate);

        //回合计数器清零
        episode_request_cnt = 0;
        episode_hit_cnt = 0;

        if (VERBOSE) {
            cout << "Episode " << i_episode << ":\t" << episode_hit_rate << ",\t" << get_mean_hit_rate() << endl;
        }

        this->i_episode++;

        return episode_hit_rate;
    }

    //处理一批请求, 返回发生miss的次数
    virtual Triple step() = 0;
};


class ActiveCacheEmu : public CacheEmu
{
private:
    ContentSet missed_content_set;

public:
    ActiveCacheEmu(int capacity, RequestLoader *loader) : CacheEmu(capacity, loader) {}

    Triple step() override
    {
        missed_content_set.clear();
        step_buf.resize(0);

        auto slice_range_ptrs = loader->get_slice_range_ptrs(this->i_slice);
        auto slice = loader->get_slice(slice_range_ptrs.first, slice_range_ptrs.second);
        this->i_slice++;  //步计数增一

        if (VERBOSE) {
            cout << "step " << i_slice << ":" << slice << endl;
        }

        for (size_t i = 0; i < slice.size; i++) {
            auto r = slice.data[i];
            step_buf.push_back(r.content_id);

            auto hit = cache.hit_test(r.content_id);
            this->hit_cnt += hit;
            this->episode_hit_cnt += hit;

            if (!hit) {
                missed_content_set.insert(r.content_id);
            }
        }
        this->request_cnt += slice.size;
        this->episode_request_cnt += slice.size;

        this->feature_manager.update(slice);

        //生成candidates及其对应的频率
        candidate_buf.resize(0);

        for (auto e: *cache.get_contents()) {
            candidate_buf.push_back(e);
        }

        for (auto e: missed_content_set) {
            candidate_buf.push_back(e);
        }

        candidate_frequency_buf.resize(0);
        for (auto e: *this->cache.get_frequencies(&candidate_buf)) {
            candidate_frequency_buf.push_back(e);
        }
        this->cache.clear_frequencies();

        return {slice.size, missed_content_set.size(), 0};
    }
};

class PassiveCacheEmu : public CacheEmu
{
private:
    Slice slice, slice_processed;

public:
    PassiveCacheEmu(int capacity, RequestLoader *loader) : CacheEmu(capacity, loader) {}

    Triple step() override
    {
        step_buf.resize(0);
        ContentType missed_element = NoneContentType;

        if (slice.size == 0) {
            auto slice_range_ptrs = loader->get_slice_range_ptrs(this->i_slice);
            this->slice = loader->get_slice(slice_range_ptrs.first, slice_range_ptrs.second);
            this->i_slice++;
        }

        if (VERBOSE) {
            cout << "Slice " << this->i_slice << ": " << slice << endl;
        }

        int idx = 0;
        while (idx < slice.size) {
            auto r = slice.data[idx];
            step_buf.push_back(r.content_id);

            auto hit = cache.hit_test(r.content_id);
            this->hit_cnt += hit;
            this->episode_hit_cnt += hit;

            idx++;

            if (!hit) {
                missed_element = r.content_id;
                break;
            }
        }
        this->slice_processed = this->slice.sub_slice(0, idx);
        this->slice = this->slice.sub_slice(idx, -1);

        this->request_cnt += this->slice_processed.size;
        this->episode_request_cnt += this->slice_processed.size;
        this->feature_manager.update(this->slice_processed);

        //生成candidates及其对应的频率
        candidate_buf.resize(0);
        for (auto &e: *cache.get_contents()) {
            candidate_buf.push_back(e);
        }

        if (missed_element != NoneContentType) {
            candidate_buf.push_back(missed_element);
        }

        candidate_frequency_buf.resize(0);
        for (auto &e: *this->cache.get_frequencies(&candidate_buf)) {
            candidate_frequency_buf.push_back(e);
        }
        this->cache.clear_frequencies();
        while (candidate_frequency_buf.size() < this->capacity + 1) {
            candidate_frequency_buf.push_back(0);
        }

        return {slice_processed.size, missed_element != NoneContentType, slice.size};
    }
};