#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <unordered_map>

#include "utils.h"
#include "request.hpp"

using namespace std;

/**
 * 用于保存特征
 */
struct Feature
{
    FeatureType *data = nullptr;
    size_t content_dims = 0;
    size_t feature_dims = 0;

    Feature() = default;

    Feature(FeatureType *data, size_t contentDims, size_t featureDims)
            : data(data), content_dims(contentDims), feature_dims(featureDims) {}

    inline FeatureType get(size_t c_dim, size_t f_dim) const
    {
        return data[c_dim * feature_dims + f_dim];
    }

    inline void set(size_t c_dim, size_t f_dim, FeatureType f)
    {
        data[c_dim * feature_dims + f_dim] = f;
    }

    inline std::pair<size_t, size_t> shape()
    {
        return {content_dims, feature_dims};
    }

    friend ostream &operator<<(ostream &os, Feature &feature)
    {
        os << "[";
        for (size_t i = 0; i < feature.content_dims; ++i) {
            os << "[";
            for (size_t j = 0; j < feature.feature_dims; ++j) {
                os << feature.get(i, j) << " ";
            }
            os << "]" << endl;
        }
        os << "]" << endl;
        return os;
    }

    FloatBuffer to_buffer()
    {
        FloatBuffer buf{};
        buf.data = data;
        buf.size = content_dims * feature_dims;
        return buf;
    }
};

class FeatureExtractor
{
protected:
    size_t feature_dims;
    vector<FeatureType> f_buf;

public:
    explicit FeatureExtractor(size_t _feature_dims) : feature_dims(_feature_dims) {}

    inline size_t get_feature_dims()
    {
        return this->feature_dims;
    }

    virtual void reset() = 0;

    virtual void update(const Slice &s) = 0;

    virtual Feature get_features(ContentVector &v) = 0;
};

class IdFeatureExtractor : public FeatureExtractor
{
public:
    IdFeatureExtractor() : FeatureExtractor(1) {}

    void reset() override
    {
        if (VERBOSE) {
            cout << "IdFeatureExtractor reset." << endl;
        }
    }

    void update(const Slice &s) override {}

    Feature get_features(ContentVector &v) override
    {
        f_buf.resize(v.size() * this->feature_dims);

        for (size_t i = 0; i < v.size(); i++) {
            f_buf[i] = v[i];
        }

        return {f_buf.data(), v.size(), feature_dims};
    }
};

class LruFeatureExtractor : public FeatureExtractor
{
private:
    vector<TimestampType> W;  //用于每个内容最后访问的时间

    TimestampType latest_time = -1;

public:
    LruFeatureExtractor() : FeatureExtractor(1), W(MAX_CONTENTS, -1) {}

    void reset() override
    {
        if (VERBOSE) {
            cout << "LruFeatureExtractor reset." << endl;
        }
        latest_time = -1;
        for (int i = 0; i < MAX_CONTENTS; i++) {
            W[i] = -1;
        }
    }

    void update(const Slice &s) override
    {
        for (size_t i = 0; i < s.size; i++) {
            auto cid = s.data[i].content_id;
            auto t = s.data[i].timestamp;
            this->W[cid] = t;
        }
        this->latest_time = s.data[s.size - 1].timestamp;
    }

    Feature get_features(ContentVector &v) override
    {
        f_buf.resize(v.size() * this->feature_dims);

        for (size_t i = 0; i < v.size(); i++) {
            //这里添加符号是为了让lru特征的顺序和lfu一致
            f_buf[i] = -(latest_time - W[v[i]]);
        }

        return {f_buf.data(), v.size(), feature_dims};
    }
};

class LfuFeatureExtractor : public FeatureExtractor
{
private:
    IntVector W;  //用于每个内容被访问的次数

public:
    LfuFeatureExtractor() : FeatureExtractor(1), W(MAX_CONTENTS, 0) {}

    void reset() override
    {
        if (VERBOSE) {
            cout << "LfuFeatureExtractor reset." << endl;
        }
        for (int i = 0; i < MAX_CONTENTS; i++) {
            W[i] = 0;
        }
    }

    void update(const Slice &s) override
    {
        for (size_t i = 0; i < s.size; i++) {
            auto cid = s.data[i].content_id;
            this->W[cid]++;
        }
    }

    Feature get_features(ContentVector &v) override
    {
        f_buf.resize(v.size() * this->feature_dims);

        for (size_t i = 0; i < v.size(); i++) {
            f_buf[i] = W[v[i]];
        }

        return {f_buf.data(), v.size(), feature_dims};
    }
};

class SWLfuFeatureExtractor : public FeatureExtractor
{
private:
    IntVector W;  //用于每个内容被访问的次数
    int history_w_len, history_num_requests;
    int i_slice = 0;
    RequestLoader *loader;

private:

    inline void deque_expired_histories(TimestampType curr_timestamp)
    {
        auto curr_i_slice = this->loader->get_i_slice_by_timestamp(curr_timestamp);
        if (curr_i_slice != this->i_slice && curr_i_slice > history_w_len) {
            auto i_slice_beg = std::max(this->i_slice - history_w_len, 0);
            auto i_slice_end = std::max(curr_i_slice - history_w_len, 0);

            for (; i_slice_beg < i_slice_end; i_slice_beg++) {
                auto range_ptr = loader->get_slice_range_ptrs(i_slice_beg);
                Slice history_slice = loader->get_slice(range_ptr.first, range_ptr.second);

                for (size_t i = 0; i < history_slice.size; i++) {
                    auto cid = history_slice.data[i].content_id;
                    this->W[cid]--;
                }
                this->history_num_requests -= history_slice.size;
            }

            this->i_slice = curr_i_slice;
        }
    }

public:
    SWLfuFeatureExtractor(int history_w_len, RequestLoader *loader)
            : FeatureExtractor(1), W(MAX_CONTENTS, 0)
    {
        this->history_w_len = history_w_len;
        this->loader = loader;
        this->i_slice = 0;
        this->history_num_requests = 0;
    }

    void reset() override
    {
        if (VERBOSE) {
            cout << "SWLfuFeatureExtractor reset." << endl;
        }
        this->i_slice = 0;
        this->history_num_requests = 0;
        for (int i = 0; i < MAX_CONTENTS; i++) {
            W[i] = 0;
        }
    }

    inline void update(const Slice &s) override
    {
        const bool flag = true;
        if (flag) {
            update1(s);
        }
        else {
            update2(s);
        }
    }

    inline void update1(const Slice &s)
    {
        //更新参数
        for (size_t i = 0; i < s.size; i++) {
            auto r = s.data[i];
            this->W[r.content_id]++;
        }
        this->history_num_requests += s.size;

        if (s.size > 0) {
            this->deque_expired_histories(s.get(-1).timestamp);
        }
    }

    inline void update2(const Slice &s)
    {
        for (size_t i = 0; i < s.size; i++) {
            auto cid = s.data[i].content_id;
            this->W[cid]++;
        }
        this->history_num_requests += s.size;

        if (i_slice >= history_w_len) {
            auto range_ptr = loader->get_slice_range_ptrs(i_slice - history_w_len);
            Slice history_slice = loader->get_slice(range_ptr.first, range_ptr.second);

            for (size_t i = 0; i < history_slice.size; i++) {
                auto cid = history_slice.data[i].content_id;
                this->W[cid]--;
            }
            this->history_num_requests -= history_slice.size;
        }
        i_slice++;
    }

    Feature get_features(ContentVector &v) override
    {
        f_buf.resize(v.size() * this->feature_dims);

        for (size_t i = 0; i < v.size(); i++) {
            f_buf[i] = (float) W[v[i]] / (history_num_requests + EPS);
        }

        return {f_buf.data(), v.size(), feature_dims};
    }
};


class OgdFeatureExtractor : public FeatureExtractor
{
private:
    unordered_map<ContentType, float *> W;      //根据内容快速找到内容特征
    vector<pair<ContentType, float> *> W_heap;  //用于保存内容的特征，并使用最小堆维护特征的顺序
    float W_sum = 0;

    size_t max_w_len = 0;

    static bool cmp(pair<ContentType, float> *a, pair<ContentType, float> *b)
    {
        return a->second > b->second;
    }

    virtual float get_eta() = 0;

    inline void delete_expired_elements(float eta)
    {
        //为了避免W无限增长，当W的大小超过最大的长度后，将特征值最小的元素从W中剔除
        float w_deleted = 0;  //用于保存被踢出去的元素的特征值
        while (W.size() > max_w_len) {
            auto min_e = W_heap.front();  //获取w最小的元素
            W.erase(min_e->first);        //将其从W中移除

            w_deleted += (min_e->second); //将其特征值加到w_deleted

            //保持堆的结构
            pop_heap(W_heap.begin(), W_heap.end(), cmp);
            W_heap.pop_back();

            //释放内存
            delete min_e;
        }

        //接下来做归一化
        float W_sum_new = 0;
        float denominator = (W_sum + eta - w_deleted);
        for (auto &iter: W) {
            *(iter.second) /= denominator;
            W_sum_new += *(iter.second);
        }
        W_sum = W_sum_new;
    }

protected:
    int count = 0;  //步计数

public:
    explicit OgdFeatureExtractor(size_t capacity) : FeatureExtractor(1)
    {
        max_w_len = capacity * 100;
        make_heap(W_heap.begin(), W_heap.end(), cmp);
    }

    void reset() override
    {
        if (VERBOSE) {
            cout << "OgdFeatureExtractor reset." << endl;
        }

        count = 0;

        for (auto w: W_heap) {
            delete w;
        }
        W.clear();
        W_heap.clear();
    }

    inline void update_single_request(Request r)
    {
        float eta = get_eta();  //OgdOpt、LFU、LRU三种的eta的计算方式不同
        auto cid = r.content_id;

        auto it = W.find(cid);
        if (it == W.end()) {
            //如果元素之前没有存储，那么创建一个新的键值对
            auto pair_ptr = new pair<ContentType, float>(cid, eta);
            //W中保存特征的指针，保证W和W_heap中的特征值的一致
            W[cid] = &(pair_ptr->second);

            //将键值对的指针保存到最小堆中，更新最小堆
            W_heap.push_back(pair_ptr);
            push_heap(W_heap.begin(), W_heap.end(), cmp);
        }
        else {
            //如果元素存储过，那么加上eta，更新最小堆
            (*it->second) += eta;
            make_heap(W_heap.begin(), W_heap.end(), cmp);
        }

        this->delete_expired_elements(eta);

        //计数增加
        this->count++;
    }

    void update(const Slice &s) override
    {
        bool use_batch_process = true;
        if (use_batch_process) {
            //方法一：批量处理请求
            float eta = get_eta();  //OgdOpt、LFU、LRU三种的eta的计算方式不同

            for (size_t i = 0; i < s.size; i++) {
                auto cid = s.data[i].content_id;

                auto it = W.find(cid);
                if (it == W.end()) {
                    //如果元素之前没有存储，那么创建一个新的键值对
                    auto pair_ptr = new pair<ContentType, float>(cid, eta);
                    //W中保存特征的指针，保证W和W_heap中的特征值的一致
                    W[cid] = &(pair_ptr->second);

                    //将键值对的指针保存到最小堆中，更新最小堆
                    W_heap.push_back(pair_ptr);
                    push_heap(W_heap.begin(), W_heap.end(), cmp);
                }
                else {
                    //如果元素存储过，那么加上eta，更新最小堆
                    (*it->second) += eta;
                    make_heap(W_heap.begin(), W_heap.end(), cmp);
                }
            }

            this->delete_expired_elements(eta);

            //计数增加
            this->count++;
        }
        else {
            //方法二：逐个处理请求
            for (size_t i = 0; i < s.size; i++) {
                auto r = s.data[i];
                this->update_single_request(r);
            }
        }
    }

    Feature get_features(ContentVector &v) override
    {
        f_buf.resize(v.size() * this->feature_dims);

        for (size_t i = 0; i < v.size(); i++) {
            auto it = W.find(v[i]);
            if (it == W.end()) {
                f_buf[i] = 0;
            }
            else {
                f_buf[i] = *(it->second);
            }
        }

        return {f_buf.data(), v.size(), feature_dims};
    }
};

class OgdOptimalFeatureExtractor : public OgdFeatureExtractor
{
protected:

    inline float get_eta() override
    {
        return 1.0 / sqrt(this->count + 1);
    }

public:
    explicit OgdOptimalFeatureExtractor(size_t capacity) : OgdFeatureExtractor(capacity) {}
};

class OgdLruFeatureExtractor : public OgdFeatureExtractor
{
protected:

    inline float get_eta() override
    {
        return 1.0;
    }

public:
    explicit OgdLruFeatureExtractor(size_t capacity) : OgdFeatureExtractor(capacity) {}
};

class OgdLfuFeatureExtractor : public OgdFeatureExtractor
{
protected:
    inline float get_eta() override
    {
        return 1.0 / (this->count + 1);
    }

public:
    explicit OgdLfuFeatureExtractor(size_t capacity) : OgdFeatureExtractor(capacity) {}
};

class FeatureManager
{
private:
    vector<FeatureExtractor *> extractors;
    vector<FeatureType> f_buf;

public:
    size_t feature_dims{0};

    FeatureManager() = default;

    virtual ~FeatureManager()
    {
        delete[] extractors.data();
    }

    void reset()
    {
        if (VERBOSE) {
            cout << "FeatureManager reset." << endl;
        }
        for (auto e: extractors) {
            e->reset();
        }
    }

    void add_feature_extractor(FeatureExtractor *extractor)
    {
        this->extractors.push_back(extractor);
        this->feature_dims += extractor->get_feature_dims();
    }

    void update(const Slice &s)
    {
        for (auto &e: this->extractors) {
            e->update(s);
        }
    }

    inline Feature get_features(ContentVector &v)
    {
        auto content_dims = v.size();
        f_buf.resize(content_dims * this->feature_dims);
        Feature features(f_buf.data(), content_dims, this->feature_dims);

        size_t f_dims = 0;
        for (auto &e: this->extractors) {
            auto f = e->get_features(v);
            for (size_t j = 0; j < content_dims; ++j) {
                for (size_t i = 0; i < f.feature_dims; ++i) {
                    features.set(j, f_dims + i, f.get(j, i));
                }
            }
            f_dims += f.feature_dims;
        }

        return features;
    }
};

