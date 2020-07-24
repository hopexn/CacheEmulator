#include "apis.h"
#include "cache_emu.hpp"

RequestLoader loader;
vector<CacheEmu *> cache_emus;

void load_dataset(ElementType *cs, TimestampType *ts, size_t size)
{
    loader.load_dataset(cs, ts, size);
}

int slice_dataset_by_time(TimestampType t_beg, TimestampType t_end, TimestampType interval)
{
    return loader.slice_by_time(t_beg, t_end, interval);
}

int init_cache_emu(int capacity, bool passive_mode)
{
    auto handler = cache_emus.size();

    cout << "Emu " << handler << ": ";

    if (passive_mode) {
        cache_emus.push_back(new PassiveCacheEmu(capacity, &loader));
        cout << "passive mode.";
    }
    else {
        cache_emus.push_back(new ActiveCacheEmu(capacity, &loader));
        cout << "active mode.";
    }

    cout << endl;

    return handler;
}

void reset(int handler)
{
    if (DEBUG) {
        cout << "Emu " << handler << " reset." << endl;
    }
    cache_emus[handler]->reset();
}

Triple step(int handler)
{
    auto res = cache_emus[handler]->step();
    return res;
}

IntBuffer get_cache_contents(int handler)
{
    auto v = cache_emus[handler]->get_cache_contents();
    return from_std_vector(*v);
}

IntBuffer get_candidates(int handler)
{
    auto v = cache_emus[handler]->get_candidates();
    return from_std_vector(*v);
}

FloatBuffer get_candidate_frequencies(int handler)
{
    auto freqs = cache_emus[handler]->get_candidate_frequencies();
    return from_std_vector(*freqs);
}

IntBuffer get_step_elements(int handler)
{
    auto step_es = cache_emus[handler]->get_step_elements();
    return from_std_vector(*step_es);
}

int get_num_step_elements(int handler)
{
    auto step_es = cache_emus[handler]->get_step_elements();
    return (int) (step_es->size());
}

void update_cache(int handler, IntBuffer v)
{
    if (VERBOSE) {
        cout << "emu[" << handler << "].new_contents: " << buffer_to_string(v) << endl;
    }
    cache_emus[handler]->update_cache((ElementType *) v.data, v.size);
}

void setup_traditional_feature_types(int handler, bool use_lfu_feature, bool use_lru_feature, bool use_ogd_opt_feature)
{
    if (use_lfu_feature) {
        cache_emus[handler]->use_lfu_feature();
    }

    if (use_lru_feature) {
        cache_emus[handler]->use_lru_feature();
    }

    if (use_ogd_opt_feature) {
        cache_emus[handler]->use_ogd_opt_feature();
    }
}

//使用带窗口衰减的LFU特征
void setup_swlfu_feature_types(int handler, int *w_lens, size_t size)
{
    for (int i = 0; i < size; ++i) {
        cache_emus[handler]->use_swlfu_feature(w_lens[i]);
    }
}

//获取所有特征
FloatBuffer get_features(int handler, ElementType *es, size_t size)
{
    ElementVector buf_e;
    copy_to_std_vector(es, size, buf_e);

    auto features = cache_emus[handler]->get_features(buf_e);

    if (VERBOSE) {
        cout << "emu[" << handler << "].get_features: " << features << endl;
    }

    return features.to_buffer();
}

int finished(int handler)
{
    return cache_emus[handler]->finished();
}

float get_mean_hit_rate(int handler)
{
    return cache_emus[handler]->get_mean_hit_rate();
}

int get_i_episode(int handler)
{
    return cache_emus[handler]->get_i_episode();
}

float on_episode_end(int handler)
{
    return cache_emus[handler]->on_episode_end();
}

size_t feature_dims(int handler)
{
    return cache_emus[handler]->feature_dims();
}
