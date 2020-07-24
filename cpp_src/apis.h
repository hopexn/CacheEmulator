#ifndef APIS_H
#define APIS_H

#include "utils.h"
#include "buffer.h"

extern "C" {

/**
 * 加载请求序列数据集
 * @param cs 请求的内容
 * @param ts 请求发生的时间
 * @param size 请求的长度
 */
void load_dataset(ElementType *cs, TimestampType *ts, size_t size);

/**
 * 将请求序列根据时间进行分片
 * @param t_beg 起始时间
 * @param t_end 终止时间
 * @param interval  间隔时间
 * @return      时间片的个数
 */
int slice_dataset_by_time(TimestampType t_beg, TimestampType t_end, TimestampType interval);

/**
 * 初始化一个缓存模拟器
 * @param capacity 缓存容量
 * @return  缓存模拟器句柄
 */
int init_cache_emu(int capacity, bool passive_mode);

/**
 * 重置模拟器
 * @param handler
 */
void reset(int handler);

/**
 * 处理一批请求
 * @param handler 缓存模拟器句柄
 * @return
 */
Triple step(int handler);

/**
 * 获取当前缓存内容
 * @param handler 缓存模拟器句柄
 * @return  当前缓存内容
 */
IntBuffer get_cache_contents(int handler);

/**
 * 获取当前候选内容
 * @param handler 缓存模拟器句柄
 * @return  当前候选内容
 */
IntBuffer get_candidates(int handler);

/**
 * 获取缓存中每一个元素在这一回合中的命中次数
 * @param handler 缓存模拟器句柄
 * @return  缓存中每一个元素在这一回合中的命中次数
 */
FloatBuffer get_candidate_frequencies(int handler);


/**
 * 获取当前步处理的所有请求元素
 * @param handler
 * @return
 */
IntBuffer get_step_elements(int handler);

/**
 * 获取当前步处理的所有请求元素个数
 * @param handler
 * @return
 */
int get_num_step_elements(int handler);


/**
 * 更新缓存内容
 * @param handler   缓存模拟器句柄
 * @param v         接下来缓存存储的内容
 */
void update_cache(int handler, IntBuffer v);

/**
 * 设置传统特征类型
 * @param handler           缓存模拟器句柄
 * @param use_lfu_feature   使用LFU特征
 * @param use_lru_feature   使用LRU特征
 * @param use_ogd_opt_feature   使用OGD_Optimal特征
 */
void setup_traditional_feature_types(int handler, bool use_lfu_feature, bool use_lru_feature, bool use_ogd_opt_feature);


/**
 * 使用带窗口衰减的LFU特征
 * @param w_lens    滑动窗口大小的列表
 * @param size      滑动窗口的个数
 */
void setup_swlfu_feature_types(int handler, int *w_lens, size_t size);

/**
 * 获取特征
 * @param handler   缓存模拟器句柄
 * @param es        需要提取特征的内容
 * @param size      内容的数量
 * @return          目标内容的特征
 */
FloatBuffer get_features(int handler, ElementType *es, size_t size);

/**
 * 缓存模拟器是否处理完所有请求
 * @param handler   缓存模拟器句柄
 * @return
 */
int finished(int handler);

/**
 * 获取当前平均命中率
 * @param handler   缓存模拟器句柄
 * @return          当前平均命中率
 */
float get_mean_hit_rate(int handler);

/**
 * 获取当前回合计数
 * @param handler
 * @return
 */
int get_i_episode(int handler);

/**
 * 结束本回合
 * @param handler   缓存模拟器句柄
 * @return          该回合平均命中率
 */
float on_episode_end(int handler);

//获取特征维度
size_t feature_dims(int handler);

};
#endif
