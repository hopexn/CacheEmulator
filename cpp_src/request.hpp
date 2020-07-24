#pragma once

#include <ostream>

using namespace std;

#include "utils.h"

struct Slice
{
    Request *data;
    size_t size;

    Slice() : data(nullptr), size(0) {}

    Slice(Request *data, size_t size) : data(data), size(size) {}

    Slice sub_slice(int beg = 0, int end = -1)
    {
        if (end == -1) { end = size; }
        ASSERT(beg >= 0 && end <= size && beg <= end);
        return Slice(data + beg, end - beg);
    }

    const Request get(int idx) const
    {
        if (idx == -1) idx += size;
        ASSERT (idx < size && idx >= 0);
        return data[idx];
    }

    friend ostream &operator<<(ostream &os, const Slice &slice)
    {
        os << "Slice(";
        for (int i = 0; i < slice.size; ++i) {
            os << slice.data[i].content_id << " ";
        }
        os << ")";
        return os;
    }
};

class RequestLoader
{
private:
    vector<Request> requests;

    vector<pair<size_t, size_t>> slice_ptrs;

    TimestampType timestamp_beg = 0, timestamp_end = 0, timestamp_interval = 1;

public:
    RequestLoader() = default;

    //导入数据集
    void load_dataset(ElementType *cs, TimestampType *ts, size_t size)
    {
        for (size_t i = 0; i < size; i++) {
            this->requests.push_back({cs[i], ts[i]});
        }
    }

    //按时间分片
    size_t slice_by_time(TimestampType t_beg, TimestampType t_end, TimestampType t_interval)
    {
        this->timestamp_beg = t_beg;
        this->timestamp_end = t_end;
        this->timestamp_interval = t_interval;

        // 计算分片数量
        size_t num_slices = ceil(1.0 * (t_end - t_beg) / t_interval);

        // 分片指针，表示一个分片的起始于终止位置
        size_t ptr_beg = 0, ptr_end = 0;

        // 记录时间
        TimestampType last_time = t_beg;

        for (size_t i = 0; i < num_slices; ++i) {
            TimestampType next_time = last_time + t_interval;
            while (ptr_end < this->get_num_requests()
                   && this->requests[ptr_end].timestamp < next_time) {
                ptr_end++;
            }
            this->slice_ptrs.emplace_back(ptr_beg, ptr_end);
            ptr_beg = ptr_end;

            last_time = next_time;
        }

        //检查同一个slice里面的所有请求是否能被get_i_slice_by_timestamp映射到一块
        for (size_t i = 0; i < num_slices; i++) {
            auto ptrs = this->get_slice_range_ptrs(i);
            auto slice = this->get_slice(ptrs.first, ptrs.second);

            if (slice.size != 0) {
                int t_slice_0 = this->get_i_slice_by_timestamp(slice.get(0).timestamp);

                for (int j = 1; j < slice.size; ++j) {
                    auto t_slice_j = this->get_i_slice_by_timestamp(slice.get(j).timestamp);
                    ASSERT(t_slice_0 == t_slice_j);
                }
            }
        }

        return num_slices;
    }

    //根据时间戳得到请求所在的slice的编号
    inline int get_i_slice_by_timestamp(TimestampType t)
    {
        ASSERT(t >= timestamp_beg && t <= timestamp_end);
        return (int) ((t - timestamp_beg) / timestamp_interval);
    }

    //获取某一个时间片的请求数据的起始与终止指针
    inline pair<size_t, size_t> get_slice_range_ptrs(size_t i_slice)
    {
        ASSERT(i_slice < slice_ptrs.size());
        return this->slice_ptrs[i_slice];
    }

    //根据起始与终止指针获取片段
    inline Slice get_slice(size_t ptr_beg, size_t ptr_end)
    {
        ASSERT((ptr_beg <= ptr_end) && (ptr_end <= this->get_num_requests()));
        auto data = &this->requests[ptr_beg];
        auto size = ptr_end - ptr_beg;
        return Slice(data, size);
    }

    //请求数量
    inline size_t get_num_requests()
    {
        return this->requests.size();
    }

    //片段数量
    inline size_t get_num_slices()
    {
        return slice_ptrs.size();
    }
};
