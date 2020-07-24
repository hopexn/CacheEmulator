# 设定编译器
CC=gcc
CXX=g++

# 记录项目的跟目录
build_dir=../build

libcacheemu: $(build_dir)/libcacheemu.so

$(build_dir)/libcacheemu.so: apis.h apis.cpp cache_emu.hpp cache.hpp request.hpp feature.hpp utils.h buffer.h
	mkdir -p $(build_dir)
	$(CXX) -o $(build_dir)/libcacheemu.so -shared -fPIC apis.cpp -std=c++17 -O2

clean:
	rm -rf $(build_dir)/libcacheemu.so
