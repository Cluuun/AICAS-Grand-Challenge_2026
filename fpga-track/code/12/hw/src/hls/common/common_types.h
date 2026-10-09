#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

// 定义匹配 XDMA 的 256-bit 接口
typedef ap_axiu<256, 0, 0, 0> axis_t;
typedef hls::stream<axis_t> axis_stream;

#endif