#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Fused RMS norm + INT4 GEMV (per-group scale): y = W_int4 @ rms_norm(x, weight, eps)
// group_size=128, NUM_GROUPS = IN_DIM/128, GROUP_ITERS = 128/8 = 16
template <int IN_DIM>
__global__ void __launch_bounds__(512)
gemv_rmsnorm_int4_k(
    const uint8_t* W, const __half* x, const __half* nw, const float* sc,
    __half* y, int N, float eps)
{
    constexpr int NUM_GROUPS = IN_DIM / 128;
    constexpr int GROUP_ITERS = 128 / 8;

    __shared__ __align__(16) __half sn[IN_DIM];
    __shared__ float iv;
    int t=threadIdx.x, wid=t/32, ln=t&31, bs=blockDim.x, nwb=bs/32;
    float sq=0;
    for(int i=t;i<IN_DIM;i+=bs){float v=__half2float(x[i]);sq+=v*v;}
    sq=warp_reduce_sum(sq);
    __shared__ float sp[32];
    if(ln==0)sp[wid]=sq;
    __syncthreads();
    if(wid==0){float tot=0;for(int i=0;i<nwb;i++)tot+=sp[i];if(ln==0)iv=rsqrtf(tot/IN_DIM+eps);}
    __syncthreads();
    for(int i=t;i<IN_DIM;i+=bs){float v=__half2float(x[i]);float w=__half2float(nw[i]);sn[i]=__float2half(v*iv*w);}
    __syncthreads();
    int row=blockIdx.x*16+wid;
    if(row>=N)return;
    const uint32_t* wr=reinterpret_cast<const uint32_t*>(W+row*(IN_DIM/2));
    const float4* x4=reinterpret_cast<const float4*>(sn);
    int ni=IN_DIM/8;
    float accum=0;
    for(int i=ln;i<ni;i+=32){
        uint32_t p;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(p) : "l"(wr+i));
        float4 xv=x4[i];
        __half2*xp=reinterpret_cast<__half2*>(&xv);
        float2 f0=__half22float2(xp[0]),f1=__half22float2(xp[1]),f2=__half22float2(xp[2]),f3=__half22float2(xp[3]);
        float is=f0.x+f0.y+f1.x+f1.y+f2.x+f2.y+f3.x+f3.y;
        float raw=(float)((p)&0xF)*f0.x+(float)((p>>4)&0xF)*f0.y+(float)((p>>8)&0xF)*f1.x+(float)((p>>12)&0xF)*f1.y+(float)((p>>16)&0xF)*f2.x+(float)((p>>20)&0xF)*f2.y+(float)((p>>24)&0xF)*f3.x+(float)(p>>28)*f3.y;
        int grp=i/GROUP_ITERS;
        float gs=__ldg(sc+row*NUM_GROUPS+grp);
        accum+=(raw-8.0f*is)*gs;
    }
    accum=warp_reduce_sum(accum);
    if(ln==0)y[row]=__float2half(accum);
}

// Fused: add + RMS norm + gate_up INT4 GEMV + SiLU*Mul (per-group scale)
// NO writeback to hidden/residual — caller handles residual update separately
template <int IN_DIM>
__global__ void __launch_bounds__(512)
gemv_addrmsnorm_silu_mul_int4_k(
    const uint8_t* W, const __half* hid, const __half* attn,
    const __half* nw, const float* sc,
    __half* y, int od, float eps)
{
    constexpr int NUM_GROUPS = IN_DIM / 128;
    constexpr int GROUP_ITERS = 128 / 8;

    __shared__ __align__(16) __half sn[IN_DIM];
    __shared__ float iv;
    int t=threadIdx.x, wid=t/32, ln=t&31, bs=blockDim.x, nwb=bs/32;

    // Phase 1: compute residual = hid + attn, store in shared memory, compute sum_sq
    float sq=0;
    for(int i=t;i<IN_DIM;i+=bs){
        float h=__half2float(hid[i]);
        float a=__half2float(attn[i]);
        float v=h+a;
        sq+=v*v;
        sn[i]=__float2half(v);
    }
    sq=warp_reduce_sum(sq);
    __shared__ float sp[32];
    if(ln==0)sp[wid]=sq;
    __syncthreads();
    if(wid==0){float tot=0;for(int i=0;i<nwb;i++)tot+=sp[i];if(ln==0)iv=rsqrtf(tot/IN_DIM+eps);}
    __syncthreads();

    // Phase 1b: apply norm weight, overwrite shared memory
    for(int i=t;i<IN_DIM;i+=bs){
        float v=__half2float(sn[i]);
        float w=__half2float(nw[i]);
        sn[i]=__float2half(v*iv*w);
    }
    __syncthreads();

    // Phase 2: gate_up INT4 GEMV + SiLU*Mul with per-group scale
    int row=blockIdx.x*16+wid;
    if(row>=od)return;
    const uint32_t* wg=reinterpret_cast<const uint32_t*>(W+row*(IN_DIM/2));
    const uint32_t* wu=reinterpret_cast<const uint32_t*>(W+(row+od)*(IN_DIM/2));
    const float4* x4=reinterpret_cast<const float4*>(sn);
    int ni=IN_DIM/8;
    float gaccum=0, uaccum=0;
    for(int i=ln;i<ni;i+=32){
        float4 xv=x4[i];
        __half2*xp=reinterpret_cast<__half2*>(&xv);
        float2 f0=__half22float2(xp[0]),f1=__half22float2(xp[1]),f2=__half22float2(xp[2]),f3=__half22float2(xp[3]);
        float is=f0.x+f0.y+f1.x+f1.y+f2.x+f2.y+f3.x+f3.y;
        uint32_t pg;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(pg) : "l"(wg+i));
        float gr=(float)((pg)&0xF)*f0.x+(float)((pg>>4)&0xF)*f0.y+(float)((pg>>8)&0xF)*f1.x+(float)((pg>>12)&0xF)*f1.y+(float)((pg>>16)&0xF)*f2.x+(float)((pg>>20)&0xF)*f2.y+(float)((pg>>24)&0xF)*f3.x+(float)(pg>>28)*f3.y;
        uint32_t pu;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(pu) : "l"(wu+i));
        float ur=(float)((pu)&0xF)*f0.x+(float)((pu>>4)&0xF)*f0.y+(float)((pu>>8)&0xF)*f1.x+(float)((pu>>12)&0xF)*f1.y+(float)((pu>>16)&0xF)*f2.x+(float)((pu>>20)&0xF)*f2.y+(float)((pu>>24)&0xF)*f3.x+(float)(pu>>28)*f3.y;
        int grp=i/GROUP_ITERS;
        float gsc=__ldg(sc+row*NUM_GROUPS+grp);
        float usc=__ldg(sc+(row+od)*NUM_GROUPS+grp);
        gaccum+=(gr-8.0f*is)*gsc;
        uaccum+=(ur-8.0f*is)*usc;
    }
    gaccum=warp_reduce_sum(gaccum);
    uaccum=warp_reduce_sum(uaccum);
    if(ln==0){
        float sv=gaccum/(1.0f+expf(-gaccum));
        y[row]=__float2half(sv*uaccum);
    }
}

torch::Tensor gemv_rmsnorm_int4_cuda(
    torch::Tensor W_packed, torch::Tensor x, torch::Tensor norm_weight,
    torch::Tensor scale, float eps)
{
    int N = W_packed.size(0);
    auto y = torch::empty({N}, x.options());
    int block = 512, grid = (N + 15) / 16;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    gemv_rmsnorm_int4_k<2048><<<grid, block, 0, stream>>>(
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N, eps);
    return y;
}

torch::Tensor gemv_addrmsnorm_silu_mul_int4_cuda(
    torch::Tensor W_packed, torch::Tensor hidden, torch::Tensor attn_out,
    torch::Tensor norm_weight, torch::Tensor scale, float eps)
{
    int out_dim = W_packed.size(0) / 2;
    auto y = torch::empty({out_dim}, hidden.options());
    int block = 512, grid = (out_dim + 15) / 16;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    gemv_addrmsnorm_silu_mul_int4_k<2048><<<grid, block, 0, stream>>>(
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(attn_out.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        out_dim, eps);
    return y;
}

// ---------------------------------------------------------------------------
// Fused RMS norm + INT8 GEMV: y = W_int8 @ rms_norm(x, weight, eps)
// ---------------------------------------------------------------------------

template <int IN_DIM>
__global__ void __launch_bounds__(256)
gemv_rmsnorm_int8_k(
    const int8_t* W, const __half* x, const __half* nw, const float* sc,
    __half* y, int N, float eps)
{
    __shared__ __align__(16) __half sn[IN_DIM];
    __shared__ float iv;
    int t=threadIdx.x, wid=t/32, ln=t&31, bs=blockDim.x, nwb=bs/32;

    // Phase 1: RMS norm → shared memory
    float sq=0;
    for(int i=t;i<IN_DIM;i+=bs){float v=__half2float(x[i]);sq+=v*v;}
    sq=warp_reduce_sum(sq);
    __shared__ float sp[32];
    if(ln==0)sp[wid]=sq;
    __syncthreads();
    if(wid==0){float tot=0;for(int i=0;i<nwb;i++)tot+=sp[i];if(ln==0)iv=rsqrtf(tot/IN_DIM+eps);}
    __syncthreads();
    for(int i=t;i<IN_DIM;i+=bs){float v=__half2float(x[i]);float w=__half2float(nw[i]);sn[i]=__float2half(v*iv*w);}
    __syncthreads();

    // Phase 2: INT8 GEMV using shared memory normed values
    int row=blockIdx.x*8+wid;
    if(row>=N)return;
    const int8_t* wr=W+row*IN_DIM;
    float rs=sc[row];
    const float4* x4=reinterpret_cast<const float4*>(sn);
    int ni=IN_DIM/8;
    float accum=0;

    for(int i=ln;i<ni;i+=32){
        int2 wv_raw;
        asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];"
                     : "=r"(reinterpret_cast<unsigned*>(&wv_raw)[0]),
                       "=r"(reinterpret_cast<unsigned*>(&wv_raw)[1])
                     : "l"(reinterpret_cast<const int2*>(wr) + i));
        const int8_t* wi=reinterpret_cast<const int8_t*>(&wv_raw);
        float4 xv=x4[i];
        __half2*xp=reinterpret_cast<__half2*>(&xv);
        float2 f0=__half22float2(xp[0]),f1=__half22float2(xp[1]),f2=__half22float2(xp[2]),f3=__half22float2(xp[3]);
        accum+=(float)wi[0]*f0.x+(float)wi[1]*f0.y+(float)wi[2]*f1.x+(float)wi[3]*f1.y
              +(float)wi[4]*f2.x+(float)wi[5]*f2.y+(float)wi[6]*f3.x+(float)wi[7]*f3.y;
    }
    accum=warp_reduce_sum(accum);
    if(ln==0)y[row]=__float2half(accum*rs);
}

// WARPS=16 variant for better bandwidth utilization on small matrices
template <int IN_DIM, int WPB>
__global__ void __launch_bounds__(WPB * 32)
gemv_rmsnorm_int8_k_wpb(
    const int8_t* W, const __half* x, const __half* nw, const float* sc,
    __half* y, int N, float eps)
{
    __shared__ __align__(16) __half sn[IN_DIM];
    __shared__ float iv;
    int t=threadIdx.x, wid=t/32, ln=t&31, bs=blockDim.x, nwb=bs/32;

    float sq=0;
    for(int i=t;i<IN_DIM;i+=bs){float v=__half2float(x[i]);sq+=v*v;}
    sq=warp_reduce_sum(sq);
    __shared__ float sp[32];
    if(ln==0)sp[wid]=sq;
    __syncthreads();
    if(wid==0){float tot=0;for(int i=0;i<nwb;i++)tot+=sp[i];if(ln==0)iv=rsqrtf(tot/IN_DIM+eps);}
    __syncthreads();
    for(int i=t;i<IN_DIM;i+=bs){float v=__half2float(x[i]);float w=__half2float(nw[i]);sn[i]=__float2half(v*iv*w);}
    __syncthreads();

    int row=blockIdx.x*WPB+wid;
    if(row>=N)return;
    const int8_t* wr=W+row*IN_DIM;
    float rs=sc[row];
    const float4* x4=reinterpret_cast<const float4*>(sn);
    int ni=IN_DIM/8;
    float accum=0;

    for(int i=ln;i<ni;i+=32){
        int2 wv_raw;
        asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];"
                     : "=r"(reinterpret_cast<unsigned*>(&wv_raw)[0]),
                       "=r"(reinterpret_cast<unsigned*>(&wv_raw)[1])
                     : "l"(reinterpret_cast<const int2*>(wr) + i));
        const int8_t* wi=reinterpret_cast<const int8_t*>(&wv_raw);
        float4 xv=x4[i];
        __half2*xp=reinterpret_cast<__half2*>(&xv);
        float2 f0=__half22float2(xp[0]),f1=__half22float2(xp[1]),f2=__half22float2(xp[2]),f3=__half22float2(xp[3]);
        accum+=(float)wi[0]*f0.x+(float)wi[1]*f0.y+(float)wi[2]*f1.x+(float)wi[3]*f1.y
              +(float)wi[4]*f2.x+(float)wi[5]*f2.y+(float)wi[6]*f3.x+(float)wi[7]*f3.y;
    }
    accum=warp_reduce_sum(accum);
    if(ln==0)y[row]=__float2half(accum*rs);
}

// Dual-row rmsnorm INT8 GEMV: each warp processes 2 rows for better latency hiding.
// RMS norm is computed once (shared), then two rows' dot products are interleaved.
template <int IN_DIM, int WPB>
__global__ void __launch_bounds__(WPB * 32)
gemv_rmsnorm_int8_dual_k(
    const int8_t* W, const __half* x, const __half* nw, const float* sc,
    __half* y, int N, float eps)
{
    __shared__ __align__(16) __half sn[IN_DIM];
    __shared__ float iv;
    int t=threadIdx.x, wid=t/32, ln=t&31, bs=blockDim.x, nwb=bs/32;

    // Phase 1: RMS norm → shared memory (shared across all rows in this block)
    float sq=0;
    for(int i=t;i<IN_DIM;i+=bs){float v=__half2float(x[i]);sq+=v*v;}
    sq=warp_reduce_sum(sq);
    __shared__ float sp[32];
    if(ln==0)sp[wid]=sq;
    __syncthreads();
    if(wid==0){float tot=0;for(int i=0;i<nwb;i++)tot+=sp[i];if(ln==0)iv=rsqrtf(tot/IN_DIM+eps);}
    __syncthreads();
    for(int i=t;i<IN_DIM;i+=bs){float v=__half2float(x[i]);float w=__half2float(nw[i]);sn[i]=__float2half(v*iv*w);}
    __syncthreads();

    // Phase 2: Dual-row INT8 GEMV
    int row0=(blockIdx.x*WPB+wid)*2;
    if(row0>=N)return;

    const int8_t* wr0=W+row0*IN_DIM;
    float rs0=sc[row0];
    const float4* x4=reinterpret_cast<const float4*>(sn);
    int ni=IN_DIM/8;
    float acc0=0, acc1=0;

    bool has_row1=(row0+1<N);
    const int8_t* wr1=has_row1?(W+(row0+1)*IN_DIM):nullptr;
    float rs1=has_row1?sc[row0+1]:0.0f;

    for(int i=ln;i<ni;i+=32){
        float4 xv=x4[i];
        __half2*xp=reinterpret_cast<__half2*>(&xv);
        float2 f0=__half22float2(xp[0]),f1=__half22float2(xp[1]),f2=__half22float2(xp[2]),f3=__half22float2(xp[3]);

        int2 wv0;
        asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];"
                     : "=r"(reinterpret_cast<unsigned*>(&wv0)[0]),
                       "=r"(reinterpret_cast<unsigned*>(&wv0)[1])
                     : "l"(reinterpret_cast<const int2*>(wr0) + i));
        const int8_t* wi0=reinterpret_cast<const int8_t*>(&wv0);
        acc0+=(float)wi0[0]*f0.x+(float)wi0[1]*f0.y+(float)wi0[2]*f1.x+(float)wi0[3]*f1.y
             +(float)wi0[4]*f2.x+(float)wi0[5]*f2.y+(float)wi0[6]*f3.x+(float)wi0[7]*f3.y;

        if(has_row1){
            int2 wv1;
            asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];"
                         : "=r"(reinterpret_cast<unsigned*>(&wv1)[0]),
                           "=r"(reinterpret_cast<unsigned*>(&wv1)[1])
                         : "l"(reinterpret_cast<const int2*>(wr1) + i));
            const int8_t* wi1=reinterpret_cast<const int8_t*>(&wv1);
            acc1+=(float)wi1[0]*f0.x+(float)wi1[1]*f0.y+(float)wi1[2]*f1.x+(float)wi1[3]*f1.y
                 +(float)wi1[4]*f2.x+(float)wi1[5]*f2.y+(float)wi1[6]*f3.x+(float)wi1[7]*f3.y;
        }
    }

    acc0=warp_reduce_sum(acc0);
    acc1=warp_reduce_sum(acc1);
    if(ln==0){
        y[row0]=__float2half(acc0*rs0);
        if(has_row1)y[row0+1]=__float2half(acc1*rs1);
    }
}

torch::Tensor gemv_rmsnorm_int8_cuda(
    torch::Tensor W_int8, torch::Tensor x, torch::Tensor norm_weight,
    torch::Tensor scale, float eps)
{
    int N = W_int8.size(0);
    auto y = torch::empty({N}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const char* env_dual = std::getenv("AICAS_RMSNORM_GEMV_DUAL");
    if (env_dual != nullptr && env_dual[0] == '1') {
        // Dual-row: each warp processes 2 rows, grid halved
        constexpr int WPB = 8;
        int block = WPB * 32;
        int grid = ((N + 1) / 2 + WPB - 1) / WPB;
        gemv_rmsnorm_int8_dual_k<2048, WPB><<<grid, block, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N, eps);
    } else {
        const char* env_w = std::getenv("AICAS_RMSNORM_GEMV_WARPS");
        int wpb = env_w ? atoi(env_w) : 8;
        if (wpb == 16) {
            int block = 512, grid = (N + 15) / 16;
            gemv_rmsnorm_int8_k_wpb<2048, 16><<<grid, block, 0, stream>>>(
                W_int8.data_ptr<int8_t>(),
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
                scale.data_ptr<float>(),
                reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
                N, eps);
        } else {
            int block = 256, grid = (N + 7) / 8;
            gemv_rmsnorm_int8_k<2048><<<grid, block, 0, stream>>>(
                W_int8.data_ptr<int8_t>(),
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
                scale.data_ptr<float>(),
                reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
                N, eps);
        }
    }
    return y;
}

// ---------------------------------------------------------------------------
// Fused: add + RMS norm + gate_up INT8 GEMV + SiLU*Mul (per-channel scale)
// ---------------------------------------------------------------------------

template <int IN_DIM, int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_addrmsnorm_silu_mul_int8_k(
    const int8_t* W, const __half* hid, const __half* attn,
    const __half* nw, const float* sc,
    __half* y, int od, float eps)
{
    __shared__ __align__(16) __half sn[IN_DIM];
    __shared__ float iv;
    int t=threadIdx.x, wid=t/32, ln=t&31, bs=blockDim.x, nwb=bs/32;

    // Phase 1: add + RMS norm → shared memory
    float sq=0;
    for(int i=t;i<IN_DIM;i+=bs){
        float h=__half2float(hid[i]);
        float a=__half2float(attn[i]);
        float v=h+a;
        sq+=v*v;
        sn[i]=__float2half(v);
    }
    sq=warp_reduce_sum(sq);
    __shared__ float sp[32];
    if(ln==0)sp[wid]=sq;
    __syncthreads();
    if(wid==0){float tot=0;for(int i=0;i<nwb;i++)tot+=sp[i];if(ln==0)iv=rsqrtf(tot/IN_DIM+eps);}
    __syncthreads();

    // Phase 1b: apply norm weight
    for(int i=t;i<IN_DIM;i+=bs){
        float v=__half2float(sn[i]);
        float w=__half2float(nw[i]);
        sn[i]=__float2half(v*iv*w);
    }
    __syncthreads();

    // Phase 2: gate_up INT8 GEMV + SiLU*Mul with per-channel scale
    int row=blockIdx.x*WARPS_PER_BLOCK+wid;
    if(row>=od)return;
    const int8_t* wg=W+row*IN_DIM;
    const int8_t* wu=W+(row+od)*IN_DIM;
    float gsc=sc[row], usc=sc[row+od];
    const float4* x4=reinterpret_cast<const float4*>(sn);
    int ni=IN_DIM/8;
    float gaccum=0, uaccum=0;

    for(int i=ln;i<ni;i+=32){
        int2 gv;
        asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];"
                     : "=r"(reinterpret_cast<unsigned*>(&gv)[0]),
                       "=r"(reinterpret_cast<unsigned*>(&gv)[1])
                     : "l"(reinterpret_cast<const int2*>(wg) + i));
        const int8_t* gi=reinterpret_cast<const int8_t*>(&gv);
        int2 uv;
        asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];"
                     : "=r"(reinterpret_cast<unsigned*>(&uv)[0]),
                       "=r"(reinterpret_cast<unsigned*>(&uv)[1])
                     : "l"(reinterpret_cast<const int2*>(wu) + i));
        const int8_t* ui=reinterpret_cast<const int8_t*>(&uv);
        float4 xv=x4[i];
        __half2*xp=reinterpret_cast<__half2*>(&xv);
        float2 f0=__half22float2(xp[0]),f1=__half22float2(xp[1]),f2=__half22float2(xp[2]),f3=__half22float2(xp[3]);
        gaccum+=(float)gi[0]*f0.x+(float)gi[1]*f0.y+(float)gi[2]*f1.x+(float)gi[3]*f1.y
               +(float)gi[4]*f2.x+(float)gi[5]*f2.y+(float)gi[6]*f3.x+(float)gi[7]*f3.y;
        uaccum+=(float)ui[0]*f0.x+(float)ui[1]*f0.y+(float)ui[2]*f1.x+(float)ui[3]*f1.y
               +(float)ui[4]*f2.x+(float)ui[5]*f2.y+(float)ui[6]*f3.x+(float)ui[7]*f3.y;
    }
    gaccum=warp_reduce_sum(gaccum);
    uaccum=warp_reduce_sum(uaccum);
    if(ln==0){
        float g=gaccum*gsc;
        float u=uaccum*usc;
        float sv=g/(1.0f+expf(-g));
        y[row]=__float2half(sv*u);
    }
}

torch::Tensor gemv_addrmsnorm_silu_mul_int8_cuda(
    torch::Tensor W_int8, torch::Tensor hidden, torch::Tensor attn_out,
    torch::Tensor norm_weight, torch::Tensor scale, float eps)
{
    int out_dim = W_int8.size(0) / 2;
    auto y = torch::empty({out_dim}, hidden.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const char* w32 = std::getenv("AICAS_FUSED_GATE_WARPS32");
    const char* w28 = std::getenv("AICAS_FUSED_GATE_WARPS28");
    const char* w26 = std::getenv("AICAS_FUSED_GATE_WARPS26");
    const char* w24 = std::getenv("AICAS_FUSED_GATE_WARPS24");
    const char* w22 = std::getenv("AICAS_FUSED_GATE_WARPS22");
    const char* w20 = std::getenv("AICAS_FUSED_GATE_WARPS20");
    const char* w16 = std::getenv("AICAS_FUSED_GATE_WARPS16");
    if (w32 != nullptr && w32[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 32;
        int block = WARPS_PER_BLOCK * 32;
        int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_addrmsnorm_silu_mul_int8_k<2048, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(attn_out.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            out_dim, eps);
    } else if (w28 != nullptr && w28[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 28;
        int block = WARPS_PER_BLOCK * 32;
        int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_addrmsnorm_silu_mul_int8_k<2048, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(attn_out.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            out_dim, eps);
    } else if (w26 != nullptr && w26[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 26;
        int block = WARPS_PER_BLOCK * 32;
        int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_addrmsnorm_silu_mul_int8_k<2048, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(attn_out.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            out_dim, eps);
    } else if (w24 != nullptr && w24[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 24;
        int block = WARPS_PER_BLOCK * 32;
        int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_addrmsnorm_silu_mul_int8_k<2048, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(attn_out.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            out_dim, eps);
    } else if (w22 != nullptr && w22[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 22;
        int block = WARPS_PER_BLOCK * 32;
        int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_addrmsnorm_silu_mul_int8_k<2048, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(attn_out.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            out_dim, eps);
    } else if (w20 != nullptr && w20[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 20;
        int block = WARPS_PER_BLOCK * 32;
        int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_addrmsnorm_silu_mul_int8_k<2048, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(attn_out.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            out_dim, eps);
    } else if (w16 != nullptr && w16[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 16;
        int block = WARPS_PER_BLOCK * 32;
        int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_addrmsnorm_silu_mul_int8_k<2048, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(attn_out.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            out_dim, eps);
    } else {
        constexpr int WARPS_PER_BLOCK = 8;
        int block = WARPS_PER_BLOCK * 32;
        int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_addrmsnorm_silu_mul_int8_k<2048, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(attn_out.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            out_dim, eps);
    }
    return y;
}

// Experimental W8A8 gate/up path:
// add + RMSNorm is still computed in FP32/FP16, then the token activation is
// dynamically quantized to signed INT8 and consumed with dp4a. This is off by
// default because greedy decoding is sensitive to activation quantization.
template <int IN_DIM>
__global__ void __launch_bounds__(256)
gemv_addrmsnorm_silu_mul_w8a8_k(
    const int8_t* W, const __half* hid, const __half* attn,
    const __half* nw, const float* sc,
    __half* y, int od, float eps)
{
    __shared__ __align__(16) __half sn[IN_DIM];
    __shared__ __align__(16) int8_t qx[IN_DIM];
    __shared__ float shared_vals[32];
    __shared__ float iv;
    __shared__ float x_scale;

    int t = threadIdx.x, wid = t / 32, ln = t & 31, bs = blockDim.x, nwb = bs / 32;

    float sq = 0.0f;
    for (int i = t; i < IN_DIM; i += bs) {
        float v = __half2float(hid[i]) + __half2float(attn[i]);
        sq += v * v;
        sn[i] = __float2half(v);
    }
    sq = warp_reduce_sum(sq);
    if (ln == 0) shared_vals[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < nwb; ++i) tot += shared_vals[i];
        if (ln == 0) iv = rsqrtf(tot / IN_DIM + eps);
    }
    __syncthreads();

    float local_max = 0.0f;
    for (int i = t; i < IN_DIM; i += bs) {
        float v = __half2float(sn[i]) * iv * __half2float(nw[i]);
        sn[i] = __float2half(v);
        local_max = fmaxf(local_max, fabsf(v));
    }
    local_max = warp_reduce_max(local_max);
    if (ln == 0) shared_vals[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        float mx = 0.0f;
        for (int i = 0; i < nwb; ++i) mx = fmaxf(mx, shared_vals[i]);
        if (ln == 0) x_scale = fmaxf(mx, 1.0e-6f) * (1.0f / 127.0f);
    }
    __syncthreads();

    float inv_scale = 1.0f / x_scale;
    for (int i = t; i < IN_DIM; i += bs) {
        float q = rintf(__half2float(sn[i]) * inv_scale);
        q = fminf(127.0f, fmaxf(-127.0f, q));
        qx[i] = static_cast<int8_t>(q);
    }
    __syncthreads();

    int row = blockIdx.x * 8 + wid;
    if (row >= od) return;

    const int8_t* wg = W + row * IN_DIM;
    const int8_t* wu = W + (row + od) * IN_DIM;
    const int2* wg2 = reinterpret_cast<const int2*>(wg);
    const int2* wu2 = reinterpret_cast<const int2*>(wu);
    const int2* x2 = reinterpret_cast<const int2*>(qx);
    int ni = IN_DIM / 8;
    int gaccum = 0;
    int uaccum = 0;

    for (int i = ln; i < ni; i += 32) {
        int2 gv = __ldg(wg2 + i);
        int2 uv = __ldg(wu2 + i);
        int2 xv = x2[i];
        gaccum = __dp4a(gv.x, xv.x, gaccum);
        gaccum = __dp4a(gv.y, xv.y, gaccum);
        uaccum = __dp4a(uv.x, xv.x, uaccum);
        uaccum = __dp4a(uv.y, xv.y, uaccum);
    }

    float gf = static_cast<float>(gaccum);
    float uf = static_cast<float>(uaccum);
    gf = warp_reduce_sum(gf);
    uf = warp_reduce_sum(uf);
    if (ln == 0) {
        float g = gf * x_scale * sc[row];
        float u = uf * x_scale * sc[row + od];
        float sv = g / (1.0f + expf(-g));
        y[row] = __float2half(sv * u);
    }
}

torch::Tensor gemv_addrmsnorm_silu_mul_w8a8_cuda(
    torch::Tensor W_int8, torch::Tensor hidden, torch::Tensor attn_out,
    torch::Tensor norm_weight, torch::Tensor scale, float eps)
{
    int out_dim = W_int8.size(0) / 2;
    auto y = torch::empty({out_dim}, hidden.options());
    int block = 256, grid = (out_dim + 7) / 8;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    gemv_addrmsnorm_silu_mul_w8a8_k<2048><<<grid, block, 0, stream>>>(
        W_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(attn_out.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        out_dim, eps);
    return y;
}

// Fused RMS norm + lm_head INT4 GEMV (large vocab, 16 warps/block, L2 bypass)
template <int IN_DIM>
__global__ void __launch_bounds__(512)
gemv_rmsnorm_lmhead_int4_k(
    const uint8_t* W, const __half* x, const __half* nw, const float* sc,
    __half* y, int N, float eps)
{
    constexpr int NUM_GROUPS = IN_DIM / 128;
    constexpr int GROUP_ITERS = 128 / 8;

    __shared__ __align__(16) __half sn[IN_DIM];
    __shared__ float iv;
    int t=threadIdx.x, wid=t/32, ln=t&31, bs=blockDim.x, nwb=bs/32;
    float sq=0;
    for(int i=t;i<IN_DIM;i+=bs){float v=__half2float(x[i]);sq+=v*v;}
    sq=warp_reduce_sum(sq);
    __shared__ float sp[32];
    if(ln==0)sp[wid]=sq;
    __syncthreads();
    if(wid==0){float tot=0;for(int i=0;i<nwb;i++)tot+=sp[i];if(ln==0)iv=rsqrtf(tot/IN_DIM+eps);}
    __syncthreads();
    for(int i=t;i<IN_DIM;i+=bs){float v=__half2float(x[i]);float w=__half2float(nw[i]);sn[i]=__float2half(v*iv*w);}
    __syncthreads();
    int row=blockIdx.x*16+wid;
    if(row>=N)return;
    const uint32_t* wr=reinterpret_cast<const uint32_t*>(W+row*(IN_DIM/2));
    const float4* x4=reinterpret_cast<const float4*>(sn);
    int ni=IN_DIM/8;
    float accum=0;
    for(int i=ln;i<ni;i+=32){
        uint32_t p;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(p) : "l"(wr+i));
        float4 xv=x4[i];
        __half2*xp=reinterpret_cast<__half2*>(&xv);
        float2 f0=__half22float2(xp[0]),f1=__half22float2(xp[1]),f2=__half22float2(xp[2]),f3=__half22float2(xp[3]);
        float is=f0.x+f0.y+f1.x+f1.y+f2.x+f2.y+f3.x+f3.y;
        float raw=(float)((p)&0xF)*f0.x+(float)((p>>4)&0xF)*f0.y+(float)((p>>8)&0xF)*f1.x+(float)((p>>12)&0xF)*f1.y+(float)((p>>16)&0xF)*f2.x+(float)((p>>20)&0xF)*f2.y+(float)((p>>24)&0xF)*f3.x+(float)(p>>28)*f3.y;
        int grp=i/GROUP_ITERS;
        float gs=__ldg(sc+row*NUM_GROUPS+grp);
        accum+=(raw-8.0f*is)*gs;
    }
    accum=warp_reduce_sum(accum);
    if(ln==0)y[row]=__float2half(accum);
}

torch::Tensor gemv_rmsnorm_lmhead_int4_cuda(
    torch::Tensor W_packed, torch::Tensor x, torch::Tensor norm_weight,
    torch::Tensor scale, float eps)
{
    int N = W_packed.size(0);
    auto y = torch::empty({N}, x.options());
    int block = 512, grid = (N + 15) / 16;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    gemv_rmsnorm_lmhead_int4_k<2048><<<grid, block, 0, stream>>>(
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N, eps);
    return y;
}

// ---------------------------------------------------------------------------
// Fused RMS norm + lm_head INT8 GEMV + argmax (two-stage)
// Eliminates: final RMS norm kernel, argmax kernel, writing 151936 logits
// Stage 1: RMS norm + INT8 GEMV + per-block argmax
// Stage 2: reduce per-block argmax to global argmax
// ---------------------------------------------------------------------------

template <int IN_DIM>
__global__ void __launch_bounds__(512)
gemv_rmsnorm_lmhead_int8_argmax_stage1_kernel(
    const int8_t* W, const __half* x, const __half* nw, const float* sc,
    float* block_max_val, int* block_max_idx,
    int N, float eps)
{
    __shared__ __align__(16) __half sn[IN_DIM];
    __shared__ float iv;
    int t=threadIdx.x, wid=t/32, ln=t&31, bs=blockDim.x, nwb=bs/32;

    // Phase 1: RMS norm → shared memory
    float sq=0;
    for(int i=t;i<IN_DIM;i+=bs){float v=__half2float(x[i]);sq+=v*v;}
    sq=warp_reduce_sum(sq);
    __shared__ float sp[32];
    if(ln==0)sp[wid]=sq;
    __syncthreads();
    if(wid==0){float tot=0;for(int i=0;i<nwb;i++)tot+=sp[i];if(ln==0)iv=rsqrtf(tot/IN_DIM+eps);}
    __syncthreads();
    for(int i=t;i<IN_DIM;i+=bs){float v=__half2float(x[i]);float w=__half2float(nw[i]);sn[i]=__float2half(v*iv*w);}
    __syncthreads();

    // Phase 2: INT8 GEMV + per-block argmax
    int row = blockIdx.x * 16 + wid;
    bool valid = (row < N);

    float accum = 0.0f;
    if (valid) {
        const int8_t* wr = W + row * IN_DIM;
        float rs = sc[row];
        const float4* x4 = reinterpret_cast<const float4*>(sn);
        int ni = IN_DIM / 8;

        for(int i=ln;i<ni;i+=32){
            int2 wv_raw;
            asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];"
                         : "=r"(reinterpret_cast<unsigned*>(&wv_raw)[0]),
                           "=r"(reinterpret_cast<unsigned*>(&wv_raw)[1])
                         : "l"(reinterpret_cast<const int2*>(wr) + i));
            const int8_t* wi = reinterpret_cast<const int8_t*>(&wv_raw);
            float4 xv = x4[i];
            __half2*xp = reinterpret_cast<__half2*>(&xv);
            float2 f0=__half22float2(xp[0]),f1=__half22float2(xp[1]),
                    f2=__half22float2(xp[2]),f3=__half22float2(xp[3]);
            accum+=(float)wi[0]*f0.x+(float)wi[1]*f0.y+(float)wi[2]*f1.x+(float)wi[3]*f1.y
                  +(float)wi[4]*f2.x+(float)wi[5]*f2.y+(float)wi[6]*f3.x+(float)wi[7]*f3.y;
        }
        accum *= rs;
    }
    accum = warp_reduce_sum(accum);

    // Warp-level argmax
    float best_val = (valid && ln == 0) ? accum : -1e30f;
    int best_row = (valid && ln == 0) ? row : N;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, best_val, offset);
        int oi = __shfl_down_sync(0xffffffff, best_row, offset);
        if (ov > best_val || (ov == best_val && oi < best_row)) {
            best_val = ov; best_row = oi;
        }
    }

    // Block-level argmax via shared memory
    __shared__ float smem_val[16];
    __shared__ int smem_idx[16];
    if (ln == 0) { smem_val[wid] = best_val; smem_idx[wid] = best_row; }
    __syncthreads();

    if (wid == 0) {
        float bv = (ln < nwb) ? smem_val[ln] : -1e30f;
        int bi = (ln < nwb) ? smem_idx[ln] : N;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, bv, offset);
            int oi = __shfl_down_sync(0xffffffff, bi, offset);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (ln == 0) {
            block_max_val[blockIdx.x] = bv;
            block_max_idx[blockIdx.x] = bi;
        }
    }
}

__global__ void argmax_reduce_kernel(
    const float* vals, const int* idxs, int64_t* out, int nblocks)
{
    float best_val = -1e30f;
    int best_idx = -1;
    for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
        float v = vals[i]; int idx = idxs[i];
        if (v > best_val || (v == best_val && idx < best_idx)) {
            best_val = v; best_idx = idx;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, best_val, offset);
        int oi = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (ov > best_val || (ov == best_val && oi < best_idx)) {
            best_val = ov; best_idx = oi;
        }
    }
    if (threadIdx.x == 0) out[0] = best_idx;
}

void gemv_rmsnorm_lmhead_int8_argmax_cuda(
    torch::Tensor W_int8, torch::Tensor x, torch::Tensor norm_weight,
    torch::Tensor scale, torch::Tensor block_vals, torch::Tensor block_idxs,
    torch::Tensor result, float eps)
{
    int N = W_int8.size(0);
    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_rmsnorm_lmhead_int8_argmax_stage1_kernel<2048>
    <<<grid, block_size, 0, stream>>>(
        W_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        block_vals.data_ptr<float>(),
        block_idxs.data_ptr<int>(),
        N, eps);

    argmax_reduce_kernel<<<1, 256, 0, stream>>>(
        block_vals.data_ptr<float>(),
        block_idxs.data_ptr<int>(),
        result.data_ptr<int64_t>(),
        grid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_rmsnorm_int4", &gemv_rmsnorm_int4_cuda, "Fused RMS norm + INT4 GEMV");
    m.def("gemv_addrmsnorm_silu_mul_int4", &gemv_addrmsnorm_silu_mul_int4_cuda,
          "Fused add+RMS norm + gate_up INT4 GEMV+SiLU");
    m.def("gemv_rmsnorm_int8", &gemv_rmsnorm_int8_cuda, "Fused RMS norm + INT8 GEMV");
    m.def("gemv_addrmsnorm_silu_mul_int8", &gemv_addrmsnorm_silu_mul_int8_cuda,
          "Fused add+RMS norm + gate_up INT8 GEMV+SiLU");
    m.def("gemv_addrmsnorm_silu_mul_w8a8", &gemv_addrmsnorm_silu_mul_w8a8_cuda,
          "Experimental fused add+RMS norm + dynamic W8A8 gate_up GEMV+SiLU");
    m.def("gemv_rmsnorm_lmhead_int4", &gemv_rmsnorm_lmhead_int4_cuda,
          "Fused RMS norm + lm_head INT4 GEMV (16 warps, L2 bypass)");
    m.def("gemv_rmsnorm_lmhead_int8_argmax", &gemv_rmsnorm_lmhead_int8_argmax_cuda,
          "Fused RMS norm + lm_head INT8 GEMV + argmax");
}
