// Original research kernels, MIT. Requant Network: https://requant.network
#include <cuda_runtime.h>

__device__ int digit(long long value, int base) {
  int r = (int)(value % base); if (r < 0) r += base;
  if (base != 13) return r >= base / 2 ? r - base : r;
  const int ds[13] = {0,1,2,3,4,-8,6,-6,8,-4,-3,-2,-1}; return ds[r];
}
__device__ unsigned code(int d) {
  int a = d < 0 ? -d : d;
  int v = a <= 4 ? a : a == 6 ? 5 : 6;
  return v | (d < 0 ? 8 : 0);
}
extern "C" __global__ void encode_i8(const long long* input, signed char* output, int size, int limbs, int base) {
  int at = blockIdx.x * blockDim.x + threadIdx.x; if (at >= size) return;
  long long value = input[at];
  for (int l = 0; l < limbs; l++) { int d = digit(value,base); output[l*(size_t)size+at] = d; value = (value-d)/base; }
}
extern "C" __global__ void encode_fp4(const long long* input, unsigned char* output, int size, int limbs) {
  int at = 2 * (blockIdx.x * blockDim.x + threadIdx.x); if (at >= size) return;
  long long a = input[at], b = input[at+1];
  for (int l = 0; l < limbs; l++) {
    int x = digit(a,13), y = digit(b,13);
    output[l*(size_t)(size/2)+at/2] = code(x) | (code(y)<<4);
    a = (a-x)/13; b = (b-y)/13;
  }
}
template<class T> __device__ void reduce(const T* partial, long long* out, int size, long long p, long long weight) {
  int at = blockIdx.x * blockDim.x + threadIdx.x; if (at >= size) return;
  long long v = (long long)partial[at] % p; if (v < 0) v += p;
  out[at] = (out[at] + v * weight) % p;
}
extern "C" __global__ void reduce_fp4(const float* partial, long long* out, int size, long long p, long long weight) { reduce(partial,out,size,p,weight); }
extern "C" __global__ void reduce_i8(const int* partial, long long* out, int size, long long p, long long weight) { reduce(partial,out,size,p,weight); }
extern "C" __global__ void simt(const long long* a, const long long* bt, long long* out, int m, int n, int k, long long p) {
  int at = blockIdx.x * blockDim.x + threadIdx.x; if (at >= m*n) return;
  int r = at/n, c = at%n; long long v = 0;
  for (int t = 0; t < k; t++) v = (v + a[(size_t)r*k+t] * bt[(size_t)c*k+t]) % p;
  out[at] = v < 0 ? v+p : v;
}
