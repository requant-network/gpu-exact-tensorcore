/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// Requant Network: FP32 output and original C ABI around the pinned CUTLASS configuration.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
using namespace cute;
// A matrix configuration
using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementD    = float;                            // Element type for D matrix operand
using         ElementC    = float;                            // Element type for C matrix operand
using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm120;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

// Kernel Perf config
using ThreadBlockShape    = Shape<_128,_128,_128>;                          // Threadblock's tile size
using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto                      // Epilogue schedule policy
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto defaults to cooperative kernel schedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,                                                   // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideC   = typename Gemm::GemmKernel::StrideC;
using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

//

static thread_local std::string error_message;
static void cuda_ok(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void cutlass_ok(cutlass::Status s){if(s!=cutlass::Status::kSuccess)throw std::runtime_error(cutlassGetStatusString(s));}
struct Plan {
 int m,n,k;StrideA sa;StrideB sb;StrideC sc;StrideD sd;LayoutSFA la;LayoutSFB lb;
 ElementA::ScaleFactorType *sfa=nullptr,*sfb=nullptr;void* workspace=nullptr;Gemm gemm;
 Plan(int a,int b,int c,const void* A,const void* B,void* D):m(a),n(b),k(c){
  using Scale=typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  sa=cutlass::make_cute_packed_stride(StrideA{},{m,k,1});sb=cutlass::make_cute_packed_stride(StrideB{},{n,k,1});
  sc=cutlass::make_cute_packed_stride(StrideC{},{m,n,1});sd=cutlass::make_cute_packed_stride(StrideD{},{m,n,1});
  la=Scale::tile_atom_to_shape_SFA(cute::make_shape(m,n,k,1));lb=Scale::tile_atom_to_shape_SFB(cute::make_shape(m,n,k,1));
  size_t na=size(filter_zeros(la)),nb=size(filter_zeros(lb));
  static_assert(sizeof(ElementA::ScaleFactorType)==1);
  cuda_ok(cudaMalloc(&sfa,na));cuda_ok(cudaMalloc(&sfb,nb));cuda_ok(cudaMemset(sfa,0x38,na));cuda_ok(cudaMemset(sfb,0x38,nb));
  auto args=arguments(A,B,D);cutlass_ok(gemm.can_implement(args));
  cuda_ok(cudaMalloc(&workspace,Gemm::get_workspace_size(args)));cuda_ok(cudaDeviceSynchronize());
 }
 Gemm::Arguments arguments(const void* A,const void* B,void* D){
  return {cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k,1},
   {(const ElementA::DataType*)A,sa,(const ElementB::DataType*)B,sb,sfa,la,sfb,lb},
   {{4.f,0.f},(const float*)D,sc,(float*)D,sd}};
 }
 ~Plan(){cudaFree(workspace);cudaFree(sfa);cudaFree(sfb);}
};
extern "C" const char* rq_error(){return error_message.c_str();}
extern "C" void* rq_fp4_create(int m,int n,int k,const void* A,const void* B,void* D){
 try{return new Plan(m,n,k,A,B,D);}catch(const std::exception& e){error_message=e.what();return nullptr;}
}
extern "C" int rq_fp4_run(Plan* p,const void* A,const void* B,void* D,cudaStream_t stream){
 try{cutlass_ok(p->gemm.run(p->arguments(A,B,D),p->workspace,stream));return 0;}
 catch(const std::exception& e){error_message=e.what();return 1;}
}
extern "C" void rq_fp4_destroy(Plan* p){delete p;}
extern "C" void* rq_i8_create(){cublasHandle_t h=nullptr;return cublasCreate(&h)==CUBLAS_STATUS_SUCCESS?h:nullptr;}
extern "C" int rq_i8_run(cublasHandle_t h,int m,int n,int k,const int8_t* A,const int8_t* B,int* D,cudaStream_t stream){
 const int one=1,zero=0;auto s=cublasSetStream(h,stream);if(s!=CUBLAS_STATUS_SUCCESS)return (int)s;
 return (int)cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&one,B,CUDA_R_8I,k,A,CUDA_R_8I,k,&zero,D,CUDA_R_32I,n,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT);
}
extern "C" void rq_i8_destroy(cublasHandle_t h){cublasDestroy(h);}
