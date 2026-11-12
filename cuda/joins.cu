// Join all exact INT8 limb products once, retaining signed-INT64 bounds.
extern "C" __global__ void join_i8(const int* partial,long long* out,const long long* weights,
                                  int count,int limbs,long long p){
    int index=blockIdx.x*256+threadIdx.x;if(index>=count)return;
    long long result=0;
    for(int i=0;i<limbs;i++)for(int j=0;j<limbs;j++){
        long long value=partial[((size_t)i*limbs+j)*count+index]%p;
        if(value<0)value+=p;
        result=(result+value*weights[i+j])%p;
    }
    out[index]=result;
}
