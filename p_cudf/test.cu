#include "concurrent_unordered_map.cuh.h"

template<typename T>
__global__ void insert(T table) {
  int i = threadIdx.x;
  table->insert(thrust::pair<int, int>(i, i+1));
}

template<typename T>
__global__ void find(T* table){
  int i = threadIdx.x;
  cycle_iterator_adapter<thrust::pair<int, int>*> it = table->find(i);
  if(it.getter() == table->end().getter()) {
    printf("\nfuck you\n");
  } else{
    printf("\n%d,%d\n",i,it.getter()->second);
  }
  it = table->find(i+100);
  if(it.getter() == table->end().getter()) {
    printf("\nfuck you\n");
  } else{
    printf("\n%d,%d\n",i,it.getter()->second);
  }
}

int main() {
  auto table = new concurrent_unordered_map<int, int, -1>(136000, -1);
  insert<<<1, 10>>>(table);
  cudaDeviceSynchronize();
  find<<<1, 10>>>(
      const_cast<concurrent_unordered_map<int, int, -1> *>(table));
  cudaDeviceSynchronize();
}