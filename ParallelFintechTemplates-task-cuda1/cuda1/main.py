import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.driver import Event

my_module = SourceModule("""
    __global__ void MatrixVectorMul(int height, int width, int* matrix, 
  int* vector, int* result){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
      if(tid < height){                    
        for(int i = 0; i < width; i++){                  
          result[tid] += (matrix[(tid * width) + i]);
        }                     
    }
  }


__global__ void GetGrade(int* clientSums, int A, int B, int length,
   int* creditGrade){
     int tid = threadIdx.x +blockIdx.x * blockDim.x;

     if(tid < length){
       if(clientSums[tid] < A){
         creditGrade[tid] = 0;
       } 
       else if(clientSums[tid] >= A && clientSums[tid] < B){
         creditGrade[tid] = 1;
       }
       else {
         creditGrade[tid] = 2;
       }
     }
  }

""")


if __name__ == '__main__':
    h_matrix = np.random.randint(0, 2, size=(10, 10))
    h_vector = np.random.randint(-100, 100, size=(10))
    h_result = np.zeros(10, dtype=np.int32)
    #Аллокация памяти на девайсе
    d_matrix = cuda.mem_alloc(h_matrix.nbytes)
    d_vector = cuda.mem_alloc(h_vector.nbytes)
    d_result = cuda.mem_alloc(h_result.nbytes)
    #Копируем данные с хоста на девайс
    cuda.memcpy_htod(d_matrix, h_matrix.ravel())
    cuda.memcpy_htod(d_vector, h_vector)
    ! nvcc main.cu
    kernel_first = my_module.get_function("MatrixVectorMul")
    kernel_first(np.int32(10), np.int32(10), d_matrix, d_vector, d_result, block=(256, 1, 1), grid=(1, 1, 1))
    cuda.memcpy_dtoh(h_result, d_result)
    d_clientSums = cuda.mem_alloc(h_result.nbytes)
    h_creditGrade = np.zeros(10, dtype=np.int32)
    d_creditGrade = cuda.mem_alloc(h_creditGrade.nbytes)cuda.memcpy_htod(d_clientSums, h_result)
    kernel_second = my_module.get_function("GetGrade")
    kernel_second(d_clientSums, np.int32(2), np.int32(50), np.int32(h_result.size), d_creditGrade, block=(256, 1, 1), grid=(1, 1, 1))
    cuda.memcpy_dtoh(h_result, d_clientSums)
