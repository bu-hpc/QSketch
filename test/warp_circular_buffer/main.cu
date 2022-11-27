



using T = unsigned int;
using uint = unsigned int;
#define buf_size 128;

// __global__ void test(T *keys, size_t sz) {
    
//     __shared__ T[35];

//     for (size_t i = 0; i < sz; i += warpSize) {
//         if (keys[i] % 8 == 7) {
//             // add to buffer
//         }
//     }
// }

__global__ void reduce_buffer(T *keys, size_t sz) {
    
    __shared__ T buffer[buf_size];
    __shared__ uint id;
    if (threadIdx.x == 0) {
        id = 0;
    }

    for (size_t i = 0; i < sz; i += warpSize) {
        T k = keys[i];
        if (k % 8 == 7) {
            // add to buffer
            uint oid = atomicAdd(&id, 1);
            buffer[oid] = k;
        }
    }
}