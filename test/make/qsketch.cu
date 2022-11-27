#include "qsketch.h"

namespace qsketch {
const bool DEBUG_MODE = false;
const bool RANDOM_SEED = true;

__global__ void test_kernel() {
    uint tid = threadIdx.x;
    if (tid == 0) {
        if (DEBUG_MODE) {
            printf("DEBUG_MODE\n");
        } else {
            printf("NON DEBUG_MODE\n");
        }
        if (RANDOM_SEED) {
            printf("RANDOM_SEED\n");
        } else {
            printf("NON RANDOM_SEED\n");
        }
    }
}
}

