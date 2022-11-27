#pragma once

namespace qsketch {

// __device__ __host__ inline bool get_hashmask(uchar hash_mask, uint id);
// __device__ __host__ inline bool set_hashmask(uchar hash_mask, uint id);

uchar *generate_hashmask(uchar *hash_mask_table, size_t n, size_t m, 
    size_t hash_mask_ones = default_values::HASH_MASK_ONES, size_t padding = 0
    uint *index_hash_mask_table = nullptr, bool need_index = false);

uchar *resize_hashmask(uchar *in, size_t sz, 
    uchar *out, int times);
    // sz : size of input hash mask table, ( in bytes)
    // times: 
    // 0, -1, 1 -> size is unchanged.
    // positive number -> size is increased 
    // negetive number -> size is reduced
}



// #pragma once

// namespace qsketch {
// uchar *generate_hashmask(uchar *hash_mask_table, size_t n, size_t m, size_t hash_mask_ones = default_values::HASH_MASK_ONES);
// }
