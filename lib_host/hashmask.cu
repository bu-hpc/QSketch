#include <qsketch.h>
#include <bitset>
namespace qsketch {

bool set(uchar *hash_mask, uint id) {
    uint cid = id / bits<uchar>;
    uchar b = 1 << (id % bits<uchar>);

    // std::cout << id << ":" << std::bitset<8>(b)
        // std::bitset<8>(hash_mask[cid]) 
        // << std::endl;
    // std::cout << std::bitset<8>(hash_mask[cid] & b) << std::endl;
    if ((hash_mask[cid] & b) == 0) {
        hash_mask[cid] |= b;
        // std::cout << std::bitset<8>(hash_mask[cid]) << std::endl;
        return true;
    }

    return false;
}

bool get(uchar *hash_mask, uint id) {
    uint cid = id / bits<uchar>;
    uchar b = 1 << (id % bits<uchar>);
    return hash_mask[cid] & b;
}

__device__ __host__ inline bool get_hashmask(uchar hash_mask, uint id) {
    return false;
}


uchar *generate_hashmask(uchar *hash_mask_table, size_t n, size_t m, 
    size_t hash_mask_ones, size_t padding,
    uint **index_hash_mask_table_ptr) {
    /*
        n : the number of hash_mask
        m : the size of hash_mask, (bits)
        padding: the last bits which are zero
    */
    #ifdef QSKETCH_DEBUG
        std::cout << "generate_hashmask" << std::endl;
        std::cout << n << ", " << m << ", " << hash_mask_ones << ", " << padding << std::endl; 
    #endif
    // auto set = [](uchar *hash_mask, uint id) {
    //     uint cid = id / bits<uchar>;
    //     uchar b = 1 << (id % bits<uchar>);

    //     if (hash_mask[cid] & b == 0) {
    //         hash_mask[cid] |= b;
    //         return true;
    //     }

    //     return false;
    // };

    size_t hash_mask_sz = ceil<size_t>(m, bits<uchar>);

    // std::cout << hash_mask_sz << std::endl;

    if (hash_mask_table == nullptr) {
        // size_t mem_sz = ceil<size_t>(n * m, sizeof(uchar));
        size_t mem_sz = n * hash_mask_sz;
        hash_mask_table = cpu_tool<uchar>.zero(hash_mask_table, mem_sz);
        if (index_hash_mask_table_ptr != nullptr
            && *index_hash_mask_table_ptr != nullptr) {
            *index_hash_mask_table_ptr = cpu_tool<uint>.zero(nullptr, n * hash_mask_ones);
        }
    }

    auto &eng = cpu_tool<uint>.eng; 
    size_t lm = m;
    std::uniform_int_distribution<uint> dis(0, lm - padding - 1);

    for (size_t i = 0; i < n; ++i) {
        size_t j = 0;
        // std::cout << i << std::endl;
        while (j < hash_mask_ones) {
            // std::cout << j << std::endl;
            uint id = dis(eng);
            if (set(hash_mask_table + i * hash_mask_sz, id)) {
                j++;
                *((*index_hash_mask_table_ptr) + i * hash_mask_ones + j) = id;
            }

#ifdef QSKETCH_DEBUG

#endif

        }
#ifdef QSKETCH_DEBUG
        #if 0
        for (size_t j = 0; j < hash_mask_sz; ++j) {
            std::cout << std::bitset<8>(hash_mask_table[i * hash_mask_sz + j]) << " ";
        }
        std::cout << std::endl;
        #endif
#endif

    }

    return hash_mask_table;
}

uchar *resize_hash_mask(uchar *in, size_t sz, uchar *out, 
    const int times) {

    auto mul = [&times](uint id) {
        if (times > 0) {
            return id * times;
        } else if (times == 0) {
            return id;
        }
        return id / times;
    };
    
    if (times % bits<uchar> != 0) {
        std::cerr << "";
        return nullptr;
    }

    if (out == nullptr) {
        // out = new uchar [sz * times];
        out = new uchar[mul(sz)];
    }

    auto &eng = cpu_tool<uint>.eng; 
    std::uniform_int_distribution<uint> dis(0, std::max(times - 1, 0));

    for (uint i = 0; i < sz; ++i) {
        if (get(in, i)) {
            set(out, mul(i) + dis(eng));
        }
    }

    return out;
}

}