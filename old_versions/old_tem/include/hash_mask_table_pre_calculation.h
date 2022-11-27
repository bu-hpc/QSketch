
template <typename T = size_t>
unsigned char *pre_cal_host(unsigned char *hash_mask_table, size_t hash_mask_table_sz, size_t hash_mask_sz) {

    // std::cout << "pre_cal_host" << std::endl;

    if (hash_mask_table == nullptr) {
        hash_mask_table = cpu_tool<unsigned char>.zero(hash_mask_table, hash_mask_sz * hash_mask_table_sz);
    }
    
    auto &eng = cpu_tool<unsigned int>.eng;
    std::uniform_int_distribution<unsigned int> dis(0, 123); // magic number
    for (size_t i = 0; i < hash_mask_table_sz; i ++) { // err i++, need to fix
        size_t j = 0; 
        while (j < HASH_MASK_ONES) {
            unsigned int id = dis(eng);
            // std::cout << id << std::endl;
            unsigned int cid = id / 8;
            unsigned int bid = id % 8;
            if ((hash_mask_table[i * 16 + cid] & (1 << bid)) == 0) 
            {
                hash_mask_table[i * 16 + cid] |= (1 << bid);
                j++;
            } else {
                // std::cout << std::bitset<8>(hash_mask_table[i * 16 + cid]) << ":" << bid << std::endl;
            }
            // std::cout << j << std::endl;
        }
 
        // if ((hash_mask_table[i * 16 + 15] & 0b11110000) != 0) {
        //     std::cout << "err: " << std::bitset<8>(hash_mask_table[i * 16 + 15]) << std::endl;
        // }

        size_t c = 0;
        unsigned char *hash_mask = hash_mask_table + i * 16;
        for (size_t j = 0; j < 16; ++j) {
            unsigned char hm = hash_mask[j];
            while (hm != 0) {
                if (hm & 1) {
                    c++;
                }
                hm = hm >> 1;
            }
        }
        // if (c == 0) {
        //     printf("err c\n");
        // }
        printf("%lu\n", c);
        
    }

    return hash_mask_table;

}


template <typename T = size_t>
unsigned char *pre_cal_host_sub_warp(unsigned char *hash_mask_table, size_t hash_mask_table_sz, size_t hash_mask_sz) {

    // std::cout << "pre_cal_host" << std::endl;
    // std::cout << "hash_mask_table_sz: " << hash_mask_table_sz << std::endl;
    // std::cout << "hash_mask_sz: " << hash_mask_sz << std::endl;
    
    if (hash_mask_table == nullptr) {
        hash_mask_table = cpu_tool<unsigned char>.zero(hash_mask_table, hash_mask_sz * hash_mask_table_sz);
    }
    
    auto &eng = cpu_tool<unsigned int>.eng;
    std::uniform_int_distribution<unsigned int> dis(0, 27); // magic number
    std::uniform_int_distribution<unsigned int> dis_last(0, 31); // magic number
    
    for (size_t i = 0; i < hash_mask_table_sz; i ++) {
        size_t j = 0; 
        // while (j < HASH_MASK_ONES - 1) 
        while (j < HASH_MASK_ONES) 
       {
            unsigned int id = dis(eng);
            unsigned int cid = id / 8;
            unsigned int bid = id % 8;
            unsigned char b = 1;
            if ((hash_mask_table[i *  hash_mask_sz + cid] & (b << bid)) == 0) 
            {
                // std::cout << i << "," << id << "," << bid << "," << cid << ";";
                // std::cout << (hash_mask_table[i *  hash_mask_sz + cid] & (b << bid)) << ",";
                hash_mask_table[i * hash_mask_sz + cid] |= (b << bid);
                // std::cout << (hash_mask_table[i *  hash_mask_sz + cid] & (b << bid)) << ",";
                j++;
            }
            // std::cout << j << ",";
        }
        // std::cout << std::endl;
 
        // while (j < HASH_MASK_ONES) {
        //     unsigned int id = dis_last(eng);
        //     unsigned int cid = id / 8;
        //     unsigned int bid = id % 8;
        //     if ((hash_mask_table[i * hash_mask_sz + cid] & (1 << bid)) == 0) 
        //     {
        //         hash_mask_table[i * hash_mask_sz + cid] |= (1 << bid);
        //         j++;
        //     }
        // }

        // size_t c = 0;
        // unsigned char *hash_mask = hash_mask_table + i * hash_mask_sz;
        // for (size_t j = 0; j < hash_mask_sz; ++j) {
        //     unsigned char hm = hash_mask[j];
        //     while (hm != 0) {
        //         if (hm & 1) {
        //             c++;
        //         }
        //         hm = hm >> 1;
        //     }
        // }
        // if (c == 0) {
        //     printf("err c\n");
        // }

        // if (c != 2) {
        //     std::cout << "err: " << c << std::endl;
        //     unsigned char *hash_mask = hash_mask_table + i * hash_mask_sz;
        //     for (size_t j = 0; j < hash_mask_sz; ++j) {
        //         unsigned char hm = hash_mask[j];
        //         for (size_t k = 0; k < 8; ++k) {
        //             if (((hm >> k) & 1)) {
        //                 std::cout << "1";    
        //             } else {
        //                 std::cout << "0";   
        //             }
                    
        //         }
        //         std::cout << " ";
        //         // while (hm != 0) {
        //         //     if (hm & 1) {
        //         //         c++;
        //         //     }
        //         //     hm = hm >> 1;
        //         // }
        //     }
        //     std::cout << std::endl;
        // }


        // printf("%lu\n", c);
    }

    return hash_mask_table;
}

