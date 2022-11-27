
template <typename T = size_t>
unsigned char *pre_cal_host(unsigned char *hash_mask_table, size_t hash_mask_table_sz, size_t hash_mask_sz) {

    std::cout << "pre_cal_host" << std::endl;

    if (hash_mask_table == nullptr) {
        hash_mask_table = cpu_tool<unsigned char>.zero(hash_mask_table, hash_mask_sz * hash_mask_table_sz);
    }
    
    auto &eng = cpu_tool<unsigned int>.eng;
    std::uniform_int_distribution<unsigned int> dis(0, 123); // magic number
    for (size_t i = 0; i < hash_mask_table_sz; ++i) {
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

        // size_t c = 0;
        // unsigned char *hash_mask = hash_mask_table + i * 16;
        // for (size_t j = 0; j < 16; ++j) {
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
        // printf("%lu\n", c);
        
    }

    return hash_mask_table;

}

