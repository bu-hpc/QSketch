#pragma once
template <typename Key_T, typename Count_T, typename Hashed_T, typename Seed_T>
struct Device_vs_Host_Test {

    Tools<Key_T> &tool_host;
    Sketch<Key_T, Count_T> &map_host;
    Sketch<Key_T, Count_T> &emap_host;
    Sketch<Key_T, Count_T> &map_device;

    Device_vs_Host_Test(Tools<Key_T> &t, Sketch<Key_T, Count_T> &m, Sketch<Key_T, Count_T> &em, Sketch<Key_T, Count_T> &md)
    : tool_host(t), map_host(m), emap_host(em), map_device(md) {
    }

    int base_test() {
        return 0;
    }

    void compare() {
        // checkKernelErrors(cudaMalloc(&keys_device, sizeof(Key_T) * keys_sz));
        // checkKernelErrors(cudaMemcpy(keys_device, keys, sizeof(Key_T) * keys_sz, cudaMemcpyHostToDevice));
        // assert(map_host.seed_sz == map_device.seed_sz);
        // assert(map_host.table_total_sz == map_device.table_total_sz);

        // Count_Min_Sketch_GPU_Host_Sim<>
    }

    template <typename Sketch_Host, typename Sketch_Device>
    void compare() {
        Sketch_Host &sh = dynamic_cast<Sketch_Host&>(map_host);
        Sketch_Device &sd = dynamic_cast<Sketch_Device&>(map_device);

        assert(sh.seed_total_sz == sd.seed_total_sz);
        assert(sh.table_total_sz == sd.table_total_sz);

        std::cout << "host: " << std::endl;
        std::cout << "seed_total_sz: " << sh.seed_total_sz << std::endl;
        std::cout << "table_total_sz: " << sh.table_total_sz << std::endl;


        std::cout << "device: " << std::endl;
        std::cout << "seed_total_sz: " << sd.seed_total_sz << std::endl;
        std::cout << "table_total_sz: " << sd.table_total_sz << std::endl;


        
        Seed_T *seeds_from_device = new Seed_T[sd.seed_total_sz];
        checkKernelErrors(cudaMemcpy(seeds_from_device, sd.seed, sizeof(Seed_T) * sd.seed_total_sz, cudaMemcpyDeviceToHost));

        {
            Seed_T *hp = sh.seed;
            Seed_T *dp = seeds_from_device;
            for (size_t i = 0; i < sd.seed_total_sz; ++i) {
                // std::cout << i << std::endl;
                // std::cout << hp[i] << std::endl;
                // std::cout << dp[i] << std::endl;
                
                if (hp[i] != dp[i]) {
                    std::cout << "err: " << std::endl;
                    return;
                }
            }
        }

        delete []seeds_from_device;


        Count_T *counts_from_device = new Count_T[sd.table_total_sz];
        checkKernelErrors(cudaMemcpy(counts_from_device, sd.table, sizeof(Count_T) * sd.table_total_sz, cudaMemcpyDeviceToHost));

        {
            Count_T *hp = sh.table;
            Count_T *dp = counts_from_device;

            size_t per = sd.table_total_sz / 100;

            for (size_t i = 0; i < sd.table_total_sz; ++i) {

                if (i % per == 0) {
                    std::cout << i / per << "%" << std::endl;
                }

                // std::cout << i << std::endl;
                // std::cout << hp[i] << std::endl;
                // std::cout << dp[i] << std::endl;
                
                if (hp[i] != dp[i]) {
                    static size_t c = 0;
                    std::cout << "c: " << (++c) << std::endl;
                    std::cout << "i: " << i << std::endl;
                    std::cout << hp[i] << std::endl;
                    std::cout << dp[i] << std::endl;
                    std::cout << "err: " << std::endl;
                    // return;
                }
            }
        }

        delete []counts_from_device;


        // Count_T *counts_from_device;
        // checkKernelErrors(cudaMalloc(&counts_from_device, sizeof(Count_T) * sd.table_total_sz));
        // checkKernelErrors(cudaMemcpy(counts_from_device, sd.table, sizeof(Count_T) * sd.table_total_sz, cudaMemcpyHostToDevice));

        // {
        //     auto hp = sh.table;
        //     auto dp = sd.table;
        //     for (size_t i = 0; i < sd.table_total_sz; ++i) {
                
        //     }
        // }

    }

    template <typename Sketch_Host, typename Sketch_Device>
    int insert_test(size_t keys_sz, size_t insert_loop = 1) {

        Sketch_Host &sh = dynamic_cast<Sketch_Host&>(map_host);
        Sketch_Device &sd = dynamic_cast<Sketch_Device&>(map_device);

        Key_T *keys = nullptr;
        // keys = tool_host.random(keys, keys_sz);

        keys = new Key_T[keys_sz];

        {
            std::default_random_engine eng;
            std::uniform_int_distribution<Key_T> dis(std::numeric_limits<Key_T>::min(), std::numeric_limits<Key_T>::max());
            std::unordered_set<Key_T> us;
            while(us.size() < keys_sz) {
                us.insert(dis(eng));
            }
            size_t i = 0;
            for (auto &val : us) {
                keys[i++] = val;
            }
        }

        {
            std::unordered_map<Key_T, Count_T> um;
            for (size_t i = 0; i < keys_sz; ++i)
            {
                um[keys[i]]++;
            }

            std::cout << "keys_sz: " << keys_sz << std::endl;
            std::cout << "um.size: " << um.size() << std::endl;
        }
        

        Key_T *keys_device = nullptr;
        checkKernelErrors(cudaMalloc(&keys_device, sizeof(Key_T) * keys_sz));
        checkKernelErrors(cudaMemcpy(keys_device, keys, sizeof(Key_T) * keys_sz, cudaMemcpyHostToDevice));

        checkKernelErrors(cudaMemcpy(sd.seed, sh.seed, sizeof(Seed_T) * sh.seed_total_sz, cudaMemcpyHostToDevice));

        for (size_t i = 0; i < insert_loop; ++i) {
            // std::cout << "insert " << i << std::endl;

            map_host.insert(keys, keys_sz);
            map_device.insert(keys_device, keys_sz);

            compare<Sketch_Host, Sketch_Device>();

            // double ts = get_run_time([&]() {
            //     map.insert(keys, keys_sz);
            // });
            // tts += ts;
            // std::cout << "insert done" << i << std::endl;
        }
        // tool.free(keys);

        tool_host.free(keys);
        cudaFree(keys_device);

        return 0;
    }

};


template <typename Key_T, typename Count_T, typename Hashed_T = size_t>
struct Test {
    Tools<Key_T> &tool;
    Sketch<Key_T, Count_T> &map;
    Sketch<Key_T, Count_T> &emap;

    bool um_insert = false;

    Test(Tools<Key_T> &t, Sketch<Key_T, Count_T> &m, Sketch<Key_T, Count_T> &em) : tool(t), map(m), emap(em) {
    }

    void clear() {
        map.clear();
        emap.clear();
    }

    // base tests

    int base_test() {
        return 0;
    }

    // perf test

    int insert_perf_test(size_t keys_sz, size_t insert_loop = 1, std::ostream &os = std::cout) {

        Key_T *keys = nullptr;
        keys = tool.random(keys, keys_sz);
        map.pre_cal(keys, std::min<size_t>(HASH_MASK_TABLE_SIZE, keys_sz), nullptr);
        double tts = 0;
        for (size_t i = 0; i < insert_loop; ++i) {
            // std::cout << "insert " << i << std::endl;
            double ts = get_run_time([&]() {
                map.insert(keys, keys_sz);
            });
            tts += ts;
            // std::cout << "insert done" << i << std::endl;
        }

        size_t insert_number = keys_sz * insert_loop;
        os << "insert perf: " << format_perf(insert_number / tts) << std::endl;
        // std::cout << format_perf(insert_number / tts) << std::endl;


        // cudaFree(keys);
        tool.free(keys);
        return 0;
    }

    int insert_perf_freq_test(size_t keys_sz, size_t insert_loop = 1, std::ostream &os = std::cout) {

        Key_T *keys = nullptr;
        keys = tool.random_freq(keys, keys_sz);
        map.pre_cal(keys, std::min<size_t>(HASH_MASK_TABLE_SIZE, keys_sz), nullptr);
        double tts = 0;
        for (size_t i = 0; i < insert_loop; ++i) {
            // std::cout << "insert " << i << std::endl;
            double ts = get_run_time([&]() {
                map.insert(keys, keys_sz);
            });
            tts += ts;
            // std::cout << "insert done" << i << std::endl;
        }

        size_t insert_number = keys_sz * insert_loop;
        os << "insert perf: " << format_perf(insert_number / tts) << std::endl;
        // std::cout << format_perf(insert_number / tts) << std::endl;


        // cudaFree(keys);
        tool.free(keys);
        return 0;
    }

    int search_perf_test(size_t insert_keys_sz, size_t search_keys_sz,
     size_t search_loop = 1, std::ostream &os = std::cout) {

        assert(insert_keys_sz >= search_keys_sz);

        Key_T *keys = nullptr;// = new Key_T[keys_sz];
        Count_T *counts = nullptr;

        double tts = 0;
        // long tdf = 0; // total difference between the accuracy counts and map counts

        keys = tool.random(keys, insert_keys_sz);
        map.pre_cal(keys, std::min<size_t>(HASH_MASK_TABLE_SIZE, insert_keys_sz), nullptr);
        map.insert(keys, insert_keys_sz);

        for (size_t i = 0; i < search_loop; ++i) {

            tool.random_shuffle(keys, search_keys_sz);
            counts = tool.zero(counts, search_keys_sz);

            double ts = get_run_time([&](){
                map.search(keys, search_keys_sz, counts);
            });
            tts += ts;

        }


        // thrust::device_vector<Count_T> dv1(counts, counts + search_keys_sz);
        // Count_T s = sum(dv1);
        // std::cout << "sum: " << s << std::endl;
        // std::cout << "tts: " << tts << std::endl;

        size_t search_number = search_keys_sz * search_loop;
        // os << "search perf time: " << tts << std::endl;
        os << "search perf: " << format_perf(search_number / tts) << std::endl;

        // cudaFree(keys);
        // cudaFree(counts);
        tool.free(keys);
        tool.free(counts);
        return 0;
    }

    int search_perf_freq_test(size_t insert_keys_sz, size_t search_keys_sz,
     size_t search_loop = 1, std::ostream &os = std::cout) {

        assert(insert_keys_sz >= search_keys_sz);

        Key_T *keys = nullptr;// = new Key_T[keys_sz];
        Count_T *counts = nullptr;

        double tts = 0;
        // long tdf = 0; // total difference between the accuracy counts and map counts

        keys = tool.random_freq(keys, insert_keys_sz);
        map.pre_cal(keys, std::min<size_t>(HASH_MASK_TABLE_SIZE, insert_keys_sz), nullptr);
        map.insert(keys, insert_keys_sz);

        for (size_t i = 0; i < search_loop; ++i) {

            tool.random_shuffle(keys, search_keys_sz);
            counts = tool.zero(counts, search_keys_sz);

            double ts = get_run_time([&](){
                map.search(keys, search_keys_sz, counts);
            });
            tts += ts;

        }


        // thrust::device_vector<Count_T> dv1(counts, counts + search_keys_sz);
        // Count_T s = sum(dv1);
        // std::cout << "sum: " << s << std::endl;
        // std::cout << "tts: " << tts << std::endl;

        size_t search_number = search_keys_sz * search_loop;
        // std::cout << "search perf time: " << tts << std::endl;
        std::cout << "search perf: " << format_perf(search_number / tts) << std::endl;

        // cudaFree(keys);
        // cudaFree(counts);
        tool.free(keys);
        tool.free(counts);
        return 0;
    }


    int search_accuracy_test(size_t insert_keys_sz, size_t search_keys_sz,
     size_t search_loop = 1, std::ostream &os = std::cout) {

        assert(insert_keys_sz >= search_keys_sz);

        Key_T *keys = nullptr;// = new Key_T[keys_sz];
        Count_T *counts = nullptr;
        Count_T *counts_2 = nullptr;
        Count_T *e_counts = nullptr;
        size_t *debug = gpu_tool<size_t>.zero(nullptr, insert_keys_sz);

        // double tts = 0;
        // long tdf = 0; // total difference between the accuracy counts and map counts

        keys = tool.random(keys, insert_keys_sz);
                    // std::cout << "p0" << std::endl;
        unsigned char *db1 = map.pre_cal(keys, std::min<size_t>(HASH_MASK_TABLE_SIZE, insert_keys_sz), debug);
        unsigned char *hb1 = new unsigned char[insert_keys_sz * 16];
        unsigned char *hb2 = new unsigned char[insert_keys_sz * 16];
        size_t *hdebug = new size_t[insert_keys_sz];

        // checkKernelErrors(cudaMemcpy(hb1, db1, insert_keys_sz * 16, cudaMemcpyDeviceToHost));
        checkKernelErrors(cudaMemcpy(hdebug, debug, insert_keys_sz * sizeof(size_t), cudaMemcpyDeviceToHost));


        keys = tool.random(keys, insert_keys_sz);

        Key_T *hk1 = new Key_T[insert_keys_sz];
        Key_T *hk2 = new Key_T[insert_keys_sz];
        checkKernelErrors(cudaMemcpy(hk1, keys, insert_keys_sz * sizeof(Key_T), cudaMemcpyDeviceToHost));


        Count_T *hc = new Count_T[search_keys_sz];
        Count_T *hec = new Count_T[search_keys_sz];




        map.insert(keys, insert_keys_sz);
                    // std::cout << "p1" << std::endl;

        emap.insert(keys, insert_keys_sz);
                    // std::cout << "p2" << std::endl;

        // std::cout << "-------------------------------------" << std::endl;
        for (size_t i = 0; i < search_loop; ++i) {

            // std::random_shuffle(keys, keys + search_keys_sz);
            // tool.random_shuffle(keys, search_keys_sz);
            counts = tool.zero(counts, search_keys_sz);
            map.search(keys, search_keys_sz, counts);

            counts_2 = tool.zero(counts_2, search_keys_sz);
            map.search(keys, search_keys_sz, counts_2);

            // std::cout << "lp0" << std::endl;
            e_counts = tool.zero(e_counts, search_keys_sz);
            emap.search(keys, search_keys_sz, e_counts);

            checkKernelErrors(cudaMemcpy(hk2, keys, insert_keys_sz * sizeof(Key_T), cudaMemcpyDeviceToHost));
            checkKernelErrors(cudaMemcpy(hc, counts, search_keys_sz * sizeof(Count_T), cudaMemcpyDeviceToHost));
            checkKernelErrors(cudaMemcpy(hec, e_counts, search_keys_sz * sizeof(Count_T), cudaMemcpyDeviceToHost));
            // checkKernelErrors(cudaMemcpy(hb2, db1, insert_keys_sz * 16, cudaMemcpyDeviceToHost));
            // std::cout << "lp1" << std::endl;

            // thrust::device_vector<Count_T> dv1(counts, counts + search_keys_sz);
            // thrust::host_vector<Count_T> hv1(dv1.begin(), dv1.end());
            // thrust::device_vector<Count_T> dv2(e_counts, e_counts + search_keys_sz);
            // thrust::host_vector<Count_T> hv2(dv2.begin(), dv2.end());
            // std::cout << "p2" << std::endl;

            // for (auto v : hv1) {
            //     std::cout << v << "; ";
            // }
            // std::cout << std::endl;

            // for (auto v : hv2) {
            //     std::cout << v << "; ";
            // }
            // std::cout << std::endl;

            // auto dc2 = diff(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
            //     thrust::device_vector<Count_T>(counts_2, counts_2 + search_keys_sz));
            // auto max2 = max_diff(dc2);
            // auto ave2 = average_diff(dc2);
            // auto greater_counts2 = greater(dc2);
            // std::cout << "max2: " << max2 << "\tave2: " << ave2 << std::endl;
            // std::cout << "greater_counts2: " << greater_counts2 << std::endl;

            auto dc3 = diff(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
                thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
            auto max = max_diff(dc3);
            auto ave = average_diff(dc3);
            auto greater_counts = greater(dc3);

            auto dc3_p = percentage(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
                thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
            auto max_p = max_diff(dc3_p);
            auto ave_p = average_diff(dc3_p);

            // Count_T *c3 = diff(counts, e_counts, search_keys_sz);
            // Count_T max = max_diff(c3, search_keys_sz);
            // double ave = average_diff(c3, search_keys_sz);
            // // percentage
            // double *cp = percentage(counts, e_counts, search_keys_sz);
            // double maxp = max_diff(cp, search_keys_sz);
            // double avep = average_diff(cp, search_keys_sz);

            // if (DEBUG) {
            //     for (size_t j = 0; j < search_keys_sz; ++j) {
            //         debug_log << keys[j] << "\t:" << counts[j] << ", " << e_counts[j] << std::endl;
            //     }
            // }

            // std::cout << "search " << i << " accuracy: " << std:: endl;
            std::cout << "max: " << max << "\tave: " << ave << std::endl;
            std::cout << "maxp: " << max_p << "%\tavep: " << ave_p << "%" << std::endl;
            std::cout << "greater_counts: " << greater_counts << std::endl;



        }
        cudaDeviceSynchronize();


        // for (size_t i = 0; i < insert_keys_sz * 16; ++i) {
        //     if (hb1[i] != hb2[i]) {
        //         std::cout << "err" << std::endl;
        //         break;
        //     }
        // }
        // std::cout << "pass 1" << std::endl;
        // for (size_t i = 0; i < insert_keys_sz; ++i)
        // {
        //     if (hk1[i] != hk2[i]) {
        //         std::cout << "err2" << std::endl;
        //         break;
        //     }
        // }
        // std::cout << "pass 2" << std::endl;

        // for (size_t i = 0; i < search_keys_sz; ++i) {
        //     if (hc[i] < hec[i]) {
        //         // std::cout << "err3" << std::endl;
        //         // break;
        //         std::cout << i << "\t<--" << hdebug[i] << " \t\t\t--> " << hk1[i] << " : " << hc[i] << ", " << hec[i] << std::endl;
        //     }
        // }
        // std::cout << "pass 3" << std::endl;

        // for (size_t i = 0; i < search_keys_sz; ++i) {
        //     std::cout << hdebug[i] << std::endl;
        // }


        delete hb1;
        delete hb2;
        delete hk1;
        delete hk2;
        delete hc;
        delete hec;
        
        tool.free(keys);
        tool.free(counts);
        tool.free(e_counts);
        return 0;
    }

    int search_accuracy_test_file_debug(Key_T *keys, size_t insert_keys_sz, size_t search_keys_sz) {
        assert(insert_keys_sz >= search_keys_sz);
        
        Count_T *counts = nullptr;
        Count_T *e_counts = nullptr;
        size_t *debug = nullptr;

        unsigned char *db1 = map.pre_cal(keys, std::min<size_t>(HASH_MASK_TABLE_SIZE, insert_keys_sz), debug);
        map.insert(keys, insert_keys_sz);
        emap.insert(keys, insert_keys_sz);

        counts = tool.zero(counts, search_keys_sz);
        map.search(keys, search_keys_sz, counts);

        e_counts = tool.zero(e_counts, search_keys_sz);
        emap.search(keys, search_keys_sz, e_counts);

        auto dc3 = diff(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
            thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
        auto max = max_diff(dc3);
        auto ave = average_diff(dc3);
        auto greater_counts = greater(dc3);

        auto dc3_p = percentage(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
            thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
        auto max_p = max_diff(dc3_p);
        auto ave_p = average_diff(dc3_p);

        std::cout << "max: " << max << "\tave: " << ave << std::endl;
        std::cout << "maxp: " << max_p << "%\tavep: " << ave_p << "%" << std::endl;
        std::cout << "greater_counts: " << greater_counts << std::endl;
        
        map.clear();
        emap.clear();

        return 0;
    }

    int search_accuracy_test_debug(size_t insert_keys_sz, size_t search_keys_sz,
     size_t search_loop = 1, std::ostream &os = std::cout) {

        assert(insert_keys_sz >= search_keys_sz);

        Key_T *keys = nullptr;// = new Key_T[keys_sz];
        Count_T *counts = nullptr;
        Count_T *counts_2 = nullptr;
        Count_T *e_counts = nullptr;
        size_t *debug = gpu_tool<size_t>.zero(nullptr, insert_keys_sz);

        // double tts = 0;
        // long tdf = 0; // total difference between the accuracy counts and map counts

        keys = tool.random(keys, insert_keys_sz);
                    // std::cout << "p0" << std::endl;
        unsigned char *db1 = map.pre_cal(keys, std::min<size_t>(HASH_MASK_TABLE_SIZE, insert_keys_sz), debug);
        unsigned char *hb1 = new unsigned char[insert_keys_sz * 16];
        unsigned char *hb2 = new unsigned char[insert_keys_sz * 16];
        size_t *hdebug = new size_t[insert_keys_sz];

        checkKernelErrors(cudaMemcpy(hb1, db1, std::min<size_t>(HASH_MASK_TABLE_SIZE, insert_keys_sz) * 16, cudaMemcpyDeviceToHost));
        checkKernelErrors(cudaMemcpy(hdebug, debug, insert_keys_sz * sizeof(size_t), cudaMemcpyDeviceToHost));


        keys = tool.random(keys, insert_keys_sz);

        Key_T *hk1 = new Key_T[insert_keys_sz];
        Key_T *hk2 = new Key_T[insert_keys_sz];
        checkKernelErrors(cudaMemcpy(hk1, keys, insert_keys_sz * sizeof(Key_T), cudaMemcpyDeviceToHost));


        Count_T *hc = new Count_T[search_keys_sz];
        Count_T *hec = new Count_T[search_keys_sz];

        size_t insert_loop = 1;
        for (size_t j = 0; j < insert_loop; ++j) {
            std::cout << "test: " << j << std::endl;
            map.insert(keys, insert_keys_sz);
            emap.insert(keys, insert_keys_sz);

            counts = tool.zero(counts, search_keys_sz);
            map.search(keys, search_keys_sz, counts);

            e_counts = tool.zero(e_counts, search_keys_sz);
            emap.search(keys, search_keys_sz, e_counts);

            checkKernelErrors(cudaMemcpy(hk2, keys, insert_keys_sz * sizeof(Key_T), cudaMemcpyDeviceToHost));
            checkKernelErrors(cudaMemcpy(hc, counts, search_keys_sz * sizeof(Count_T), cudaMemcpyDeviceToHost));
            checkKernelErrors(cudaMemcpy(hec, e_counts, search_keys_sz * sizeof(Count_T), cudaMemcpyDeviceToHost));

            for (size_t i = 0; i < search_keys_sz; ++i) {
                if (hc[i] < hec[i]) {
                    // std::cout << "err3" << std::endl;
                    // break;
                    std::cout << i << "\t<--" << hdebug[i] << " \t\t\t--> " << hk1[i] << "\t:\t";// << hc[i] << ", " << hec[i] << std::endl;
                    for (size_t j = 0; j < 16; ++j) {
                        auto c = hb1[i * 16 + j];
                        std::bitset<8> bs(c);
                        std::cout << bs << " ";
                    }
                    std::cout << std::endl;
                }
            }

            auto dc3 = diff(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
                thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
            auto max = max_diff(dc3);
            auto ave = average_diff(dc3);
            auto greater_counts = greater(dc3);

            auto dc3_p = percentage(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
                thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
            auto max_p = max_diff(dc3_p);
            auto ave_p = average_diff(dc3_p);

            std::cout << "max: " << max << "\tave: " << ave << std::endl;
            std::cout << "maxp: " << max_p << "%\tavep: " << ave_p << "%" << std::endl;
            std::cout << "greater_counts: " << greater_counts << std::endl;
            
            map.clear();
            emap.clear();
        }

        return 0;


        map.insert(keys, insert_keys_sz);
                    // std::cout << "p1" << std::endl;

        emap.insert(keys, insert_keys_sz);
                    // std::cout << "p2" << std::endl;

        // std::cout << "-------------------------------------" << std::endl;
        for (size_t i = 0; i < search_loop; ++i) {

            // std::random_shuffle(keys, keys + search_keys_sz);
            // tool.random_shuffle(keys, search_keys_sz);
            counts = tool.zero(counts, search_keys_sz);
            map.search(keys, search_keys_sz, counts);

            counts_2 = tool.zero(counts_2, search_keys_sz);
            map.search(keys, search_keys_sz, counts_2);

            // std::cout << "lp0" << std::endl;
            e_counts = tool.zero(e_counts, search_keys_sz);
            emap.search(keys, search_keys_sz, e_counts);

            checkKernelErrors(cudaMemcpy(hk2, keys, insert_keys_sz * sizeof(Key_T), cudaMemcpyDeviceToHost));
            checkKernelErrors(cudaMemcpy(hc, counts, search_keys_sz * sizeof(Count_T), cudaMemcpyDeviceToHost));
            checkKernelErrors(cudaMemcpy(hec, e_counts, search_keys_sz * sizeof(Count_T), cudaMemcpyDeviceToHost));
            checkKernelErrors(cudaMemcpy(hb2, db1, std::min<size_t>(HASH_MASK_TABLE_SIZE, insert_keys_sz) * 16, cudaMemcpyDeviceToHost));
            // std::cout << "lp1" << std::endl;

            // thrust::device_vector<Count_T> dv1(counts, counts + search_keys_sz);
            // thrust::host_vector<Count_T> hv1(dv1.begin(), dv1.end());
            // thrust::device_vector<Count_T> dv2(e_counts, e_counts + search_keys_sz);
            // thrust::host_vector<Count_T> hv2(dv2.begin(), dv2.end());
            // std::cout << "p2" << std::endl;

            // for (auto v : hv1) {
            //     std::cout << v << "; ";
            // }
            // std::cout << std::endl;

            // for (auto v : hv2) {
            //     std::cout << v << "; ";
            // }
            // std::cout << std::endl;

            auto dc2 = diff(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
                thrust::device_vector<Count_T>(counts_2, counts_2 + search_keys_sz));
            auto max2 = max_diff(dc2);
            auto ave2 = average_diff(dc2);
            auto greater_counts2 = greater(dc2);
            std::cout << "max2: " << max2 << "\tave2: " << ave2 << std::endl;
            std::cout << "greater_counts2: " << greater_counts2 << std::endl;

            auto dc3 = diff(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
                thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
            auto max = max_diff(dc3);
            auto ave = average_diff(dc3);
            auto greater_counts = greater(dc3);

            auto dc3_p = percentage(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
                thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
            auto max_p = max_diff(dc3_p);
            auto ave_p = average_diff(dc3_p);

            // Count_T *c3 = diff(counts, e_counts, search_keys_sz);
            // Count_T max = max_diff(c3, search_keys_sz);
            // double ave = average_diff(c3, search_keys_sz);
            // // percentage
            // double *cp = percentage(counts, e_counts, search_keys_sz);
            // double maxp = max_diff(cp, search_keys_sz);
            // double avep = average_diff(cp, search_keys_sz);

            // if (DEBUG) {
            //     for (size_t j = 0; j < search_keys_sz; ++j) {
            //         debug_log << keys[j] << "\t:" << counts[j] << ", " << e_counts[j] << std::endl;
            //     }
            // }

            // std::cout << "search " << i << " accuracy: " << std:: endl;
            std::cout << "max: " << max << "\tave: " << ave << std::endl;
            std::cout << "maxp: " << max_p << "%\tavep: " << ave_p << "%" << std::endl;
            std::cout << "greater_counts: " << greater_counts << std::endl;



        }
        cudaDeviceSynchronize();


        for (size_t i = 0; i < std::min<size_t>(HASH_MASK_TABLE_SIZE, insert_keys_sz) * 16; ++i) {
            if (hb1[i] != hb2[i]) {
                std::cout << "err" << std::endl;
                break;
            }
        }
        std::cout << "pass 1" << std::endl;
        for (size_t i = 0; i < insert_keys_sz; ++i)
        {
            if (hk1[i] != hk2[i]) {
                std::cout << "err2" << std::endl;
                break;
            }
        }
        std::cout << "pass 2" << std::endl;

        for (size_t i = 0; i < search_keys_sz; ++i) {
            if (hc[i] < hec[i]) {
                // std::cout << "err3" << std::endl;
                // break;
                std::cout << i << "\t<--" << hdebug[i] << " \t\t\t--> " << hk1[i] << " : " << hc[i] << ", " << hec[i] << std::endl;
            }
        }
        std::cout << "pass 3" << std::endl;

        // for (size_t i = 0; i < search_keys_sz; ++i) {
        //     std::cout << hdebug[i] << std::endl;
        // }


        delete hb1;
        delete hb2;
        delete hk1;
        delete hk2;
        delete hc;
        delete hec;
        
        tool.free(keys);
        tool.free(counts);
        tool.free(e_counts);
        return 0;
    }

    int search_accuracy_test_large(size_t insert_keys_sz, size_t search_keys_sz) {

        // assert(insert_keys_sz >= search_keys_sz);

        // Key_T *keys = nullptr;// = new Key_T[keys_sz];
        // Count_T *counts = nullptr;
        // Count_T *e_counts = nullptr;

        // // double tts = 0;
        // // long tdf = 0; // total difference between the accuracy counts and map counts

        // static size_t batch_size = 1024 * 1024;
        // size_t rmi = insert_keys_sz;
        // size_t rms = search_keys_sz;

        // std::vector<int> search_batch_ids;
        // for (int i = 0; i < ceil<size_t>(insert_keys_sz, batch_size); ++i) {
        //     search_batch_ids.push_back(i);
        // }
        // std::random_shuffle(search_batch_ids.begin(), search_batch_ids.end());
        // std::unordered_set<int> us;
        // for (int i = 0; i < ceil<size_t>(search_keys_sz, batch_size); ++i) {
        //     us.insert(search_batch_ids[i]);
        // }

        // size_t ids = 0;
        // while (rmi > 0) {
        //     size_t cpd = std::min(rmi, batch_size);

        //     keys = tool.random(keys, cpd);
        //     map.insert(keys, cpd);
        //     if (us.find(ids) != us.end()) {
        //         emap.insert(keys, cpd);
        //     }
        //     ++ids;
        //     rmi -= cpd;
        // }

        // Unordered_Map<Key_T, Count_T, Hashed_T> *um = dynamic_cast<Unordered_Map<Key_T, Count_T, Hashed_T> *>(&emap);

        // auto bit = um->um.begin();
        // Key_T *h_keys = nullptr;
        // Count_T *h_e_counts = nullptr;
        // h_keys = new Key_T[batch_size];
        // h_e_counts = new Count_T[batch_size];
        // while (rms > 0) {
        //     size_t cpd = std::min(rms, batch_size);
            
        //     for (size_t i = 0; i < cpd; ++i) {
        //         h_keys[i] = bit->first;
        //         h_e_counts[i] = bit->second;
        //         ++bit;
        //     }


        // }


        // counts = tool.zero(counts, search_keys_sz);
        // map.search(keys, search_keys_sz, counts);
        // e_counts = tool.zero(e_counts, search_keys_sz);
        // emap.search(keys, search_keys_sz, e_counts);

        // auto dc3 = diff(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
        //     thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
        // auto max = max_diff(dc3);
        // auto ave = average_diff(dc3);

        // auto dc3_p = percentage(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
        //     thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
        // auto max_p = max_diff(dc3_p);
        // auto ave_p = average_diff(dc3_p);

        // std::cout << "max: " << max << "\tave: " << ave << std::endl;
        // std::cout << "maxp: " << max_p << "%\tavep: " << ave_p << "%" << std::endl;

        // tool.free(keys);
        // tool.free(counts);
        // tool.free(e_counts);
        return 0;
    }

    int search_accuracy_freq_test(size_t insert_keys_sz, size_t search_keys_sz,
     size_t search_loop = 1, std::ostream &os = std::cout) {

        assert(insert_keys_sz >= search_keys_sz);

        Key_T *keys = nullptr;// = new Key_T[keys_sz];
        Count_T *counts = nullptr;
        Count_T *e_counts = nullptr;

        // double tts = 0;
        // long tdf = 0; // total difference between the accuracy counts and map counts

        keys = tool.random_freq(keys, insert_keys_sz);
                    // std::cout << "p0" << std::endl;
        map.pre_cal(keys, std::min<size_t>(HASH_MASK_TABLE_SIZE, insert_keys_sz), nullptr);
        map.insert(keys, insert_keys_sz);
                    // std::cout << "p1" << std::endl;

        emap.insert(keys, insert_keys_sz);
                    // std::cout << "p2" << std::endl;

        // std::cout << "-------------------------------------" << std::endl;
        for (size_t i = 0; i < search_loop; ++i) {

            // std::random_shuffle(keys, keys + search_keys_sz);
            // tool.random_shuffle(keys, search_keys_sz);
            counts = tool.zero(counts, search_keys_sz);
            map.search(keys, search_keys_sz, counts);
            // std::cout << "lp0" << std::endl;
            e_counts = tool.zero(e_counts, search_keys_sz);
            emap.search(keys, search_keys_sz, e_counts);
            // std::cout << "lp1" << std::endl;

            // thrust::device_vector<Count_T> dv1(counts, counts + search_keys_sz);
            // thrust::host_vector<Count_T> hv1(dv1.begin(), dv1.end());
            // thrust::device_vector<Count_T> dv2(e_counts, e_counts + search_keys_sz);
            // thrust::host_vector<Count_T> hv2(dv2.begin(), dv2.end());
            // std::cout << "p2" << std::endl;

            // for (auto v : hv1) {
            //     std::cout << v << "; ";
            // }
            // std::cout << std::endl;

            // for (auto v : hv2) {
            //     std::cout << v << "; ";
            // }
            // std::cout << std::endl;

            auto dc3 = diff(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
                thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
            auto max = max_diff(dc3);
            auto ave = average_diff(dc3);
            auto greater_counts = greater(dc3);

            auto dc3_p = percentage(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
                thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
            auto max_p = max_diff(dc3_p);
            auto ave_p = average_diff(dc3_p);

            // Count_T *c3 = diff(counts, e_counts, search_keys_sz);
            // Count_T max = max_diff(c3, search_keys_sz);
            // double ave = average_diff(c3, search_keys_sz);
            // // percentage
            // double *cp = percentage(counts, e_counts, search_keys_sz);
            // double maxp = max_diff(cp, search_keys_sz);
            // double avep = average_diff(cp, search_keys_sz);

            // if (DEBUG) {
            //     for (size_t j = 0; j < search_keys_sz; ++j) {
            //         debug_log << keys[j] << "\t:" << counts[j] << ", " << e_counts[j] << std::endl;
            //     }
            // }

            // std::cout << "search " << i << " accuracy: " << std:: endl;

            // if (max >= 102400) 
            {
                std::cout << "max: " << max << "\tave: " << ave << std::endl;
                std::cout << "maxp: " << max_p << "%\tavep: " << ave_p << "%" << std::endl;
                std::cout << "greater_counts: " << greater_counts << std::endl;
                

                // return -1;
            }
            // error_max = std::max<size_t>(error_max, max);



        }


        
        tool.free(keys);
        tool.free(counts);
        tool.free(e_counts);
        return 0;
    }


    int accuracy_test(size_t insert_keys_sz, size_t search_keys_sz,
        size_t insert_loop, size_t search_loop, std::ostream &os = std::cout) {

        Key_T *keys = nullptr;// = new Key_T[keys_sz];
        Count_T *counts = nullptr;
        Count_T *e_counts = nullptr;

        for (size_t i = 0; i < insert_loop; ++i) {
            keys = tool.random(keys, insert_keys_sz);
            map.insert(keys, insert_keys_sz);
            emap.insert(keys, insert_keys_sz);
        }



        
        // std::cout << "-------------------------------------" << std::endl;
        for (size_t i = 0; i < search_loop; ++i) {

            // std::random_shuffle(keys, keys + search_keys_sz);
            // tool.random_shuffle(keys, search_keys_sz);
            counts = tool.zero(counts, search_keys_sz);
            map.search(keys, search_keys_sz, counts);

            e_counts = tool.zero(e_counts, search_keys_sz);
            emap.search(keys, search_keys_sz, e_counts);

            // for (size_t i = 0; i < search_keys_sz; ++i) {
            //     if (e_counts[i] == 0) {
            //         std::cout << "err" << std::endl;
            //         return 0;
            //     }
            // }
            // std::cout << "p1" << std::endl;

            // thrust::device_vector<Count_T> dv1(counts, counts + search_keys_sz);
            // thrust::host_vector<Count_T> hv1(dv1.begin(), dv1.end());
            // thrust::device_vector<Count_T> dv2(e_counts, e_counts + search_keys_sz);
            // thrust::host_vector<Count_T> hv2(dv2.begin(), dv2.end());
            // std::cout << "p2" << std::endl;

            // for (auto v : hv1) {
            //     std::cout << v << "; ";
            // }
            // std::cout << std::endl;

            // for (auto v : hv2) {
            //     std::cout << v << "; ";
            // }
            // std::cout << std::endl;

            auto dc3 = diff(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
                thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
            auto max = max_diff(dc3);
            auto ave = average_diff(dc3);

            auto dc3_p = percentage(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
                thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
            auto max_p = max_diff(dc3_p);
            auto ave_p = average_diff(dc3_p);

            // Count_T *c3 = diff(counts, e_counts, search_keys_sz);
            // Count_T max = max_diff(c3, search_keys_sz);
            // double ave = average_diff(c3, search_keys_sz);
            // // percentage
            // double *cp = percentage(counts, e_counts, search_keys_sz);
            // double maxp = max_diff(cp, search_keys_sz);
            // double avep = average_diff(cp, search_keys_sz);

            // if (DEBUG) {
            //     for (size_t j = 0; j < search_keys_sz; ++j) {
            //         debug_log << keys[j] << "\t:" << counts[j] << ", " << e_counts[j] << std::endl;
            //     }
            // }

            // std::cout << "search " << i << " accuracy: " << std:: endl;
            std::cout << "max: " << max << "\tave: " << ave << std::endl;
            std::cout << "maxp: " << max_p << "%\tavep: " << ave_p << "%" << std::endl;



        }


        
        tool.free(keys);
        tool.free(counts);
        tool.free(e_counts);
        return 0;
    }

    int accuracy_test_freq(size_t insert_keys_sz, size_t search_keys_sz,
        size_t insert_loop, size_t search_loop, std::ostream &os = std::cout) {

        Key_T *keys = nullptr;// = new Key_T[keys_sz];
        Count_T *counts = nullptr;
        Count_T *e_counts = nullptr;


        for (size_t i = 0; i < insert_loop; ++i) {
            keys = tool.random_freq(keys, insert_keys_sz,
                std::numeric_limits<Key_T>::min(),std::numeric_limits<Key_T>::max(),
                0.99, 1, 1, 4, 128);
            map.insert(keys, insert_keys_sz);
            emap.insert(keys, insert_keys_sz);
        }



        
        // std::cout << "-------------------------------------" << std::endl;
        for (size_t i = 0; i < search_loop; ++i) {

            // std::random_shuffle(keys, keys + search_keys_sz);
            // tool.random_shuffle(keys, search_keys_sz);
            counts = tool.zero(counts, search_keys_sz);
            map.search(keys, search_keys_sz, counts);

            e_counts = tool.zero(e_counts, search_keys_sz);
            emap.search(keys, search_keys_sz, e_counts);

            // for (size_t i = 0; i < search_keys_sz; ++i) {
            //     if (e_counts[i] == 0) {
            //         std::cout << "err" << std::endl;
            //         return 0;
            //     }
            // }
            // std::cout << "p1" << std::endl;

            // thrust::device_vector<Count_T> dv1(counts, counts + search_keys_sz);
            // thrust::host_vector<Count_T> hv1(dv1.begin(), dv1.end());
            // thrust::device_vector<Count_T> dv2(e_counts, e_counts + search_keys_sz);
            // thrust::host_vector<Count_T> hv2(dv2.begin(), dv2.end());
            // std::cout << "p2" << std::endl;

            // for (auto v : hv1) {
            //     std::cout << v << "; ";
            // }
            // std::cout << std::endl;

            // for (auto v : hv2) {
            //     std::cout << v << "; ";
            // }
            // std::cout << std::endl;

            auto dc3 = diff(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
                thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
            auto max = max_diff(dc3);
            auto ave = average_diff(dc3);

            auto dc3_p = percentage(thrust::device_vector<Count_T>(counts, counts + search_keys_sz), 
                thrust::device_vector<Count_T>(e_counts, e_counts + search_keys_sz));
            auto max_p = max_diff(dc3_p);
            auto ave_p = average_diff(dc3_p);

            // Count_T *c3 = diff(counts, e_counts, search_keys_sz);
            // Count_T max = max_diff(c3, search_keys_sz);
            // double ave = average_diff(c3, search_keys_sz);
            // // percentage
            // double *cp = percentage(counts, e_counts, search_keys_sz);
            // double maxp = max_diff(cp, search_keys_sz);
            // double avep = average_diff(cp, search_keys_sz);

            // if (DEBUG) {
            //     for (size_t j = 0; j < search_keys_sz; ++j) {
            //         debug_log << keys[j] << "\t:" << counts[j] << ", " << e_counts[j] << std::endl;
            //     }
            // }

            // std::cout << "search " << i << " accuracy: " << std:: endl;
            std::cout << "max: " << max << "\tave: " << ave << std::endl;
            std::cout << "maxp: " << max_p << "%\tavep: " << ave_p << "%" << std::endl;



        }


        
        tool.free(keys);
        tool.free(counts);
        tool.free(e_counts);
        return 0;
    }
};
