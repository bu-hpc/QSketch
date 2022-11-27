#pragma once
// #define default_device_buf_sz 100
namespace qsketch {
template <typename Key_T, typename Count_T, typename Hashed_T = size_t>
struct Test {
    Tools<Key_T> &tool_key;
    Tools<Count_T> &tool_count;
    Sketch<Key_T, Count_T> &map;
    Sketch<Key_T, Count_T> &emap;
    size_t device_buf_sz;
    const static size_t default_device_buf_sz = 1 << 24;
    Test(Tools<Key_T> &tk, Tools<Count_T> &tc, Sketch<Key_T, Count_T> &m, Sketch<Key_T, Count_T> &em, 
        size_t __device_buf_sz = default_device_buf_sz) : tool_key(tk), tool_count(tc), map(m), emap(em),
        device_buf_sz(__device_buf_sz) {
    }

    void test() {

    }

    int insert_perf_test_old(size_t keys_sz, size_t insert_loop = 1, std::ostream &os = std::cout) {

        Key_T *keys = nullptr;
        auto &tool = tool_key;
        keys = tool.random(keys, keys_sz);
        // std::cout << "insert_perf_test p0" << std::endl;
        // map.pre_cal(keys, std::min<size_t>(HASH_MASK_TABLE_SIZE, keys_sz), nullptr);
        // std::cout << "insert_perf_test p1" << std::endl;
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
        os << "insert perf old: " << format_perf(insert_number / tts) << std::endl;
        os << "time usage:  " << tts << "s" << std::endl;

        // cudaFree(keys);
        tool.free(keys);
        return 0;
    }
    int insert_perf_freq_test_old(size_t keys_sz, size_t insert_loop = 1, std::ostream &os = std::cout) {
        auto &tool = tool_key;
        
        Key_T *keys = nullptr;
        keys = tool.random_freq(keys, keys_sz);
        double tts = 0;
        for (size_t i = 0; i < insert_loop; ++i) {
            double ts = get_run_time([&]() {
                map.insert(keys, keys_sz);
            });
            tts += ts;
        }

        size_t insert_number = keys_sz * insert_loop;
        os << "insert perf: " << format_perf(insert_number / tts) << std::endl;


        tool.free(keys);
        return 0;
    }


    int search_perf_test_old(size_t insert_keys_sz, size_t search_keys_sz, size_t search_loop = 1, std::ostream &os = std::cout) {
        auto &tool = tool_key;
        assert(insert_keys_sz >= search_keys_sz);

        Key_T *keys = nullptr;// = new Key_T[keys_sz];
        Count_T *counts = nullptr;

        double tts = 0;
        // long tdf = 0; // total difference between the accuracy counts and map counts

        keys = tool.random(keys, insert_keys_sz);
        map.insert(keys, insert_keys_sz);

        for (size_t i = 0; i < search_loop; ++i) {

            tool.random_shuffle(keys, search_keys_sz);
            counts = tool.zero(counts, search_keys_sz);

            double ts = get_run_time([&](){
                map.search(keys, search_keys_sz, counts);
            });
            tts += ts;

        }

        size_t search_number = search_keys_sz * search_loop;
        os << "search perf old: " << format_perf(search_number / tts) << std::endl;
        os << "time usage:  " << tts << "s" << std::endl;
        tool.free(keys);
        tool.free(counts);
        return 0;
    }

    int search_perf_freq_test_old(size_t insert_keys_sz, size_t search_keys_sz,
     size_t search_loop = 1, std::ostream &os = std::cout) {
        auto &tool = tool_key;

        assert(insert_keys_sz >= search_keys_sz);

        Key_T *keys = nullptr;// = new Key_T[keys_sz];
        Count_T *counts = nullptr;

        double tts = 0;
        // long tdf = 0; // total difference between the accuracy counts and map counts

        keys = tool.random_freq(keys, insert_keys_sz);
        map.insert(keys, insert_keys_sz);

        for (size_t i = 0; i < search_loop; ++i) {

            tool.random_shuffle(keys, search_keys_sz);
            counts = tool.zero(counts, search_keys_sz);

            double ts = get_run_time([&](){
                map.search(keys, search_keys_sz, counts);
            });
            tts += ts;

        }

        size_t search_number = search_keys_sz * search_loop;
        std::cout << "search perf: " << format_perf(search_number / tts) << std::endl;

        tool.free(keys);
        tool.free(counts);
        return 0;
    }

    double insert_perf_test(size_t keys_sz, std::ostream &os = std::cout) {
        
        Key_T *keys = nullptr;
        double tts = 0;
        size_t work_load = keys_sz;
        while (keys_sz > 0) {
            size_t batch_sz = std::min(keys_sz, device_buf_sz / sizeof(Key_T));
            keys = tool_key.random(keys, batch_sz);
            double ts = get_run_time([&]() {
                map.insert(keys, batch_sz);
            });
            tts += ts;
            keys_sz -= batch_sz;
        }
        // os << "insert perf: " << format_perf(work_load / tts) << std::endl;
        // tool_key.free(keys);
        // return 0;

        tool_key.free(keys);
        return format_perf_d(work_load / tts);
    }

    double search_perf_test(size_t keys_sz, std::ostream &os = std::cout) {
        Key_T *keys = nullptr;
        Count_T *counts = nullptr;
        double tts = 0;
        size_t work_load = keys_sz;
        while (keys_sz > 0) {
            size_t batch_sz = std::min(keys_sz, device_buf_sz / sizeof(Key_T));
            keys = tool_key.random(keys, batch_sz);
            counts = tool_count.zero(counts, batch_sz);
            double ts = get_run_time([&]() {
                map.search(keys, batch_sz, counts);
            });
            tts += ts;
            keys_sz -= batch_sz;
        }
        // os << "search perf: " << format_perf(work_load / tts) << std::endl;
        
        // tool_key.free(keys);
        // tool_count.free(counts);
        // return 0;

        tool_key.free(keys);
        tool_count.free(counts);
        return format_perf_d(work_load / tts);
    }

    double search_accuracy_test(size_t insert_keys_sz, size_t search_keys_sz,
        std::ostream &os = std::cout) {
        using emap_type = Unordered_Map<Key_T, Count_T>;
        emap_type *emr = static_cast<emap_type *>(&emap);
        emr->limit = search_keys_sz;

        Key_T *keys = nullptr;

        // size_t emap_insert_rm = search_keys_sz;
        while (insert_keys_sz > 0) {
            size_t batch_sz = std::min(insert_keys_sz, device_buf_sz / sizeof(Key_T));
            keys = tool_key.random(keys, batch_sz);
            // keys = tool_key.zero(keys, batch_sz);
            map.insert(keys, batch_sz);
            emap.insert(keys, batch_sz);
            insert_keys_sz -= batch_sz;
        }

        
        Count_T *e_counts = tool_count.zero(nullptr, device_buf_sz / sizeof(Key_T));
        Count_T *counts = tool_count.zero(nullptr, device_buf_sz / sizeof(Key_T));

#ifdef QSKETCH_ANALYZE

        Count_T max_all = 0;
        Percentage_T ave_all = 0.0;
        size_t greater_counts_all = 0;

        Percentage_T max_p_all = 0.0;
        Percentage_T ave_p_all = 0.0;

        size_t search_keys_sz_analyze = search_keys_sz;

#endif


        while (search_keys_sz > 0) {
            size_t batch_sz = std::min(search_keys_sz, device_buf_sz / sizeof(Key_T));
            size_t emr_rt = emr->get_counts(keys, batch_sz, e_counts);

            // if (emr_rt != batch_sz) 
            // {
            //     std::cout << "err: " << emr_rt << "," << batch_sz << std::endl;
            // }
            // map.search(keys, 4, counts);
            map.search(keys, batch_sz, counts);

#ifdef QSKETCH_ANALYZE

            thrust::device_vector<Count_T> dvc(counts, counts + batch_sz);
            thrust::device_vector<Count_T> dvec(e_counts, e_counts + batch_sz);


            auto dc = diff(dvc, dvec);
            // // std::cout << dc.size() << std::endl;
            // Count_T max = thrust::reduce(dc.begin(), dc.end(), Count_T(0), thrust::maximum<Count_T>());
            Count_T max = max_diff(dc);
            size_t greater_counts = greater(dc);
            max_all = std::max<Count_T>(max_all, max);
            ave_all += sum(dc);
            greater_counts_all += greater_counts;

            auto dc_p = percentage(dvc, dvec);
            Percentage_T max_p = max_diff(dc_p);
            max_p_all = std::max<Percentage_T>(max_p_all, max_p);
            ave_p_all += sum(dc_p);
    #ifdef QSKETCH_DEBUG
            if (false) {
                thrust::host_vector<Count_T> hvc(dvc);
                thrust::host_vector<Count_T> hvec(dvec);
                auto jt = hvec.begin();
                for (auto it = hvc.begin(); it != hvc.end(); ++it) {
                    std::cout << (*it) << "<-->" << (*jt) << std::endl;
                    ++jt;
                }
                
            }
    #endif

#endif
            search_keys_sz -= batch_sz;
        }


#ifdef QSKETCH_ANALYZE

        ave_all /= search_keys_sz_analyze;
        ave_p_all /= search_keys_sz_analyze;

        // std::cout << "max: " << max_all << "\tave: " << ave_all << std::endl;
        // std::cout << "maxp: " << max_p_all << "%\tavep: " << ave_p_all << "%" << std::endl;
        // std::cout << "greater_counts: " << greater_counts_all << std::endl;

#endif

        // counts = tool.zero(counts, batch_sz);


        // os << "search perf: " << format_perf(work_load / tts) << std::endl;
        
        tool_key.free(keys);
        tool_count.free(counts);
        tool_count.free(e_counts);
        return ave_p_all;
        // return 0;
    }

    double search_accuracy_freq_test(size_t insert_keys_sz, size_t search_keys_sz,
     size_t search_loop = 1, std::ostream &os = std::cout) {
        auto &tool = tool_key;
        assert(insert_keys_sz >= search_keys_sz);

        Key_T *keys = nullptr;// = new Key_T[keys_sz];
        Count_T *counts = nullptr;
        Count_T *e_counts = nullptr;

        keys = tool.random_freq(keys, insert_keys_sz);
        // map.pre_cal(keys, std::min<size_t>(HASH_MASK_TABLE_SIZE, insert_keys_sz), nullptr);
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


        {
            std::cout << "max: " << max << "\tave: " << ave << std::endl;
            std::cout << "maxp: " << max_p << "%\tavep: " << ave_p << "%" << std::endl;
            std::cout << "greater_counts: " << greater_counts << std::endl;
        }

        // tool_key.free(keys);
        // tool_count.free(counts);
        // tool_count.free(e_counts);

        // return ave_p;

        if (false){
            Count_T *h_counts = new Count_T[search_keys_sz];
            Count_T *h_e_counts = new Count_T[search_keys_sz];
            Count_T *h_keys = new Key_T[insert_keys_sz];

            CUDA_CALL(cudaMemcpy(h_counts, counts, search_keys_sz * sizeof(Count_T), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(h_e_counts, e_counts, search_keys_sz * sizeof(Count_T), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(h_keys, keys, insert_keys_sz * sizeof(Key_T), cudaMemcpyDeviceToHost));

            using p = std::tuple<Count_T, Count_T, Count_T>;
            std::vector<p> v;

            for (size_t j = 0; j < search_keys_sz; ++j) {
                if (h_counts[j] < h_e_counts[j]) {
                    // std::cout << h_keys[j] << ":" << h_counts[j] << "," << h_e_counts[j] << std::endl;
                    // v.push_back(p(h_keys[j], h_counts[j], h_e_counts[j]));

                    // if (h_e_counts[j] == 1) {
                    //     std::cout << h_keys[j] << ":" << h_counts[j] << "," << h_e_counts[j] << std::endl;
                    // }
                }
                // if (h_keys[j] == 4087563711) {
                //     static int c = 0;
                //     std::cout << "find: " << (c++) << std::endl;
                // }
                // if (h_e_counts[j] >= 2) {
                //     std::cout << h_keys[j] << "," << h_e_counts[j] << std::endl;
                // }
            }

            std::sort(v.begin(), v.end());

            for (auto &val : v) {
                std::cout << std::get<0>(val) << ":" << std::get<1>(val) << "," << std::get<2>(val) << std::endl;
            }

        }
        
        tool.free(keys);
        tool.free(counts);
        tool.free(e_counts);
        return ave_p;
    }


    void clear() {
        map.clear();
        emap.clear();
    }
};
}