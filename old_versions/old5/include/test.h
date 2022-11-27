#pragma once

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

    int insert_perf_test(size_t keys_sz, size_t insert_loop = 1) {

        Key_T *keys = nullptr;
        double tts = 0; // total_time_span
            keys = tool.random(keys, keys_sz);

        for (size_t i = 0; i < insert_loop; ++i) {
            // std::cout << "insert " << i << std::endl;
            double ts = get_run_time([&]() {
                map.insert(keys, keys_sz);
            });
            tts += ts;
            // std::cout << "insert done" << i << std::endl;
        }

        size_t insert_number = keys_sz * insert_loop;
        // std::cout << "insert perf: " << format_perf(insert_number / tts) << std::endl;
        std::cout << format_perf(insert_number / tts) << std::endl;


        cudaFree(keys);
        return 0;
    }

    int search_perf_test(size_t insert_keys_sz, size_t search_keys_sz,
     size_t search_loop = 1) {

        assert(insert_keys_sz >= search_keys_sz);

        Key_T *keys = nullptr;// = new Key_T[keys_sz];
        Count_T *counts = nullptr;

        double tts = 0;
        long tdf = 0; // total difference between the accuracy counts and map counts

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

        size_t insert_number = insert_keys_sz * search_loop;
        std::cout << "search perf: " << format_perf(insert_number / tts) << std::endl;

        cudaFree(keys);
        cudaFree(counts);
        return 0;
    }

    int search_accuracy_test(size_t insert_keys_sz, size_t search_keys_sz,
     size_t search_loop = 1) {

        assert(insert_keys_sz >= search_keys_sz);

        Key_T *keys = nullptr;// = new Key_T[keys_sz];
        Count_T *counts = nullptr;
        Count_T *e_counts = nullptr;

        // double tts = 0;
        // long tdf = 0; // total difference between the accuracy counts and map counts

        keys = tool.random(keys, insert_keys_sz);
        map.insert(keys, insert_keys_sz);
        emap.insert(keys, insert_keys_sz);
        // std::cout << "-------------------------------------" << std::endl;
        for (size_t i = 0; i < search_loop; ++i) {

            // std::random_shuffle(keys, keys + search_keys_sz);
            // tool.random_shuffle(keys, search_keys_sz);
            counts = tool.zero(counts, search_keys_sz);
            map.search(keys, search_keys_sz, counts);

            e_counts = tool.zero(e_counts, search_keys_sz);
            emap.search(keys, search_keys_sz, e_counts);
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


        
        cudaFree(keys);
        cudaFree(counts);
        cudaFree(e_counts);
        return 0;
    }
};
