#include "lib/lib.h"

std::unordered_set<void *> pointers;

int main(int argc, char const *argv[])
{
    size_t insert_keys_size = 32 * 1024;
    size_t search_keys_size = 1024;

    using Key_T = unsigned int; 
    using Count_T = unsigned int;

    CPU_Tools<Key_T> tool;
    // Unordered_Map<Key_T, Count_T> map;
    Count_Min_Sketch_CPU<Key_T, Count_T> map;
    Unordered_Map<Key_T, Count_T> emap;
    // Test<unsigned int, unsigned int> test(tool, static_cast<Sketch<Key_T, Count_T> &>(map), static_cast<Sketch<Key_T, Count_T> &>(emap));
    Test<unsigned int, unsigned int> test(tool, (map), (emap));

    // test.insert_perf_test(insert_keys_size);
    // test.clear();
    // test.search_perf_test(insert_keys_size, search_keys_size);
    // test.clear();
    test.search_accuracy_test(insert_keys_size, search_keys_size);
    // map.print(debug_log);
    return 0;
}