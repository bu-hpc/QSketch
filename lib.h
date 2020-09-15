#ifndef LIB_HEADER
#define LIB_HEADER 

#include <iostream>
#include <string>
#include <functional>
#include <fstream>
#include <exception>

using ulong = unsigned long;
using uing = unsigned int;
using uchar = unsigned char;
using default_hash_function_seed_type = ulong;

enum File_Type
{
    binary
};

static struct {
    struct {
        const std::string urandom = "/dev/urandom";
    } path;
} global;

struct File_Header {
    size_t sz;
    File_Type file_type;
};


template<typename T>
T * generate_random_data(size_t sz, const std::function<void(void *)> &random_allocator = {});

int read_file(const std::string &file_name, void **buf, size_t &sz);
int write_file(const std::string &file_name, void *buf, size_t sz);



template<typename T>
T * generate_random_data(size_t sz, const std::function<void(void *)> &random_allocator) {
    try {
        T *buf = new T[sz];

        if (random_allocator) {
            for (size_t i = 0; i < sz; ++i) {
                random_allocator(buf + i);
            }
        } else {
            std::ifstream ifs(global.path.urandom);
            ifs.get((char *)(buf), sizeof(T) * sz);
            ifs.close();
        }

        return buf;

    } catch (std::exception &e) {
        throw;
    }
    
    return nullptr;
}
#endif
