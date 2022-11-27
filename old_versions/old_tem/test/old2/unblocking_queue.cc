#include <atomic>
#include <iostream>
#include <random>
#include <thread>

using namespace std;

// template<typename T>
struct Queue {

    int *buffer;
    std::atomic<int> sz;
    std::atomic<int> max_size;
    std::atomic<int> b;
    std::atomic<int> e;

    Queue(int _sz) {
        sz.store(0);
        max_size.store(_sz);
        b.store(0);
        e.store(0);
        // : max_size(_sz)
        buffer = new int[max_size];
    }
    void push(const int &val) {
        while (true) {
            auto old_v = sz.fetch_add(1);
            if (old_v < max_size) {
                buffer[e % max_size] = val;
                e++;
                return;
            } else {
                sz.fetch_sub(1);
            }
        }
    }

    bool try_push(const int &val) {
        auto old_v = sz.fetch_add(1);
        if (old_v < max_size) {
            buffer[e % max_size] = val;
            e++;
            return true;
        } else {
            sz.fetch_sub(1);
        }
        return false;
        
    }

    int pop() {
        while (true) {
            auto old_v = sz.fetch_sub(1);
            if (old_v > 0) {
                auto rt = buffer[b % max_size];
                b++;
                return rt;
            } else {
                sz.fetch_add(1);
            }
        }
    }

    bool try_pop(int *rt) {
        auto old_v = sz.fetch_sub(1);
        if (old_v > 0) {
            *rt = buffer[b % max_size];
            b++;
            return true;
        } else {
            sz.fetch_add(1);
        }
        return false;
        
    }
};

size_t c[8] = {0,0,0,0,0,0,0,0};

// void write(Queue *q, int id) {
//     std::default_random_engine eng;
//     std::uniform_int_distribution<int> dis(0, 100000);
//     while (true) {
//         int w = dis(eng);
//         q->push(w);
//         c[id]++;
//         // std::cout << "write: " << w << std::endl;
//     }
// }

// void read(Queue *q, int id) {
//     while (true) {
//         auto r = q->pop();
//         c[id]++;
//         // std::cout << "read: " << r << std::endl;
//     }
// }


void write(Queue *q, int id) {
    std::default_random_engine eng;
    std::uniform_int_distribution<int> dis(0, 100000);
    while (true) {
        int w = dis(eng);
        if (q->try_push(w)) {
            c[id]++;
        }
        
        // std::cout << "write: " << w << std::endl;
    }
}

void read(Queue *q, int id) {
    int r;
    while (true) {
        if (q->try_pop(&r)) {
            c[id]++;
        }
        // auto r = q->pop();
        
        // std::cout << "read: " << r << std::endl;
    }
}

int main(int argc, char const *argv[])
{
    Queue q(1024);
    // Queue &rq = q;

    // vector<thread *>
    std::thread r1(read, &q, 0);
    std::thread r2(read, &q, 1);
    std::thread r3(read, &q, 2);
    std::thread r4(read, &q, 3);
    std::thread w1(write, &q, 4);
    std::thread w2(write, &q, 5);
    // std::thread w3(write, &q, 6);
    // std::thread w4(write, &q, 7);

    using namespace std::chrono;
    duration<double> time_span;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    while (true) {
        size_t sum = 0;
        for (size_t i = 0; i < 8; ++i) {
            sum += c[i];
        }
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        time_span = duration_cast<decltype(time_span)>(t2 - t1);
        std::cout << "\r" << sum / time_span.count() << "                            ";
    }

    // r.join();
    // w.join();

    return 0;
}


    
    // run();

    
    // return time_span.count();;