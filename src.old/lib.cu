// #include <lib.h>

// int read_file(const std::string &file_name, void **buf, size_t &sz) {
//     std::ifstream ifs(file_name, std::ios::binary);
//     if (!ifs) {
//         return -1;
//     }
    
//     ifs.seekg (0, ifs.end);
//     sz = ifs.tellg();
//     ifs.seekg (0, ifs.beg);

//     *buf = new unsigned char[sz];

//     ifs.read ((char *)(*buf),sz);

//     ifs.close();

//     return 0;
// }
// int write_file(const std::string &file_name, void *buf, size_t sz) {
//     std::ofstream ofs(file_name, std::ios::binary);
//     if (!ofs) {
//         return -1;
//     }
//     ofs.write((char *)(buf), sz);
//     ofs.close();
//     return 0;
// }