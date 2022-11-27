#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

int main(int argc, char const *argv[])
{
    ifstream ifs("slab_hash.txt");
    int lineNum = 1;
    string line;
    while (ifs) {
        getline(ifs, line);
        if (line[0] == '(' && line[1] == '1') {
            cout << line << endl;
        }
    }
    return 0;
}