#include <iostream>
#include <vector>
#include <array>

using namespace std;

int count32(int a, int b, int c) {
	std::array<int, 4> count = {0, 0, 0, 0};
	count[a/8]++;
	count[b/8]++;
	count[c/8]++;
	int ans = 0;
	for (auto &val : count) {
		if (val !=0) {
			ans++;
		}
	}
	return ans;
}

int cal32() {
	int s = 0;
	int t = 0;
	for (int i = 0; i < 32; ++i) {
		for (int j = i + 1; j < 32; ++j) {
			for (int k = j + 1; k < 32; ++k) {
				s++;
				int tem = count32(i, j, k);
				cout << tem << endl;
				t += tem;
			}
		}
	}
	cout << t << "/" << s << endl;
	return 0;
}


int count16(int a, int b, int c) {
	std::array<int, 2> count = {0, 0};
	count[a/8]++;
	count[b/8]++;
	count[c/8]++;
	int ans = 0;
	for (auto &val : count) {
		if (val !=0) {
			ans++;
		}
	}
	return ans;
}


int cal16() {
	int s = 0;
	int t = 0;
	for (int i = 0; i < 16; ++i) {
		for (int j = i + 1; j < 16; ++j) {
			for (int k = j + 1; k < 16; ++k) {
				s++;
				int tem = count16(i, j, k);
				t += tem;
			}
		}
	}
	cout << t << "/" << s << endl;
	return 0;
}

int main(int argc, char const *argv[])
{
	// cout << t << "/" << s << endl;
	cal32();
	cal16();
	return 0;
}
