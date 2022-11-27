import numpy as np
import matplotlib.pyplot as plt


label_height = 0.01;
default_fontsize = 12;

def set_label(rs):
	for r in rs:
		h = r.get_height()
		plt.text(x = r.get_x() + r.get_width() / 2,
				y = h + label_height,
				s = h,
				ha = 'center',
				fontsize=default_fontsize)

size = 4
x = np.arange(size)
# a = np.array([395.63, 1175.23, 1145.53, 1142.08]); # 2080ti wlf = 0.5
# b = np.array([399.663, 1196.62, 1172.78, 1162.04]); #2080ti wlf = 1.0
# c = np.array([421.288, 691.001, 1206.58, 1040.63]); # p100 wlf = 0.5
# d = np.array([427.328, 692.506, 1242.24, 1153.16]); # p100 wlf = 1.0



a = np.array([395.63,399.663,421.288,427.328]); # thread
b = np.array([1175.23,1196.62,691.001,692.506]); # warp
c = np.array([1145.53,1172.78,1206.58,1242.24]); # sub warp
d = np.array([1142.08,1162.04,1040.63,1153.16]); # msws

total_width, n = 0.8, 4
width = total_width / n
# t = ['d = 1, w = 2 ^ 27 / 1', 'd = 2, w = 2 ^ 27 / 2', 'd = 3, w = 2 ^ 27 / 3']
t = ['2080 TI, WLF = 0.5', '2080 TI, WLF = 1.0', 'P100, WLF = 0.5', 'P100, WLF = 1.0']

ra = plt.bar(x, a,  width=width, label='Thread Insert')
rb = plt.bar(x + width, b, width=width, label='Warp Insert')
rc = plt.bar(x + width * 2, c, width=width, label='Sub Warp Insert')
rd = plt.bar(x + width * 3, d, width=width, label='MSWS Insert')

# plt.xticks(x + total_width / 2, t, fontsize=default_fontsize, rotation=35)
plt.xticks(x, t, fontsize=default_fontsize, rotation=35)
plt.yticks(fontsize=default_fontsize)

# plt.xlabel('Thread Sketches with different d')
plt.ylabel('Throughput (Million operations/s)', fontsize=default_fontsize)
# plt.ylim(0, 1.2)

# set_label(ra)
# set_label(rb)

plt.subplots_adjust(left = 0.18, right = 0.7, bottom=0.25)
# plt.legend(fontsize=default_fontsize)
# plt.legend(bbox_to_anchor=(0.5, -0.5), loc='lower center')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()
