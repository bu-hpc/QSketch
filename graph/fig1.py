import numpy as np
import matplotlib.pyplot as plt


label_height = 25;
default_fontsize = 18;

def set_label(rs):
	for r in rs:
		h = r.get_height()
		plt.text(x = r.get_x() + r.get_width() / 2,
				y = h + label_height,
				s = h,
				ha = 'center',
				fontsize=default_fontsize)


size = 3
x = np.arange(size)
a = np.array([1189.56, 594.891, 396.551])
b = np.array([2789.49, 1464.48, 992.341])
a = np.around(a, decimals=1)
b = np.around(b, decimals=1)


total_width, n = 0.8, 2
width = total_width / n
# t = ['d = 1, w = 2 ^ 27 / 1', 'd = 2, w = 2 ^ 27 / 2', 'd = 3, w = 2 ^ 27 / 3']
t = ['d = 1', 'd = 2', 'd = 3']

ra = plt.bar(x, a,  width=width, label='INSERT')
rb = plt.bar(x + width, b, width=width, label='QUERY')
plt.xticks(x + total_width / 2 - width / 2, t, fontsize=default_fontsize)
plt.yticks(fontsize=default_fontsize)

# plt.xlabel('Thread Sketches with different d')
plt.ylabel('Throughput (Million operations/s)', fontsize=default_fontsize)
plt.ylim(0, 3000)

set_label(ra)
set_label(rb)

plt.subplots_adjust(left = 0.17, right = 0.99)
plt.legend(fontsize=default_fontsize)
plt.show()
