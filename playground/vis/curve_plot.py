import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 5))  # 宽8英寸，高6英寸，比例为4:3

x = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
y = [42.4, 44.5, 46.6, 44.2, 43.7, 39.2]

plt.plot(x, y, color='blue', marker='o')
plt.grid(True)

fz =16
plt.ylim(37.5, 50)
plt.xlabel('Grid Resolution in VCCS', fontsize=fz)
plt.ylabel('mIoU scores (%)', fontsize=fz)
# plt.title('Curve Plot')

plt.tick_params(axis='x', labelsize=fz-2)
plt.tick_params(axis='y', labelsize=fz-2)

plt.yticks(np.arange(37.5, 51, 2.5))


index = 2
plt.text(x[index], y[index], f'{x[index], y[index]}', ha='center', va='bottom', color='red', fontsize=fz-2)

plt.savefig('curve_plot2.png')
plt.show()
