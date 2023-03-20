import numpy as np
import matplotlib.pyplot as plt

n_BS = 30
y = []
for i in range(0, n_BS):
    y.append(np.random.lognormal(0, 8, 1)[0])
y_max = 100
y = np.array(y)
# y = np.clip(y, 0, y_max)


x = np.arange(0, n_BS, 1)

# plt.plot(x, y)
# plt.show()
print(y)
