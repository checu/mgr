import numpy as np
import math

labels = np.random.randint(2, size=1000)

grid = np.indices((10, 100))

grid_x_flatted = grid[0].flatten()
grid_y_flatted = grid[1].flatten()

x_mean = 180
y_mean = 240

x_0_sum = 0
x_1_sum = 0
y_0_sum = 0
y_1_sum = 0

sum_0 = 0
sum_1 = 0

n_0 = 0
n_1 = 0


for c in range(len(labels)):

    if labels[c] == 0:

        x_0_sum = x_0_sum + (grid_x_flatted[c] - x_mean)**2
        y_0_sum = y_0_sum + (grid_y_flatted[c] - y_mean)**2
        sum_0 = sum_0 =(x_0_sum + y_0_sum)
        n_0 += 1

    else:

        x_1_sum = x_1_sum + (grid_x_flatted[c] - x_mean)**2
        y_1_sum = y_1_sum + (grid_y_flatted[c] - y_mean)**2
        sum_1 = sum_1 = (x_1_sum + y_1_sum)
        n_1 += 1

RMS_0 = math.sqrt(sum_0/n_0)

RMS_1 = math.sqrt(sum_1/n_1)

print("RMS_0:",RMS_0,"RMS_1:",RMS_1)
