import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

print_x = ['%.1f' % v for v in x]  
print_y = ['%.3f' % v for v in y]

print(print_y)
print(print_x)
plt.plot(x,y)
plt.ylim(-0.1, 6)
plt.show()