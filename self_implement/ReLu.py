import numpy as np

class ReLu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


# x = np.array([[0.1, -0.5],[-2.0, 1.0]])
# print("x : ", x)

# relu = ReLu()

# out = relu.forward(x)

# print(out)