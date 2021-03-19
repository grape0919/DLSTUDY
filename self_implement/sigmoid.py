import numpy as np
import matplotlib.pylab as plt

class sigmoid():

    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y
        
    def backward(self, dy):
        out = dy * self.y * (1.0-self.y)
        return out


'''
Sigmoid  ([*]x * -1 -> [exp]exp(-x) -> [+]1+exp(-x) -> [/]1/(1+exp(-x)) -> y)
각 연산에 대한 역전파 계산식

'/' 는 y = 1/x 계산을 뜻한다.
이를 미분하면 ∂y/∂x = -1/xˆ2 = -yˆ2 이다.
따라서 역전파시 상류에서 온 값에 -yˆ2를 곱해서 하류로 보낸다.

'+' 는 별다른 계산 없이 상류값을 하류로 보낸다.

'exp' 는 y = exp(x) 연산고, 미분값은 ∂y/∂x = exp(x)이다.
상류로부터 온 값에 exp(x) 를 곱하여 하류로 보낸다.

'x' 는 순전파 때 입력값과 곱한값을 서로 바꿔서 곱한다.
순전파 때는 '하류값 * -1 = 상류값' 이었으나 역전파때는 '상류값 * -1 = 하류값' 이 된다.

위의 계산에 의해 Sigmoid 의 역전파는 (∂L/∂y)yˆ2exp(-x) 가 된다.
따라서 하나의 노드로 역전파를 구현 할 수 있다.

해당 식을 좀더 축양하면 아래와 같다.
(∂L/∂y)y(1-y)

'''