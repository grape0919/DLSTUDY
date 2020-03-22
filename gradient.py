import numpy as np

def numerical_gradient(f, x):   #수치 미분 - 함수의 접점의 접선을 구하기
    #아주 작은 상수(x변화량) 으로 접선을 구할 경우 큰 오차 발생
    #두개의 직선을 이용하여 접선을 구한다.
    #gradient = f(x-h)-f(x+h)/2h
    h = 1e-4 #접점에서 아주 작은 상수를 더하거나 빼서 x변화량을 구한다.
    grad = np.zeros_like(x)  #미분값을 결과를 넣기위한 배열 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad



# 경사 하강법
# 
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    # 초기 x 값에서 미분해나가며, 현재값보다 커질때까지 학습반복한다.
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
