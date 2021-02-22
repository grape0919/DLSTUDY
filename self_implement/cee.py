import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
'''
손실 함수 = 학습 지표 : 얼마나 오답에 가까운지 나타냄

교차 엔트로피 오차
np.log를 개산할떄 아주 작은 수 인 delta를 더하는 이유는
np.log(0) 인 경우 inf 가 되기때문에 더이상 계산을 진행할 수 없음.


ont hot encoding 
        0   1   2   3   4   5
학습    0.0 0.1 0.5 0.3 0.1 0.0
정답    0.0 0.0 1.0 0.0 0.0 0.0

'''

t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
a = mean_squared_error(np.array(y), np.array(t))
b = cross_entropy_error(np.array(y), np.array(t))
print(a)
print(b)
