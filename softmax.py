'''
출력층에서는 활성화 함수가 달라진다.
일반적으로 회귀에는 *항등 함수*
분류에는 *소프트맥스 함수*를 사용한다.
'''

'''
소프트맥스 함수는 
분자는 입력신호의 지수함수,
분모는 모든 입력신호의 지수함수의 합이다
'''

import numpy as np


a = np.array([0.3, 2.9, 4.0])


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a

    return y

