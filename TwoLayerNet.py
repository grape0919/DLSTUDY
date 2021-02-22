# 학습 알고리즘 구현
'''
 1단계 - 미니 배치
        데이터에서 무작위로 정해진 수 만큼 뽑는것이 미니배치.
        미니배치의 손실함 수 값을 줄이는 것이 목표
 2단계 - 기울기 
        미니배치의 손실함수 값을 줄이기위해 각 가중치 매개변수의
        기울기를 구해 기울기방향으로 값을 줄여나감.
 3단계 - 매개변수 값 갱신
 4단계 - 반복
'''
import numpy as np
from sigmoid import sigmoid
from softmax import softmax
from cee import cross_entropy_error
from gradient import numerical_gradient

print("Completed importing libs")

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        #W 는 가중치
        #b 는 편향

    def predict(self, x):       #설정된 가중치와 편향으로 예측(계산)
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)        #활성함수는 sigmoid 함수 사용
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)         #출력층 활성함수는 softmac 함수 사용

        return y                #결과 출력

    def loss(self, x, t):
        y = self.predict(x)     #계산

        return cross_entropy_error(y, t) #손실함수는 cee 사용


    def accuracy(self, x, t):    #정확도 측정 
        y = self.predict(x)     #예측(계산)
        y = np.argmax(y, axis=1)#최대값 인덱스 2차원 배열 내에서 찾음
        t = np.argmax(t, axis=1)#최대값 인덱스 2차원 배열 내에서 찾음

        accuracy = np.sum(y == t) / float(x.shape[0]) #최대 값이 같은(확률이 제일 높은 인덱스가 같을 경우) 1을 더해줌
        #   정답과 일치하는 y 의 개수 / 입력 개수
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        '''
        def loss_W(W)
            return self.loss(x, t)
        '''

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads

print("TwoLayerNet")