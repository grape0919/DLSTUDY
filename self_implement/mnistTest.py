import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle
from sigmoid import sigmoid
from softmax import softmax

def get_data():#mnist data 가져오기
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network(): #pickle 로 저장된 가중치, 편향  불러오기
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3

    y = softmax(a3)


x, t = get_data() # mnist data
network = init_network() # 가중치, 편향 세팅

accuracy_cnt = 0

for i in range(len(x)):
    y = predict(network, x[i])
    # 추론을 하면 결과값으로 
    # 10개의 출력 값이 나온다.( 0 ~ 9 중 어떤 것에 가까운지 추론)

    p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1
    
print("Accuracy:" + str(float(accuracy_cnt)/len(x)))