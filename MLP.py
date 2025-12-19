import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 인공신경망(MLP) 클래스
class NeuralNetwork:
    def __init__(self, activation, layers=[784, 16, 16, 10]):
        self.layers = layers

        # 편향(Biases): 입력층을 제외한 2번째 층부터 존재
        # 초기에는 랜덤한 값으로 채움 -> 학습을 통해 편형값 조절
        self.biases = [np.random.randn(y, 1) for y in layers[1:]] # np.random.randn(x, y) -> x*y 2D array반환
        
        # 가중치(Weights): 앞 층(x)과 뒤 층(y) 사이의 연결
        # 편향과 마찬가지로 무작위로 채움 -> 학습을 통해 조절
        # 활성화 함수로 sigmoid를 사용하냐 relu를 사용하냐에 따라서 초기 가중치 값 설정 방식이 달라짐(relu 에서는 중간층 뉴런의 값이 0~1 사이일 필요가 없기 때문)
        if activation == "sigmoid":
            self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        elif activation == "relu": # 조금 더 값이 큰 무작위로 채우는 He 초기화
            self.weights = [np.random.randn(y, x) * np.sqrt(2/x) for x, y in zip(layers[:-1], layers[1:])]

        self.activation = activation

    # 활성화 함수 1: 시그모이드
    def sigmoid(self, z):
        # 오버플로우 방지를 위해 z값의 범위 제한
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    # 시그모이드 미분(역전파 알고리즘에서 사용)
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    # 활성화 함수 2: relu
    def relu(self, z):
        return np.maximum(0, z)
    
    # ReLU의 미분: 0보다 크면 1, 아니면 0
    def relu_prime(self, z):
        return (z > 0).astype(float)
    
    # 활성화 함수 두 개의 성능을 비교하기 위해 두 가지를 각각 적용해볼 수 있게 코드 작성
    def activate(self, z):
        if self.activation == "sigmoid": return self.sigmoid(z)
        else: return self.relu(z)

    def activate_prime(self, z):
        if self.activation == "sigmoid": return self.sigmoid_prime(z)
        else: return self.relu_prime(z)
    
    # 순전파 알고리즘 -> 입력으로 부터 출력 뉴런 활성화
    def feedforward(self, a):
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            # 순전파 계산 공식: a' = activate(Wa + b) (W=가중치, b=편향, a=현재 레이터, a'=다음 레이어, activate=활성화 함수)
            z = np.dot(w, a) + b
            a = self.activate(z)
        
        # relu를 써도 마지막 출력층은 0~1 사이의 값을 가져야 하므로, 항상 마지막 출력층 순전파는 sigmoid를 사용
        b, w = self.biases[-1], self.weights[-1]
        z = np.dot(w, a) + b
        a = self.sigmoid(z)

        return a
    
    # 역전파 알고리즘
    def backprop(self, x, y):
        # 기울기를 담을 리스트 (0으로 초기화)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # 1. 순전파 하며 값 저장하기
        a = x
        alist = [x] # 모든 층의 출력값(a)을 저장할 리스트 (입력층 포함)
        zs = []     # 모든 층의 가중입력(z = Wa+b)을 저장할 리스트
        
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, a) + b
            zs.append(z)
            a = self.activate(z)
            alist.append(a)
        b, w = self.biases[-1], self.weights[-1]
        z = np.dot(w, a) + b
        zs.append(z)
        a = self.sigmoid(z)
        alist.append(a)
        
        # 2. 역전파
        
        # 2-1. 출력층(L)의 오차(delta) 계산
        # 공식 1: delta^L = (a^L - y) * sigmoid'(z^L)
        # 비용함수로 MSE(평균제곱오차)를 미분하면 (a - y)가 됨.
        # Note: relu를 사용하더라도 출력층은 sigmoid를 사용하기 때문에 sigmoid'으로 계산해줘야함.
        delta = (alist[-1] - y) * self.sigmoid_prime(zs[-1])
        
        # 출력층의 기울기 저장
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, alist[-2].transpose())
        
        # 2-2. 역방향으로 이동하며 오차 전파
        # L-1, L-2, ... , 2번째 층까지 거꾸로 반복
        for i in range(2, len(self.layers)):
            z = zs[-i]
            sp = self.activate_prime(z)
            
            # 공식 2: delta^l = (W^{l+1}전치 * delta^{l+1}) * sigma'(z^l)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            
            # 해당 층의 기울기 저장 (공식 3, 4)
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, alist[-i-1].transpose())
            
        return (nabla_b, nabla_w)
    
    def update_mini_batch(self, mini_batch, eta):
        """
        mini_batch: (x, y) 튜플들의 리스트
        eta: 학습률 (Learning Rate)
        """
        # 기울기 누적 변수 초기화
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # 배치 안에 있는 모든 데이터에 대해 역전파 수행
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 구한 기울기를 누적시킴
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        # 가중치 업데이트
        # w = w - (eta / n) * gradient
        self.weights = [w - (eta / len(mini_batch)) * nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb 
                       for b, nb in zip(self.biases, nabla_b)]
        
    # 모델 훈련
    def fit(self, X, y, epochs, mini_batch_size, eta, show_process=False):
        """
        epochs: 전체 데이터를 몇 번 반복 학습할지
        mini_batch_size: 데이터를 몇 개씩 묶을지 (하나씩 하기엔 너무 학습이 오래걸리므로, 여러개씩 묶어서 한번에 학습)
        eta: 학습률
        """

        training_data = [(x.reshape(784, 1), y.reshape(10, 1)) for x, y in zip(X, y)]

        n = len(training_data)
        
        for i in range(epochs):
            # 데이터를 무작위로 섞음
            np.random.shuffle(training_data)
            
            # 미니배치로 잘라냄
            mini_batches = [
                training_data[j:j+mini_batch_size]
                for j in range(0, n, mini_batch_size)
            ]
            
            # 각 미니배치에 대해 학습 수행
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                
            if show_process: print(f"Epoch {i+1} 완료: {self.evaluate(X, y, show_process)}/{len(X)}")

    # 테스트 데이터를 받아 맞은 갯수 반환
    def evaluate(self, X, y, show_process=False):
        test_data = [(x.reshape(784, 1), y.reshape(10, 1)) for x, y in zip(X, y)]
        test_results = []
        for x, y in test_data:
            output = self.feedforward(x)
            
            prediction = np.argmax(output) 
            actual = np.argmax(y)
            
            test_results.append(prediction == actual)
        
        if show_process: self.visualize_predictions(test_data)
        return sum(test_results)
    
    # 신경망이 어떻게 예측하는지 보여주기 위해 실제 이미지와 예측값을 보여주는 메서드
    def visualize_predictions(self, test_data, num_samples=10):
        # 테스트 데이터 중 랜덤하게 샘플 선택
        indices = np.random.choice(len(test_data), num_samples, replace=False)
        
        plt.figure(figsize=(15, 3))
        
        for i, idx in enumerate(indices):
            x, y = test_data[idx]
            
            # 예측
            output = self.feedforward(x)
            prediction = np.argmax(output)
            actual = np.argmax(y)
            
            # 시각화
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(x.reshape(28, 28), cmap='gray')
            color = 'green' if prediction == actual else 'red'
            plt.title(f"Pred: {prediction}\nTrue: {actual}", color=color)
            plt.axis('off')
            
        plt.show()

# 데이터셋 불러오기(28*28 픽셀 손글씨 이미지 데이터셋 로드(numpy 배열 형태로))
print('downloading data set...')
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print('download complete!')

# 1. 데이터와 정답 레이블 분리
X=mnist.data
y=mnist.target

# 2. 데이터 전처리
y=y.astype(np.int8) # 데이터 형태 변경
X=X/255.0 # 0~1 사이로 데이터 정규화
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)

from sklearn.preprocessing import OneHotEncoder # 원-핫 인코딩

enc = OneHotEncoder()
y_train_encoded = enc.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test_encoded = enc.transform(y_test.reshape(-1, 1)).toarray()

# 3. 모델 생성 및 학습
sigmoidNN = NeuralNetwork(activation="sigmoid")
sigmoidNN.fit(X_train, y_train_encoded, epochs=10, mini_batch_size=10, eta=3.0)
print(f"accuracy(sigmoid): {sigmoidNN.evaluate(X_test, y_test_encoded, show_process=True) / len(X_test) * 100}%")

reluNN = NeuralNetwork(activation="relu")
reluNN.fit(X_train, y_train_encoded, epochs=10, mini_batch_size=10, eta=0.1)
print(f"accuracy(relu): {reluNN.evaluate(X_test, y_test_encoded, show_process=True) / len(X_test) * 100}%")

# 4. sigmoid와 relu 비교
hyper_parameter = [
    {"epochs":10, "mini_batch_size":10, "eta":0.01}, # 낮은 학습률
    {"epochs":10, "mini_batch_size":10, "eta":0.1},  # 중간 학습률
    {"epochs":10, "mini_batch_size":10, "eta":0.5},  # 높은 학습률
    {"epochs":10, "mini_batch_size":10, "eta":3.0},  # 매우 높은 학습률
]

sig_accuracy = []
relu_accuracy = []
for param in hyper_parameter:
    sigmoidNN = NeuralNetwork(activation="sigmoid")
    reluNN = NeuralNetwork(activation="relu")
    sigmoidNN.fit(X_train, y_train_encoded, epochs=param["epochs"], mini_batch_size=param["mini_batch_size"], eta=param["eta"])
    reluNN.fit(X_train, y_train_encoded, epochs=param["epochs"], mini_batch_size=param["mini_batch_size"], eta=param["eta"])
    sig_accuracy.append((sigmoidNN.evaluate(X_test, y_test_encoded)) / len(X_test) * 100)
    relu_accuracy.append((reluNN.evaluate(X_test, y_test_encoded)) / len(X_test) * 100)

x = np.arange(len(hyper_parameter))  # X축 위치 (0, 1, 2, 3...)
width = 0.35  # 막대 너비 설정

fig, ax = plt.subplots(figsize=(12, 6))

# 실험 결과 시각화
rects1 = ax.bar(x - width/2, sig_accuracy, width, label='Sigmoid', color='skyblue')
rects2 = ax.bar(x + width/2, relu_accuracy, width, label='ReLU', color='lightcoral')

# X축 라벨 만들기
labels = []
for p in hyper_parameter:
    label = f"Epochs: {p['epochs']}\nBatch: {p['mini_batch_size']}\nEta: {p['eta']}"
    labels.append(label)

# 축 및 제목 설정
ax.set_ylabel('Accuracy (%)')
ax.set_title('Sigmoid vs ReLU Accuracy by Hyperparameters')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 100) # Y축 범위를 0~100%로 고정
ax.grid(axis='y', linestyle='--', alpha=0.7) # 가로 격자 추가

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

