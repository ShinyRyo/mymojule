class perceptron:
    # パラメータの初期化
    def __init__(self, w1,w2,b):
        import numpy as np      
        self.params = {}
        self.params['W1'] = np.array([[w1],[w2]])
        self.params['b1'] = np.array([b])

    # 順伝播
    def forward(self, x):
        W1 = self.params['W1']
        b1 = self.params['b1']

        a1 = np.dot(x, W1) + b1
        y = a1     
        return y