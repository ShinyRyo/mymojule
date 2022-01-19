import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def perceptron(x1, x2, w1, w2, b):
    a = w1 * x1 + w2 * x2 + b
    return a

def step_function(x):
    try: #数値型を想定する時
        if x < 0:
            a=0
        else:
            a=1
    except: #pandasを想定する時はどっちでも大丈夫だが下の方がスマート
        #a = [1 if i>0 else 0 for i in x]
        a = np.array(x > 0, dtype=np.int)
    return a

def plot3d(x1, x2, w1, w2, b):
    #3Dで描写するための空間の準備
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    #3Dで描写するためのデータの準備
    x1_range=x1
    x2_range=x2
    x1 = np.arange(x1_range.min(), x1_range.max(), 0.01) #x1を0から1の範囲で0.01ずつ刻んでいきます
    x2 = np.arange(x2_range.min(), x1_range.max(), 0.01) #x2を0から1の範囲で0.01ずつ刻んでいきます
    X, Y = np.meshgrid(x1, x2)
    ##AND回路の出力を3Dで描写するためのデータを算出
    y=perceptron(X,Y, w1, w2, b)#AND(X,Y)
    Z=step_function(y)

    ax.plot_wireframe(X, Y, Z)
    plt.show()