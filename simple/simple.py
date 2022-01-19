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