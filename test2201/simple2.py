import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import OrderedDict

def data_gen():
    Xin = pd.DataFrame(data=np.array([[0,0],[0,1],[1,0],[1,1]]),
                  columns=['x1', 'x2'])
    t_label = np.array([[0],[0],[0],[1]])
    return Xin, t_label

class perceptron:
    # パラメータの初期化
    def __init__(self, input_size, output_size):
        import numpy as np      
        self.params = {}
        self.params['W1'] = np.random.randn(input_size,output_size)
        self.params['b1'] = np.zeros(output_size)
        # レイヤの生成
        self.layers = OrderedDict()

        self.lastLayer = SoftmaxWithLoss()
    # 順伝播
    def forward(self, x):
        W1 = self.params['W1']
        b1 = self.params['b1']

        a1 = np.dot(x, W1) + b1
        y = a1     
        return y

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return 
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = x
        self.loss = self.y, self.t
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

class Softmax():
  """Softmax activation function.
  Example without mask:
  >>> inp = np.asarray([1., 2., 1.])
  >>> layer = tf.keras.layers.Softmax()
  >>> layer(inp).numpy()
  array([0.21194157, 0.5761169 , 0.21194157], dtype=float32)
  >>> mask = np.asarray([True, False, True], dtype=bool)
  >>> layer(inp, mask).numpy()
  array([0.5, 0. , 0.5], dtype=float32)
  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
  Output shape:
    Same shape as the input.
  Args:
    axis: Integer, or list of Integers, axis along which the softmax
      normalization is applied.
  Call arguments:
    inputs: The inputs, or logits to the softmax layer.
    mask: A boolean mask of the same shape as `inputs`. Defaults to `None`. The
      mask specifies 1 to keep and 0 to mask.
  Returns:
    softmaxed output with the same shape as `inputs`.
  """

  def __init__(self, axis=-1, **kwargs):
    super(Softmax, self).__init__(**kwargs)
    self.supports_masking = True
    self.axis = axis

  def call(self, inputs, mask=None):
    if mask is not None:
      # Since mask is 1.0 for positions we want to keep and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -1e.9 for masked positions.
      adder = (1.0 - math_ops.cast(mask, inputs.dtype)) * (
          _large_compatible_negative(inputs.dtype))

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      inputs += adder
    if isinstance(self.axis, (tuple, list)):
      if len(self.axis) > 1:
        return math_ops.exp(inputs - math_ops.reduce_logsumexp(
            inputs, axis=self.axis, keepdims=True))
      else:
        return backend.softmax(inputs, axis=self.axis[0])
    return backend.softmax(inputs, axis=self.axis)

  def get_config(self):
    config = {'axis': self.axis}
    base_config = super(Softmax, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    return input_shape