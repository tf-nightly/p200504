import numpy as np
import matplotlib.pyplot as plt

n_time = 10
n_in = 1
n_mid = 20
n_out = 1

eta = 0.001
epochs = 51
# epochs = 510
batch_size = 8
interval = 5
# interval = 50

# $$[-2\pi, 2\pi]$$ つまり2周期。
sin_x = np.linspace(-2*np.pi, 2*np.pi)
assert sin_x.shape == (50,)
# ノイズを加える。ノイズは標準偏差0.1の正規分布とする。
sin_y = np.sin(sin_x) + 0.1*np.random.randn(len(sin_x))
# sin_y = np.sin(sin_x) + 0.01*np.random.randn(len(sin_x))
assert sin_y.shape == (50,)

n_sample = len(sin_x)-n_time
assert n_sample == 40
input_data = np.zeros((n_sample, n_time, n_in))
assert input_data.shape == (40, 10, 1)
correct_data = np.zeros((n_sample, n_out))
assert correct_data.shape == (40, 1)
for i in range(0, n_sample):  # 各標本を作成する。
  a = sin_y[i:i + n_time]
  assert a.shape == (10,)
  input_data[i] = a.reshape(-1, 1)
  assert input_data[i].shape == (10, 1)
  correct_data[i] = sin_y[i + n_time:i + n_time + 1]
  assert correct_data[i].shape == (1,)

class SimpleRNNLayer:
  """RNN (recurrent neural network) 層。"""
  def __init__(self, n_upper, n):
    # Xavierの初期値。$$\sim N(0, \frac{1}{n}).$$
    self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
    # Xavierの初期値。$$\sim N(0, \frac{1}{n}).$$
    self.v = np.random.randn(n, n) / np.sqrt(n)
    self.b = np.zeros(n)
  def forward(self, x, y_prev):
    # $$$ U^{(t)} = X^{(t)}W + Y^{(t-1)}V + B. $$$
    u = np.dot(x, self.w) + np.dot(y_prev, self.v) + self.b
    # $$$ Y^{(t)} = \phi(U^{(t)}) $$$
    # tanh:
    # $$$\phi(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}.$$$
    self.y = np.tanh(u)
  def backward(self, x, y, y_prev, grad_y):
    # $$$ \delta^{(t)} = \frac{\partial E}{\partial u^{(t)}} = \frac{\partial E}{\partial y^{(t)}} \frac{\partial y^{(t)}}{\partial u^{(t)}}. $$$
    # Where
    # $$$ \frac{\partial y^{(t)}}{\partial u^{(t)}} = \phi' = (1 - y^2). $$$
    delta = grad_y * (1 - y**2)

    # $$$ \frac{\partial E}{\partial W} = \sum_{t=1}^{\tau} X^{(t)\rm{T}} \Delta^{(t)}. $$$
    self.grad_w += np.dot(x.T, delta)
    # $$$ \frac{\partial E}{\partial V} = \sum_{t=1}^{\tau} Y^{(t-1)\rm{T}} \Delta^{(t)}. $$$
    self.grad_v += np.dot(y_prev.T, delta)
    # $$$ \frac{\partial E}{\partial B} = \left( \sum_{t=1}^{\tau} \sum_{k=1}^{h} \delta_{ki} \ ... \right). $$$
    self.grad_b += np.sum(delta, axis=0)

    # $$$ \frac{\partial E}{\partial X^{(t)}} = \Delta^{(t)} W^{\rm{T}}. $$$
    self.grad_x = np.dot(delta, self.w.T)
    # $$$ \frac{\partial E}{\partial Y^{(t-1)}} = \Delta^{(t)} V^{\rm{T}}. $$$
    self.grad_y_prev = np.dot(delta, self.v.T)
  def reset_sum_grad(self):
    self.grad_w = np.zeros_like(self.w)
    self.grad_v = np.zeros_like(self.v)
    self.grad_b = np.zeros_like(self.b)
  def update(self, eta):
    self.w -= eta * self.grad_w
    self.v -= eta * self.grad_v
    self.b -= eta * self.grad_b

class OutputLayer:
  """全結合層。"""
  def __init__(self, n_upper, n):
    # Xavierの初期値。$$\sim N(0, \frac{1}{n}).$$
    self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
    self.b = np.zeros(n)
  def forward(self, x):
    self.x = x
    # $$ U = XW + B. $$
    u = np.dot(x, self.w) + self.b
    # 恒等関数。$$ \phi(x) = x. $$
    self.y = u
  def backward(self, t):
    delta = self.y - t

    # $$$ \frac{\partial E}{\partial W} = X^{\rm{T}} \Delta. $$$
    self.grad_w = np.dot(self.x.T, delta)
    # $$$ \frac{\partial E}{\partial B} = \left( \sum_{k=1}^{h} \delta_{ki} \ ... \right). $$$
    self.grad_b = np.sum(delta, axis=0)
    # $$$ \frac{\partial E}{\partial X} = \Delta W^{\rm{T}}. $$$
    self.grad_x = np.dot(delta, self.w.T)
  def update(self, eta):
    self.w -= eta * self.grad_w
    self.b -= eta * self.grad_b

rnn_layer = SimpleRNNLayer(n_in, n_mid)
output_layer = OutputLayer(n_mid, n_out)

def train(x_mb, t_mb):
  y_rnn = np.zeros((len(x_mb), n_time+1, n_mid))
  y_prev = y_rnn[:, 0, :]
  for i in range(n_time):
    x = x_mb[:, i, :]
    rnn_layer.forward(x, y_prev)
    y = rnn_layer.y
    y_rnn[:, i+1, :] = y
    y_prev = y

  output_layer.forward(y)

  output_layer.backward(t_mb)
  grad_y = output_layer.grad_x

  rnn_layer.reset_sum_grad()
  for i in reversed(range(n_time)):
    x = x_mb[:, i, :]
    y = y_rnn[:, i+1, :]
    y_prev = y_rnn[:, i, :]
    rnn_layer.backward(x, y, y_prev, grad_y)
    grad_y = rnn_layer.grad_y_prev

  rnn_layer.update(eta)
  output_layer.update(eta)

def predict(x_mb):
  y_prev = np.zeros((len(x_mb), n_mid))
  for i in range(n_time):
    x = x_mb[:, i, :]
    rnn_layer.forward(x, y_prev)
    y = rnn_layer.y
    y_prev = y
  output_layer.forward(y)
  return output_layer.y

def get_error(x, t):
  """残差平方和 (residual sum of squares) を返す。"""
  y = predict(x)
  # $$$ L = \frac12\sum_b^B (y_b - t_b)^2. $$$
  return 1.0 / 2.0 * np.sum(np.square(y - t))

error_record = []
n_batch = len(input_data) // batch_size
assert n_batch == 5
for i in range(epochs):
  index_random = np.arange(len(input_data))
  assert index_random.shape == (40,)
  np.random.shuffle(index_random)

  # 訓練する。
  for j in range(n_batch):  # それぞれのバッチについて。
    mb_index = index_random[j*batch_size : (j+1)*batch_size]
    x_mb = input_data[mb_index, :]
    assert x_mb.shape == (8, 10, 1)
    t_mb = correct_data[mb_index, :]
    assert t_mb.shape == (8, 1)
    train(x_mb, t_mb)
  # 1エポックの訓練が終わった。

  error = get_error(input_data, correct_data)
  error_record.append(error)

  if i % interval == 0:
    a = "Epoch: {}/{}, Error: {}"
    a = a.format(i + 1, epochs, error)
    print(a)
    predicted = input_data[0].reshape(-1).tolist()
    for i in range(n_sample):
      x = np.array(predicted[-n_time:]).reshape(1, n_time, 1)
      y = predict(x)
      predicted.append(float(y[0, 0]))

    plt.plot(range(len(sin_y)), sin_y.tolist(), label=r"$y$")
    plt.plot(range(len(predicted)), predicted, label=r"$\hat{y}$")
    plt.legend()
    plt.show()

plt.plot(range(1, len(error_record)+1), error_record)
plt.xlabel("epochs")
plt.ylabel("Residual Sum of Squares")
plt.show()
































