import matplotlib.pyplot as plt
import numpy as np

n_time = 8
n_in = 2
n_mid = 32
n_out = 1

eta = 0.01
n_learn = 5_001
interval = 500

max_num = 2**n_time
binaries = np.zeros((max_num, n_time), dtype=int)
for i in range(max_num):
  num10 = i  # 10進数表現。
  for j in range(n_time):
    pow2 = 2 ** (n_time - 1 - j)  # 2の累乗。
    binaries[i, j] = num10 // pow2
    num10 %= pow2
print(binaries)

class SimpleRNNLayer:
  def __init__(self, n_upper, n):
    self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
    self.v = np.random.randn(n, n) / np.sqrt(n)
    self.b = np.zeros(n)
  def forward(self, x, y_prev):
    u = np.dot(x, self.w) + np.dot(y_prev, self.v) + self.b
    self.y = np.tanh(u)
  def backward(self, x, y, y_prev, grad_y):
    delta = grad_y * (1 - y**2)

    self.grad_w += np.dot(x.T, delta)
    self.grad_v += np.dot(y_prev.T, delta)
    self.grad_b += np.sum(delta, axis=0)

    self.grad_x = np.dot(delta, self.w.T)
    self.grad_y_prev = np.dot(delta, self.v.T)
  def reset_sum_grad(self):
    self.grad_w = np.zeros_like(self.w)
    self.grad_v = np.zeros_like(self.v)
    self.grad_b = np.zeros_like(self.b)
  def update(self, eta):
    self.w -= eta * self.grad_w
    self.v -= eta * self.grad_v
    self.b -= eta * self.grad_b

class RNNOutputLayer:
  """全結合層。"""
  def __init__(self, n_upper, n):
    self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
    self.b = np.zeros(n)
  def forward(self, x):
    self.x = x
    u = np.dot(x, self.w) + self.b
    self.y = 1 / (1 + np.exp(-u))  # Sigmoid function.
  def backward(self, x, y, t):
    delta = (y - t) * y * (1 - y)

    self.grad_w += np.dot(x.T, delta)
    self.grad_b += np.sum(delta, axis=0)
    self.grad_x = np.dot(delta, self.w.T)
  def reset_sum_grad(self):
    self.grad_w = np.zeros_like(self.w)
    self.grad_b = np.zeros_like(self.b)
  def update(self, eta):
    self.w -= eta * self.grad_w
    self.b -= eta * self.grad_b

rnn_layer = SimpleRNNLayer(n_in, n_mid)
output_layer = RNNOutputLayer(n_mid, n_out)

def train(x_mb, t_mb):
  y_rnn = np.zeros((len(x_mb), n_time + 1, n_mid))
  y_out = np.zeros((len(x_mb), n_time, n_out))

  y_prev = y_rnn[:, 0, :]
  for i in range(n_time):
    x = x_mb[:, i, :]
    rnn_layer.forward(x, y_prev)
    y = rnn_layer.y
    y_rnn[:, i + 1, :] = y
    y_prev = y

    output_layer.forward(y)
    y_out[:, i, :] = output_layer.y

  output_layer.reset_sum_grad()
  rnn_layer.reset_sum_grad()
  grad_y = 0
  for i in reversed(range(n_time)):
    x = y_rnn[:, i + 1, :]
    y = y_out[:, i, :]
    t = t_mb[:, i, :]
    output_layer.backward(x, y, t)
    grad_x_out = output_layer.grad_x

    x = x_mb[:, i, :]
    y = y_rnn[:, i + 1, :]
    y_prev = y_rnn[:, i, :]
    rnn_layer.backward(x, y, y_prev, grad_y + grad_x_out)
    grad_y = rnn_layer.grad_y_prev

  rnn_layer.update(eta)
  output_layer.update(eta)
  return y_out

def get_error(y, t):
  """残差平方和を返す。"""
  return 1.0 / 2.0 * np.sum(np.square(y - t))

for i in range(n_learn):
  # 出題の10進数表現。和が最大値を超えないように2で割る。
  num1 = np.random.randint(max_num // 2)
  num2 = np.random.randint(max_num // 2)
  # 出題の2進数表現。
  x1 = binaries[num1]
  x2 = binaries[num2]
  x_in = np.zeros((1, n_time, n_in))
  assert x_in.shape == (1, 8, 2)
  x_in[0, :, 0] = x1
  x_in[0, :, 1] = x2

  # 第1軸で反転し、桁が小さい方を古い時刻にする。
  x_in = np.flip(x_in, axis=1)

  # 正解の2進数表現。
  t = binaries[num1 + num2]
  t_in = t.reshape(1, n_time, n_out)
  assert t_in.shape == (1, 8, 1)
  t_in = np.flip(t_in, axis=1)

  y_out = train(x_in, t_in)
  assert y_out.shape == (1, 8, 1)
  y = np.flip(y_out, axis=1).reshape(-1)

  error = get_error(y_out, t_in)

  if i % interval == 0:
    # 2進数の結果。浮動小数点数から2値にする。
    y2 = np.where(y < 0.5, 0, 1)
    # 10進数の結果。
    y10 = 0
    for j in range(len(y)):
      pow2 = 2 ** (n_time - 1 - j)
      y10 += y2[j] * pow2

    print("n_learn:", i)  # 現在のステップ数。
    print("error:", error)
    print("output:, y2")
    print("correct:", t)

    if (y2 == t).all():
      c = "\(^_^)/"
    else:
      c = "orz"
    a = "{}: {} + {} = {}"
    a = a.format(c, num1, num2, y10)
    print(a)
    print(15 * "-- ")



