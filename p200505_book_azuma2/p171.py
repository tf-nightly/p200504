import cupy as np
import matplotlib.pyplot as plt
# import numpy as np

n_time = 20
n_mid = 128

eta = 0.01
clip_const = 0.02  # ノルムの最大値を決める定数。
beta = 2
epoch = 50
batch_size = 128

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def clip_grad(grads, max_norm):
  norm = np.sqrt(np.sum(grads * grads))
  r = max_norm / norm
  if r < 1:
    clipped_grads = grads * r
  else:
    clipped_grads = grads
  return clipped_grads

with open("kaijin20.txt", mode="r", encoding="utf-8") as f:
  text = f.read()
print("文字数: {}".format(len(text)))

chars_list = sorted(list(set(text)))
n_chars = len(chars_list)
print("文字数（重複なし）: {}".format(n_chars))

char_to_index = {}
index_to_char = {}
for i, char in enumerate(chars_list):
  char_to_index[char] = i
  index_to_char[i] = char

seq_chars = []
next_chars = []
for i in range(0, len(text) - n_time):
  seq_chars.append(text[i:i + n_time])
  next_chars.append(text[i + n_time])

input_data = np.zeros((len(seq_chars), n_time, n_chars), dtype=np.bool)
correct_data = np.zeros((len(seq_chars), n_chars), dtype=np.bool)
for i, chars in enumerate(seq_chars):
  # To one-hot representation.
  correct_data[i, char_to_index[next_chars[i]]] = 1
  for j, char in enumerate(chars):
    input_data[i, j, char_to_index[char]] = 1

class LSTMLayer:
  def __init__(self, n_upper, n):
    self.w = np.random.randn(4, n_upper, n) / np.sqrt(n_upper)
    self.v = np.random.randn(4, n, n) / np.sqrt(n)
    self.b = np.zeros((4, n))
  def forward(self, x, y_prev, c_prev):
    u = np.matmul(x, self.w) + np.matmul(y_prev, self.v) + self.b.reshape(4, 1, -1)

    a0 = sigmoid(u[0])
    a1 = sigmoid(u[1])
    a2 = np.tanh(u[2])
    a3 = sigmoid(u[3])
    self.gates = np.stack((a0, a1, a2, a3))

    self.c = a0 * c_prev + a1 * a2
    self.y = a3 * np.tanh(self.c)
  def bacward(self, x, y, c, y_prev, c_prev, gates, grad_y, grad_c):
    a0, a1, a2, a3 = gates
    tanh_c = np.tanh(c)
    r = grad_c + (grad_y * a3) * (1 - tanh_c ** 2)

    delta_a0 = r * c_prev * a0 * (1 - a0)
    delta_a1 = r * a2 * a1 * (1 - a1)
    delta_a2 = r * a1 * (1 - a2 ** 2)
    delta_a3 = grad_y * tanh_c * a3 * (1 - a3)

    deltas = np.stack((delta_a0, delta_a1, delta_a2, delta_a3))

    self.grad_w += np.matmul(x.T, deltas)
    self.grad_v += np.matmul(y_prev.T, deltas)
    self.grad_b += np.sum(deltas, axis=1)

    grad_x = np.matmul(deltas, self.w.transpose(0, 2, 1))
    self.grad_x = np.sum(grad_x, axis=0)

    grad_y_prev = np.matmul(deltas, self.v.transpose(0, 2, 1))
    self.grad_y_prev = np.sum(grad_y_prev, axis=0)

    self.grad_c_prev = r * a0
  def reset_sum_grad(self):
    self.grad_w = np.zeros_like(self.w)
    self.grad_v = np.zeros_like(self.v)
    self.grad_b = np.zeros_like(self.b)
  def update(self, eta):
    self.w -= eta * self.grad_w
    self.v -= eta * self.grad_v
    self.b -= eta * self.grad_b
  def clip_grads(self, clip_const):
    self.grad_w = clip_grad(self.grad_w, clip_const * np.sqrt(self.grad_w.size))
    self.grad_v = clip_grad(self.grad_v, clip_const * np.sqrt(self.grad_v.size))

class OutputLayer:
  """全結合層。"""
  def __init__(self, n_upper, n):
    self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
    self.b = np.zeros(n)
  def forward(self, x):
    self.x = x
    u = np.dot(x, self.w) + self.b
    # Softmax function.
    self.y = np.exp(u) / np.sum(np.exp(u), axis=1).reshape(-1, 1)
  def backward(self, t):
    delta = self.y - t
    self.grad_w = np.dot(self.x.T, delta)
    self.grad_b = np.sum(delta, axis=0)
    self.grad_x = np.dot(delta, self.w.T)
  def update(self, eta):
    self.w -= eta * self.grad_w
    self.b -= eta * self.grad_b

lstm_layer = LSTMLayer(n_chars, n_mid)
output_layer = OutputLayer(n_mid, n_chars)

def train(x_mb, t_mb):
  y_rnn = np.zeros((len(x_mb), n_time + 1, n_mid))
  c_rnn = np.zeros((len(x_mb), n_time + 1, n_mid))
  gates_rnn = np.zeros((4, len(x_mb), n_time, n_mid))
  y_prev = y_rnn[:, 0, :]
  c_prev = c_rnn[:, 0, :]
  for i in range(n_time):
    x = x_mb[:, i, :]
    lstm_layer.forward(x, y_prev, c_prev)

    y = lstm_layer.y
    y_rnn[:, i + 1, :] = y
    y_prev = y

    c = lstm_layer.c
    c_rnn[:, i + 1, :] = c
    c_prev = c

    gates = lstm_layer.gates
    gates_rnn[:, :, i, :] = gates
  output_layer.forward(y)
  output_layer.backward(t_mb)
  grad_y = output_layer.grad_x
  grad_c = np.zeros_like(lstm_layer.c)

  lstm_layer.reset_sum_grad()
  for i in reversed(range(n_time)):
    x = x_mb[:, i, :]
    y = y_rnn[:, i + 1, :]
    c = c_rnn[:, i + 1, :]
    y_prev = y_rnn[:, i, :]
    c_prev = c_rnn[:, i, :]
    gates = gates_rnn[:, :, i, :]

    lstm_layer.bacward(x, y, c, y_prev, c_prev, gates, grad_y, grad_c)
    grad_y = lstm_layer.grad_y_prev
    grad_c = lstm_layer.grad_c_prev

  lstm_layer.clip_grads(clip_const)
  lstm_layer.update(eta)
  output_layer.update(eta)

def predict(x_mb):
  y_prev = np.zeros((len(x_mb), n_mid))
  c_prev = np.zeros((len(x_mb), n_mid))
  for i in range(n_time):
    x = x_mb[:, i, :]
    lstm_layer.forward(x, y_prev, c_prev)
    y = lstm_layer.y
    y_prev = y
    c = lstm_layer.c
    c_prev = c
  output_layer.forward(y)
  return output_layer.y

def get_error(x, t):
  """交差エントロピー誤差を返す。"""
  limit = 1_000
  if len(x) > limit:  # 測定サンプル数の上限を設定する。
    index_random = np.arange(len(x))
    np.random.shuffle(index_random)
    x = x[index_random[:limit], :]
    t = t[index_random[:limit], :]
  y = predict(x)
  return -np.sum(t * np.log(y + 1e-7)) / batch_size

def create_text():
  prev_text = text[0:n_time]
  create_text = prev_text
  print("Seed: {}".format(create_text))

  for i in range(200):
    # one-hot表現にする。
    x = np.zeros((1, n_time, n_chars))
    for j, char in enumerate(prev_text):
      x[0, j, char_to_index[char]] = 1
    # 予測を行って次の文字を得る。
    y = predict(x)
    # 確率分布を調整する。
    p = y[0] ** beta
    # pの合計を1にする。
    p = p / np.sum(p)
    next_index = np.random.choice(len(p), size=1, p=p)
    next_char = index_to_char[int(next_index[0])]
    create_text += next_char
    prev_text = prev_text[1:] + next_char
  print(create_text)
  print()

error_record = []
# 1エポックあたりのバッチ数。
n_batch = len(input_data) // batch_size
for i in range(epoch):
  index_random = np.arange(len(input_data))
  np.random.shuffle(index_random)
  for j in range(n_batch):
    mb_index = index_random[j * batch_size:(j + 1) * batch_size]
    x_mb = input_data[mb_index, :]
    t_mb = correct_data[mb_index, :]
    train(x_mb, t_mb)
    a = "\rEpoch: {}/{}, {}/{}"
    a = a.format(i + 1, epoch, j + 1, n_batch)
    print(a, end="")
  error = get_error(input_data, correct_data)
  error_record.append(error)
  print("  Error: {}".format(error))
  create_text()
plt.plot(range(1, len(error_record) + 1), error_record, label="error")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.show()





























