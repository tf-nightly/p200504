import cupy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

img_size = 8
n_time = 4
n_in = img_size
n_mid = 128
n_out = img_size
n_disp = 10

eta = 0.01
epochs = 201
batch_size = 32
interval = 10

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

digits = datasets.load_digits()
digits = np.asarray(digits.data)
digits_imgs = digits.reshape(-1, img_size, img_size)
digits_imgs /= 15

disp_imgs = digits_imgs[:n_disp]
train_imgs = digits_imgs[n_disp:]
n_sample_in_img = img_size - n_time
n_sample = len(train_imgs) * n_sample_in_img

input_data = np.zeros((n_sample, n_time, n_in))
correct_data = np.zeros((n_sample, n_out))
for i in range(len(train_imgs)):
  for j in range(n_sample_in_img):
    sample_id = i * n_sample_in_img + j
    input_data[sample_id] = train_imgs[i, j:j + n_time]
    correct_data[sample_id] = train_imgs[i, j + n_time]

x_train, x_test, t_train, t_test = train_test_split(input_data, correct_data)

class GRULayer:
  def __init__(self, n_upper, n):
    self.w = np.random.randn(3, n_upper, n) / np.sqrt(n_upper)
    self.v = np.random.randn(3, n, n) / np.sqrt(n)
  def forward(self, x, y_prev):
    a0 = sigmoid(np.dot(x, self.w[0]) + np.dot(y_prev, self.v[0]))
    a1 = sigmoid(np.dot(x, self.w[1]) + np.dot(y_prev, self.v[1]))
    a2 = np.tanh(np.dot(x, self.w[2]) + np.dot(a1 * y_prev, self.v[2]))
    self.gates = np.stack((a0, a1, a2))
    self.y = (1 - a0) * y_prev + a0 * a2
  def backward(self, x, y, y_prev, gates, grad_y):
    a0, a1, a2 = gates

    delta_a2 = grad_y * a0 * (1 - a2 ** 2)
    self.grad_w[2] += np.dot(x.T, delta_a2)
    self.grad_v[2] += np.dot((a1 * y_prev).T, delta_a2)

    delta_a0 = grad_y * (a2 - y_prev) * a0 * (1 - a0)
    self.grad_w[0] += np.dot(x.T, delta_a0)
    self.grad_v[0] += np.dot(y_prev.T, delta_a0)

    s = np.dot(delta_a2, self.v[2].T)
    delta_a1 = s * y_prev * a1 * (1 - a1)
    self.grad_w[1] += np.dot(x.T, delta_a1)
    self.grad_v[1] += np.dot(y_prev.T, delta_a1)

    self.grad_x = np.dot(delta_a0, self.w[0].T)
    + np.dot(delta_a1, self.w[1].T)
    + np.dot(delta_a2, self.w[2].T)

    self.grad_y_prev = np.dot(delta_a0, self.v[0].T)
    + np.dot(delta_a1, self.v[1].T)
    + a1 * s + grad_y * (1 - a0)
  def reset_sum_grad(self):
    self.grad_w = np.zeros_like(self.w)
    self.grad_v = np.zeros_like(self.v)
  def update(self, eta):
    self.w -= eta * self.grad_w
    self.v -= eta * self.grad_v

class OutputLayer:
  """全結合層。"""
  def __init__(self, n_upper, n):
    self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
    self.b = np.zeros(n)
  def forward(self, x):
    self.x = x
    u = np.dot(x, self.w) + self.b
    # Sigmoid function.
    self.y = 1 / (1 + np.exp(-u))
  def backward(self, t):
    delta = (self.y - t) * self.y * (1 - self.y)

    self.grad_w = np.dot(self.x.T, delta)
    self.grad_b = np.sum(delta, axis=0)
    self.grad_x = np.dot(delta, self.w.T)
  def update(self, eta):
    self.w -= eta * self.grad_w
    self.b -= eta * self.grad_b

gru_layer = GRULayer(n_in, n_mid)
output_layer = OutputLayer(n_mid, n_out)

def train(x_mb, t_mb):
  y_rnn = np.zeros((len(x_mb), n_time + 1, n_mid))
  gates_rnn = np.zeros((3, len(x_mb), n_time, n_mid))
  y_prev = y_rnn[:, 0, :]
  for i in range(n_time):
    x = x_mb[:, i, :]
    gru_layer.forward(x, y_prev)

    y = gru_layer.y
    y_rnn[:, i + 1, :] = y
    y_prev = y

    gates = gru_layer.gates
    gates_rnn[:, :, i, :] = gates
  output_layer.forward(y)

  output_layer.backward(t_mb)
  grad_y = output_layer.grad_x

  gru_layer.reset_sum_grad()
  for i in reversed(range(n_time)):
    x = x_mb[:, i, :]
    y = y_rnn[:, i + 1, :]
    y_prev = y_rnn[:, i, :]
    gates = gates_rnn[:, :, i, :]
    gru_layer.backward(x, y, y_prev, gates, grad_y)
    grad_y = gru_layer.grad_y_prev
  gru_layer.update(eta)
  output_layer.update(eta)

def predict(x_mb):
  y_prev = np.zeros((len(x_mb), n_mid))
  for i in range(n_time):
    x = x_mb[:, i, :]
    gru_layer.forward(x, y_prev)
    y = gru_layer.y
    y_prev = y
  output_layer.forward(y)
  return output_layer.y

def get_error(x, t):
  y = predict(x)
  return np.sum(np.square(y - t)) / len(x)

def generate_images():
  plt.figure(figsize=(10, 1))
  for i in range(n_disp):
    ax = plt.subplot(1, n_disp, i + 1)
    plt.imshow(disp_imgs[i].tolist(), cmap="Greys_r")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()

  gen_imgs = disp_imgs.copy()
  plt.figure(figsize=(10, 1))
  for i in range(n_disp):
    for j in range(n_sample_in_img):
      x = gen_imgs[i, j:j + n_time].reshape(1, n_time, img_size)
      gen_imgs[i, j + n_time] = predict(x)[0]
    ax = plt.subplot(1, n_disp, i + 1)
    plt.imshow(gen_imgs[i].tolist(), cmap="Greys_r")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()

n_batch = len(x_train) // batch_size
for i in range(epochs):
  index_random = np.arange(len(x_train))
  np.random.shuffle(index_random)
  for j in range(n_batch):
    mb_index = index_random[j * batch_size:(j + 1) * batch_size]
    x_mb = x_train[mb_index, :]
    t_mb = t_train[mb_index, :]
    train(x_mb, t_mb)
  if i % interval == 0:
    error_train = get_error(x_train, t_train)
    error_test = get_error(x_test, t_test)
    a = "Epoch: {}/{}, Error_train: {}, Error_test: {}"
    a = a.format(i, epochs - 1, error_train, error_test)
    print(a)
    generate_images()















