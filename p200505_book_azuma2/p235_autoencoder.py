import cupy as np
import matplotlib.pyplot as plt
# import numpy as np
from sklearn import datasets

img_size = 8
n_in_out = img_size * img_size
assert n_in_out == 64
n_mid = 16

eta = 0.01
epochs = 41
batch_size = 32
interval = 4

digits_data = datasets.load_digits()
x_train = np.asarray(digits_data.data)
# 範囲を[0, 1]にする。
x_train /= 15
assert x_train.shape == (1_797, 64)

class BaseLayer:
  def update(self, eta):
    self.w -= eta * self.grad_w
    self.b -= eta * self.grad_b

class MiddleLayer(BaseLayer):
  def __init__(self, n_upper, n):
    # He initialization.
    self.w = np.random.randn(n_upper, n) * np.sqrt(2 / n_upper)
    assert self.w.shape == (64, 16)
    self.b = np.zeros(n)
    assert self.b.shape == (16,)
  def forward(self, x):
    self.x = x
    self.u = np.dot(x, self.w) + self.b
    self.y = np.where(self.u <= 0, 0, self.u)  # ReLU.
    assert self.y.shape[1:] == (16,)
  def backward(self, grad_y):
    delta = grad_y * np.where(self.u <= 0, 0, 1)

    self.grad_w = np.dot(self.x.T, delta)
    self.grad_b = np.sum(delta, axis=0)
    self.grad_x = np.dot(delta, self.w.T)

class OutputLayer(BaseLayer):
  def __init__(self, n_upper, n):
    # Xavier initialization.
    self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
    assert self.w.shape == (16, 64)
    self.b = np.zeros(n)
    assert self.b.shape == (64,)
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

middle_layer = MiddleLayer(n_in_out, n_mid)
output_layer = OutputLayer(n_mid, n_in_out)

def forward_propagation(x_mb):
  middle_layer.forward(x_mb)
  output_layer.forward(middle_layer.y)

def backpropagation(t_mb):
  output_layer.backward(t_mb)
  middle_layer.backward(output_layer.grad_x)

def update_params():
  middle_layer.update(eta)
  output_layer.update(eta)

def get_error(y, t):
  return 1.0 / 2.0 * np.sum(np.square(y - t))

error_record = []
n_batch = len(x_train) // batch_size
assert n_batch == 56
for i in range(epochs):
  index_random = np.arange(len(x_train))
  np.random.shuffle(index_random)
  for j in range(n_batch):
    mb_index = index_random[j * batch_size:(j + 1) * batch_size]
    x_mb = x_train[mb_index, :]

    forward_propagation(x_mb)
    backpropagation(x_mb)

    update_params()
  forward_propagation(x_train)
  error = get_error(output_layer.y, x_train)
  error_record.append(error)

  if i % interval == 0:
    a = "Epoch: {}/{}, Error: {}"
    a = a.format(i + 1, epochs, error)
    print(a)

plt.plot(range(1, len(error_record) + 1), error_record)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()







n_img = 10
middle_layer.forward(x_train[:n_img])
output_layer.forward(middle_layer.y)

plt.figure(figsize=(10, 3))
for i in range(n_img):
  ax = plt.subplot(3, n_img, i + 1)
  plt.imshow(x_train[i].reshape(img_size, -1).tolist(), cmap="Greys_r")
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  ax = plt.subplot(3, n_img, i + 1 + n_img)
  plt.imshow(middle_layer.y[i].reshape(4, -1).tolist(), cmap="Greys_r")
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  ax = plt.subplot(3, n_img, i + 1 + 2 * n_img)
  plt.imshow(output_layer.y[i].reshape(img_size, -1).tolist(), cmap="Greys_r")
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

















