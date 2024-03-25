from flax import linen as nn


class DenseModel(nn.Module):
	config: dict

	def setup(self):
		self.dense1 = nn.Dense(self.config['hidden_dim'])
		self.dense2 = nn.Dense(self.config['hidden_dim'])
		self.out_mean = nn.Dense(features=self.config['out_dim'])
		self.out_var = nn.Dense(features=self.config['out_dim'])
		self.dropout = nn.Dropout(rate=self.config['dropout'])

	def __call__(self, x, training=False):
		x = x.reshape((x.shape[0], -1))
		x = self.dense1(x)
		x = nn.relu(x)
		x = self.dense2(x)
		x = nn.relu(x)
		x = self.dropout(x, deterministic=not training)
		mean = self.out_mean(x)
		var = self.out_var(x)
		return mean, var


class CNN(nn.Module):
	config: dict

	def setup(self):
		self.conv1 = nn.Conv(features=32, kernel_size=(3, 3))
		self.conv2 = nn.Conv(features=64, kernel_size=(3, 3))
		self.dense = nn.Dense(features=self.config['dense_dim'])
		self.out_mean = nn.Dense(features=self.config['out_dim'])
		self.out_var = nn.Dense(features=self.config['out_dim'])
		self.layernorm = nn.LayerNorm()
		self.dropout = nn.Dropout(rate=self.config['dropout'])

	def __call__(self, x, training=False):
		x = self.conv1(x)
		x = nn.relu(x)
		x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
		x = self.conv2(x)
		x = nn.relu(x)
		x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
		x = x.reshape((x.shape[0], -1))
		x = self.dense(x)
		x = nn.relu(x)
		x = self.dropout(x, deterministic=not training)
		x = self.layernorm(x)
		mean = self.out_mean(x)
		var = self.out_var(x)
		return mean, var
