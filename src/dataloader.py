import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')


class PendulumDataloader():
    def __init__(self, training_class):
        self.configs = training_class.configs
        self.train_obs, self.train_targets, self.val_obs, self.val_targets, self.test_obs, self.test_targets = self.load_data()
        self.all_data = self.train_obs, self.train_targets, self.val_obs, self.val_targets, self.test_obs, self.test_targets
        self.train_data = self.train_obs, self.train_targets
        self.val_data = self.val_obs, self.val_targets
        self.test_data = self.test_obs, self.test_targets
        self.data_dict = {'train': self.train_data, 'val': self.val_data, 'test': self.test_data}
        self.data_sizes = {'train': 2000, 'val': 1000, 'test': 1000}


    def load_data(self):
        data = np.load('../data/pendulum_data.npz')
        return data['train_obs'], data['train_targets'], data['valid_obs'], data['valid_targets'], data['test_obs'], data['test_targets']


    def all_data_shape(self):
        for data in self.all_data:
            print(data.shape, data.dtype)
    

    def batch_data(self, mode='train'):
        if mode == 'train':
            batch_size = self.configs['batch_size']
            min = np.random.randint(0, 51)
            # max = np.random.randint(min + 40, 81)
            max = 80
        elif mode == 'val':
            batch_size = 256
            min, max = 0, 100
        first_dim_indices = np.random.choice(self.data_sizes[mode], size=batch_size, replace=False)
        second_dim_indices = np.random.choice(100, size=batch_size, replace=True)
        if self.configs['architecture'] == 'CNN':
            return self.data_dict[mode][0][first_dim_indices, second_dim_indices], self.data_dict[mode][1][first_dim_indices, second_dim_indices]
        else:
            batch_inputs = self.data_dict[mode][0][first_dim_indices][:,min:max]
            batch_labels = self.data_dict[mode][1][first_dim_indices][:,min:max]
            if mode == 'train':
                batch_inputs = np.array(np.pad(batch_inputs, ((0, 0), (80 - (max - min), 0), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0))
                batch_labels = np.array(np.pad(batch_labels, ((0, 0), (80 - (max - min), 0), (0, 0)), mode='constant', constant_values=0))
            mask = np.zeros_like(batch_labels)
            mask[:, -(max - min):] = 1
            return batch_inputs, batch_labels, mask


    def show_multiple_data(self, data_list):
        fig, axs = plt.subplots(1, len(data_list), figsize=(10, 3))
        for i, data in enumerate(data_list):
            axs[i].hist(data.flatten(), bins=50)
        plt.show()


    def plot_targets(self, targets):
        plt.figure(figsize=(5, 3))
        for i in range(targets.shape[2]):
            plt.plot(targets[0,:,i], label=f'Target {i+1}')
        plt.legend()
        plt.show()


    def plot_data_and_targets(self, data_type='train', plot_targets=(0,10)):
        fig, axs = plt.subplots(1, 3, figsize=(15, 3))
        
        for i, data in enumerate(self.data_dict[data_type]):
            axs[i].hist(data.flatten(), bins=50)
        
        for i in range(self.data_dict[data_type][1].shape[2]):
            for j in range(plot_targets[0], plot_targets[1]):
                axs[-1].plot(self.data_dict[data_type][1][j,:,i], label=f'Target {i+1}')
        plt.show()