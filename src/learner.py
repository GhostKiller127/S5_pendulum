from typing import Any
import jax
import jax.numpy as np
from jax import random
from flax.training import train_state
import functools
import optax
from architectures import DenseModel, CNN
from s5 import S5


class Learner:
    def __init__(self, training_class):
        self.configs = training_class.configs
        self.main_rng = random.PRNGKey(self.configs['jax_seed'])
        self.training_steps = self.configs['warmup_steps'] + self.configs['training_steps']
        self.architecture_parameters = self.configs['parameters'][self.configs['architecture']]
        if self.configs['architecture'] == 'S5':
            self.architecture = S5(self.architecture_parameters).s5
        elif self.configs['architecture'] == 'CNN':
            self.architecture = CNN(self.architecture_parameters)
        elif self.configs['architecture'] == 'Dense':
            self.architecture = DenseModel(self.architecture_parameters)
        self.parameters = self.initialize_parameters()
        if self.configs['lr_finder'] == True:
            self.learning_rate_fn = self.exponential_growth_fn()
        else:
            self.learning_rate_fn = self.create_learning_rate_fn()
        self.train_state = self.create_train_state(self.learning_rate_fn)

    
    def show_parameters(self):
        self.main_rng, init_rng, drop_rng = random.split(self.main_rng, num=3)
        if self.configs['architecture'] == 'S5':
            training = True
            return print(self.architecture.tabulate({"params": init_rng, "dropout": drop_rng}, np.ones(self.architecture_parameters['input_shape']),
                                                    training, vcompute_flops=True, compute_vjp_flops=True))
        else:
            return print(self.architecture.tabulate({"params": init_rng, "dropout": drop_rng}, np.ones(self.architecture_parameters['input_shape']), compute_flops=True, compute_vjp_flops=True))


    def initialize_parameters(self):
        self.main_rng, init_rng, drop_rng = random.split(self.main_rng, num=3)
        if self.configs['architecture'] == 'S5':
            training = True
            variables = self.architecture.init({"params": init_rng, "dropout": drop_rng}, np.ones(self.configs['parameters']['S5']['input_shape']), training)
            params = variables["params"]
            if self.architecture_parameters["batchnorm"]:
                batch_stats = variables['batch_stats']
                return params, batch_stats
            return params
        else:
            return self.architecture.init({"params": init_rng, "dropout": drop_rng}, np.ones(self.configs['parameters'][self.configs['architecture']]['input_shape']))['params']


    def create_train_state(self, learning_rate_fn):
        tx = optax.adamw(learning_rate_fn, weight_decay=self.configs['weight_decay'])
        if self.configs['architecture'] == 'S5' and self.architecture_parameters["batchnorm"]:
            class TrainState(train_state.TrainState):
                batch_stats: Any
            return TrainState.create(apply_fn=self.architecture.apply, params=self.parameters[0], tx=tx, batch_stats=self.parameters[1])
        else:
            return train_state.TrainState.create(apply_fn=self.architecture.apply, params=self.parameters, tx=tx)


    def print_shapes(self, path=[]):
        def print_shapes_temp(parameters, current_path):
            for k, v in parameters.items():
                if isinstance(v, dict):
                    print_shapes_temp(v, current_path + [k])
                elif isinstance(v, np.ndarray):
                    print('.'.join(current_path + [k]), v.shape, v.dtype)
        if self.configs['architecture'] == 'S5' and self.architecture_parameters["batchnorm"]:
            print_shapes_temp(self.parameters[0], path)
        else:
            print_shapes_temp(self.parameters, path)

    
    def map_nested_fn(self, fn):
        def map_fn(nested_dict):
            return {k: (map_fn(v) if hasattr(v, "keys") else fn(k, v)) for k, v in nested_dict.items()}
        return map_fn


    def get_trainable_parameters(self):
        fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
        if self.configs['architecture'] == 'S5' and self.architecture_parameters["batchnorm"]:
            params = self.parameters[0]
        else:
            params = self.parameters
        param_sizes = self.map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
        print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")
    

    def create_learning_rate_fn(self):
        warmup_fn = optax.linear_schedule(init_value=0., end_value=self.configs['learning_rate'], transition_steps=self.configs['warmup_steps'])
        cosine_fn = optax.cosine_decay_schedule(init_value=self.configs['learning_rate'], decay_steps=self.configs['training_steps'])
        schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn], boundaries=[self.configs['warmup_steps']])
        return schedule_fn
    

    def exponential_growth_fn(self):
        self.initial_lr = 1e-8
        self.end_lr = 1e-2
        self.training_steps = 10000
        def create_learning_rate_fn(step):
            return self.initial_lr * ((self.end_lr / self.initial_lr) ** (step / self.training_steps))
        return create_learning_rate_fn
    

    def train_batch(self, batch_inputs, batch_labels, steps, sequence_length):
        if self.configs['architecture'] == 'S5' and self.architecture_parameters["batchnorm"]:
            batchnorm = True
        else:
            batchnorm = False
        self.main_rng, drop_rng = random.split(self.main_rng, num=2)
        self.train_state, loss, lr = train_batch_(self.train_state, drop_rng, batch_inputs, batch_labels, self.learning_rate_fn, steps, batchnorm, sequence_length)
        return loss, lr
    

    def validate_batch(self, batch_inputs, batch_labels):
        if self.configs['architecture'] == 'S5' and self.architecture_parameters["batchnorm"]:
            batchnorm = True
        else:
            batchnorm = False
        loss = validate_batch_(self.train_state, batch_inputs, batch_labels, batchnorm)
        return loss


def GaussianNLLL(mean, var, batch_labels):
    loss = np.mean(0.5 * (np.log(np.maximum(var, np.ones_like(var))) + ((mean - batch_labels) ** 2) / np.maximum(var, np.ones_like(var))))
    return loss


def mean_abs_error(mean, batch_labels):
    loss = np.mean(np.abs(mean - batch_labels))
    return loss


def stepwise_mean_abs_error(mean, batch_labels):
    loss = np.mean(np.abs(mean - batch_labels), axis=0)
    return loss


def mean_squared_error(mean, batch_labels):
    loss = np.mean(np.square(mean - batch_labels))
    return loss


@functools.partial(jax.jit, static_argnums=(4, 6))
def train_batch_(state, rng, batch_inputs, batch_labels, learning_rate_fn, step, batchnorm, mask):
    def loss_fn(params):
        training = True
        if batchnorm:
            (mean, var), updates = state.apply_fn({"params": params, "batch_stats": state.batch_stats}, batch_inputs, training, rngs={"dropout": rng}, mutable=["intermediates", "batch_stats"])
        else:
            (mean, var), updates = state.apply_fn({"params": params}, batch_inputs, training, rngs={"dropout": rng}, mutable=["intermediates"])
        mean_masked = mean * mask
        # var_masked = var * mask
        labels_masked = batch_labels * mask
        loss = mean_abs_error(mean_masked, labels_masked)
        # loss = GaussianNLLL(mean_masked, var_masked, labels_masked)
        return loss, updates
    (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=updates["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)
    lr = learning_rate_fn(step)
    return state, loss, lr


@functools.partial(jax.jit, static_argnums=3)
def validate_batch_(state, batch_inputs, batch_labels, batchnorm):
    training = False
    if batchnorm:
        mean, var = state.apply_fn({'params': state.params, "batch_stats": state.batch_stats}, batch_inputs, training)
    else:
        mean, var = state.apply_fn({'params': state.params}, batch_inputs, training)
    loss = stepwise_mean_abs_error(mean, batch_labels)
    return loss

