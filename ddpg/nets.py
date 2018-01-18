import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from common.nets import LinearNet
from common.modules.NoisyLinear import NoisyLinear

def to_torch_variable(x, dtype='float32'):
    if isinstance(x, Variable):
        return x
    if not isinstance(x, torch.FloatTensor):
        x = torch.from_numpy(np.asarray(x, dtype=dtype))
    # if self.gpu:
    #     x = x.cuda()
    return Variable(x)

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, n_observation, n_action,
                 layers, activation=torch.nn.ELU,
                 layer_norm=False,
                 parameters_noise=False, parameters_noise_factorised=False,
                 last_activation=torch.nn.Tanh, init_w=3e-3):
        super(Actor, self).__init__()

        if parameters_noise:
            def linear_layer(x_in, x_out):
                return NoisyLinear(x_in, x_out, factorised=parameters_noise_factorised)
        else:
            linear_layer = nn.Linear

        self.feature_net = LinearNet(
            layers=[n_observation] + layers,
            activation=activation,
            layer_norm=layer_norm,
            linear_layer=linear_layer)
        self.policy_net = LinearNet(
            layers=[self.feature_net.output_shape, n_action],
            activation=last_activation,
            layer_norm=False
        )
        self.init_weights(init_w)

    def init_weights(self, init_w):
        for layer in self.feature_net.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data = fanin_init(layer.weight.data.size())

        for layer in self.feature_net.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-init_w, init_w)

    def forward(self, observation):
        x = to_torch_variable(observation)
        x = self.feature_net.forward(x)
        x = self.policy_net.forward(x)
        return x


class Critic(nn.Module):
    def __init__(self, n_observation, n_action,
                 layers, activation=torch.nn.ELU,
                 layer_norm=False,
                 parameters_noise=False, parameters_noise_factorised=False,
                 init_w=3e-3):
        super(Critic, self).__init__()

        if parameters_noise:
            def linear_layer(x_in, x_out):
                return NoisyLinear(x_in, x_out, factorised=parameters_noise_factorised)
        else:
            linear_layer = nn.Linear

        self.feature_net = LinearNet(
            layers=[n_observation + n_action] + layers,
            activation=activation,
            layer_norm=layer_norm,
            linear_layer=linear_layer)
        self.value_net = nn.Linear(self.feature_net.output_shape, 1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        for layer in self.feature_net.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data = fanin_init(layer.weight.data.size())

        self.value_net.weight.data.uniform_(-init_w, init_w)

    def forward(self, observation, action):
        x = torch.cat((observation, action), dim=1)
        x = self.feature_net.forward(x)
        x = self.value_net.forward(x)
        return x


class Base(nn.Module):
    def __init__(self, n_observation, n_action,
                 layers, activation=torch.nn.ELU,
                 layer_norm=False,
                 parameters_noise=False, parameters_noise_factorised=False,
                 last_activation=torch.nn.Tanh, init_w=3e-3):
        super(Base, self).__init__()

        if parameters_noise:
            def linear_layer(x_in, x_out):
                return NoisyLinear(x_in, x_out, factorised=parameters_noise_factorised)
        else:
            linear_layer = nn.Linear

        self.feature_net = LinearNet(
            layers=[n_observation] + layers,
            activation=activation,
            layer_norm=layer_norm,
            linear_layer=linear_layer)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        for layer in self.feature_net.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data = fanin_init(layer.weight.data.size())

        for layer in self.feature_net.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-init_w, init_w)

    def forward(self, observation):
        x = to_torch_variable(observation)
        x = self.feature_net.forward(x)
        return x

class CriticHead(nn.Module):
    def __init__(self, base, n_observation, n_action,
                 layers, activation=torch.nn.ELU,
                 layer_norm=False,
                 parameters_noise=False, parameters_noise_factorised=False,
                 init_w=3e-3):
        super(CriticHead, self).__init__()
        self.base = base
        self.value_net = nn.Linear(self.base.feature_net.output_shape, 1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.value_net.weight.data.uniform_(-init_w, init_w)

    def forward(self, observation):
        x = self.base.forward(observation)
        # x = torch.cat((x, action), dim=1)
        x = self.value_net.forward(x)
        return x



class ActorHead(nn.Module):
    def __init__(self, base, n_observation, n_action,
                 layers, activation=torch.nn.ELU,
                 layer_norm=False,
                 parameters_noise=False, parameters_noise_factorised=False,
                 last_activation=torch.nn.Tanh, init_w=3e-3):
        super(ActorHead, self).__init__()
        self.base = base

        self.policy_net = LinearNet(
            layers=[self.base.feature_net.output_shape, n_action],
            activation=last_activation,
            layer_norm=False
        )
        self.init_weights(init_w)

    def init_weights(self, init_w):
        for layer in self.policy_net.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-init_w, init_w)

    def forward(self, observation):
        x = observation
        x = self.base.forward(x)
        x = self.policy_net.forward(x)
        return x



class DynamicsHead(nn.Module):
    def __init__(self, base, n_observation, n_action,
                 layers, activation=torch.nn.ELU,
                 layer_norm=False,
                 parameters_noise=False, parameters_noise_factorised=False,
                 init_w=3e-3):
        super(DynamicsHead, self).__init__()
        self.base = base
        self.value_net = nn.Linear(self.base.feature_net.output_shape + n_action, n_observation)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.value_net.weight.data.uniform_(-init_w, init_w)

    def forward(self, observation, action):
        action = to_torch_variable(action)
        x = self.base.forward(observation)
        x = torch.cat((x, action), dim=1)
        x = self.value_net.forward(x)
        return x
