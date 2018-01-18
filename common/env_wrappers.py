import numpy as np
import gym
from gym.spaces import Box
import sys

from common.state_transform import StateVelCentr


def create_env(args):
    # env = Pendulum()
    env = gym.make('Pendulum-v0')
    return env
