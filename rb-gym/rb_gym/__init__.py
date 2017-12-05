import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='FrozenLakeA2C-v0',
    entry_point='rb_gym.envs:FrozenLakeEnv'
)

register(
    id='RadRoomSimple-v0',
    entry_point='rb_gym.envs:RadRoomSimple'
)
