from .envs import CassieEnv
from .envs.cassie_env_noclock import CassieEnv_noclock

from gym.envs.registration import register

register(
    id='Cassie-v0',
    entry_point='gym_cassie.envs.cassie_env:CassieEnv',
)

register(
    id='Cassie-v1',
    entry_point='gym_cassie.envs.cassie_env_noclock:CassieEnv_noclock',
)

