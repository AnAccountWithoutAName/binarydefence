from typing import Any
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    rewards: jax.Array  # (2, n) float32: rewards[a, i] = 1 iff a == hidden_bit_string[i]
    index: jax.Array  # scalar int32: the index the agent has to guess the bit of (the "level")


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 1


class BinaryDefence(environment.Environment[EnvState, EnvParams]):
    """One-step game: guess the hidden bit at the level's index, reward 1 if right, else 0.

    The level is the index. reset_to_level starts an episode on a given index (used by UED);
    reset_env draws the index uniformly. The observation is the one-hot index, so a policy is a
    map index -> bit, the RL counterpart of a bitstring individual.
    """

    def __init__(self, hidden_bits: jax.Array):
        super().__init__()
        self.hidden_bit_string = jnp.asarray(hidden_bits, dtype=jnp.int32)
        self.n = int(self.hidden_bit_string.shape[0])

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        reward = state.rewards[action, state.index]
        state = state.replace(time=state.time + 1)
        done = self.is_terminated(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_to_level(
        self, key: jax.Array, index: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Start an episode on the given index (the level)."""
        hidden = self.hidden_bit_string.astype(jnp.float32)
        rewards = jnp.stack((1.0 - hidden, hidden))
        state = EnvState(time=0, rewards=rewards, index=jnp.asarray(index, dtype=jnp.int32))
        return self.get_obs(state), state

    def reset_env(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:
        """Reset to a uniformly random index."""
        index = jax.random.randint(key, (), 0, self.n)
        return self.reset_to_level(key, index, params)

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """One-hot encoding of the index."""
        return jax.nn.one_hot(state.index, self.n, dtype=jnp.float32)

    def is_terminated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Every episode is a single guess."""
        return jnp.array(True)

    @property
    def name(self) -> str:
        return "BinaryDefence"

    @property
    def num_actions(self) -> int:
        return 2

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0, 1, (self.n,))

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "rewards": spaces.Box(0, 1, (2, self.n)),
                "index": spaces.Discrete(self.n),
            }
        )
