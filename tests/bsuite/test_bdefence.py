import jax
import jax.numpy as jnp
from gymnax.environments.bdefence.defence import BinaryDefence
import gymnax




def test_bdef_reset_initializes_named_fields():
    """Reset should populate rewards, regret, and time in the right slots."""
    env = BinaryDefence(hidden_bits= jnp.array([0,1,1,0,1,1,1]))

    _, state = env.reset_env(jax.random.key(0), env.default_params)

    assert state.rewards.shape == (2, 7)
    assert state.index.shape == (1,) 
    assert jnp.asarray(state.time).shape == ()



def test_simple_bandit_step_updates_regret_and_time():
    """Step should preserve rewards and increment regret/time."""
    env = BinaryDefence(hidden_bits= jnp.array([0,1,1,0,1,1,1]))
    params = env.default_params

    _, state = env.reset_env(jax.random.key(0), params)
    action = 0

    _, next_state, reward, done, _ = env.step_env(
        jax.random.key(1), state, action, params
    )

    assert jnp.array_equal(next_state.rewards, state.rewards)
    assert jnp.array_equal(reward, state.rewards[action])
    assert next_state.time == 1
    assert done
    assert jnp.asarray(next_state.time).shape == ()
    assert jnp.array_equal(next_state.index,state.index)

def test_gymnax_make():
    env, params = gymnax.make("BinaryDefence", hidden_bits = jnp.array([0,1,1,0,1,1,1]))
    _, state = env.reset_env(jax.random.key(0), params)
    action = 0

    _, next_state, reward, done, _ = env.step_env(
        jax.random.key(1), state, action, params
    )

    assert jnp.array_equal(next_state.rewards, state.rewards)
    assert jnp.array_equal(reward, state.rewards[action])
    assert next_state.time == 1
    assert done
    assert jnp.asarray(next_state.time).shape == ()
    assert jnp.array_equal(next_state.index,state.index)

