"""PDCoEA on BinaryDefence, with PPO as the policy variation operator.

The algorithm of jaxued/examples/kinetix_pdcoea.py (standard --algorithm pdcoea, --policy_mutation ppo)
with BinaryDefence as the environment. One generation:

    for every slot j = 1..lam (in parallel):
        sample policies phi1, phi2 and levels theta1, theta2 from the populations
        roll out (phi1, theta1), (phi2, theta1), (phi1, theta2), (phi2, theta2) on N parallel envs
        if g(phi1, theta2) <= g(phi1, theta1) <= g(phi2, theta1): keep (phi1, theta1) else keep (phi2, theta2)
        next level population:  Mutate(theta^j)
        next policy population: copy of phi^j
    PPO update of every selected phi^j on its own rollouts (individuals share no actor parameters)
    value update of the global critic on all four rollouts of every slot

A level is an index i in [0, n). As in jaxued/examples/kinetix_pdcoea.py, each offspring level draws a
number of edits k ~ Geometric on {0, 1, ...} with mean --chi (k = 0 leaves it unchanged), run as a masked
scan of --max_edits steps. One edit moves the index to another one drawn uniformly from the other n - 1. The payoff g(phi, theta) is the regret
R*(theta) - R(phi, theta) = 1 - (fraction of the N rollouts that guessed the bit right), since a
level's optimal return is 1. The policy minimises g and the level maximises it.

Every episode is one step, so a rollout is one reset and one step per env, and GAE reduces to
advantage = r - V(s) and target = r. The network is not recurrent (as kinetix_pdcoea's --network
mlp): a fully per-individual actor MLP and a global critic.

The critic does not affect the policies here. Each individual's actor batch is N rollouts of a
single level, so V(s) is the same for all of them, and actor_loss's per-individual advantage
normalisation subtracts it out. Individuals interact only through selection.

The PPO pieces (actor_loss, ppo_update, clip_grads_per_group, gather_population, where_slot) are
those of jaxued/examples/maze_pdcoea.py with the time axis and the LSTM carry removed; jaxued is not
installed in this venv, so they are copied rather than imported.

Array layout: (N, lam, ...) for rollouts, (C, N, lam, ...) over the C = 4 tournament combinations,
(lam,) for the level population.

Run from the BinaryDefence root (the gymnax fork is not installed in .venv, so put it on the path):
    PYTHONPATH=. .venv/Scripts/python.exe experiments/bdefence_pdcoea.py
"""
import argparse
import time

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState as BaseTrainState

from gymnax.environments.bdefence import BinaryDefence

# Parameter subtrees of PopulationActor that have a leading population axis: all of the actor.
# The critic's modules are named so that none of them match.
POPULATION_PARAMS = ("actor_embed", "actor0", "actor1")


class TrainState(BaseTrainState):
    levels: chex.Array = struct.field(pytree_node=True)  # level population, (lam,) int32 indices
    num_generations: int = 0


# region Network
class PopulationActor(nn.Module):
    """Per-individual MLP: embedding ("actor_embed"), hidden layer ("actor0") and action head ("actor1").

    Nothing is shared between individuals. Inputs have shape (B, lam, obs_dim); individual j acts on column j.
    """
    action_dim: int
    width: int = 64

    @nn.compact
    def __call__(self, obs):
        vmap = lambda cls: nn.vmap(cls, in_axes=1, out_axes=1, variable_axes={"params": 0}, split_rngs={"params": True})
        x = nn.tanh(vmap(nn.Dense)(self.width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_embed")(obs))
        x = nn.tanh(vmap(nn.Dense)(self.width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor0")(x))
        logits = vmap(nn.Dense)(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(x)
        return distrax.Categorical(logits=logits)


class GlobalCritic(nn.Module):
    """One value function V(obs) for the whole population. Accepts any batch shape."""
    width: int = 64

    @nn.compact
    def __call__(self, obs):
        x = nn.tanh(nn.Dense(self.width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic0")(obs))
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(x)
        return jnp.squeeze(v, axis=-1)


class ActorCritic(nn.Module):
    """The policy population ("actor") and the global critic ("critic")."""
    action_dim: int

    def setup(self):
        self.actor = PopulationActor(self.action_dim)
        self.critic = GlobalCritic()

    def __call__(self, obs):
        return self.actor(obs), self.critic(obs)

    def policy(self, obs):
        return self.actor(obs)

    def value(self, obs):
        return self.critic(obs)
# endregion


# region Population helpers
def is_population_path(path):
    """Whether a params (or optimiser-moment) leaf belongs to POPULATION_PARAMS, i.e. has a leading population axis."""
    return bool({getattr(p, "key", None) for p in path} & set(POPULATION_PARAMS))


def gather_population(tree, index):
    """`tree` with the per-individual subtrees gathered along the population axis.

    Works on the params and on optimiser states whose moments mirror them (Adam's mu and nu).
    """
    return jax.tree_util.tree_map_with_path(lambda path, x: x[index] if is_population_path(path) else x, tree)


def clip_grads_per_group(grads, max_norm):
    """Clips each individual's actor gradient to max_norm on its own, and the shared embedding and
    the critic as one group each. Returns the clipped grads and the per-individual norms (lam,)."""
    def group(path):
        keys = {getattr(p, "key", None) for p in path}
        if "critic" in keys:
            return "critic"
        return "population" if keys & set(POPULATION_PARAMS) else "shared"

    leaves = jax.tree_util.tree_flatten_with_path(grads)[0]
    sq = {"critic": 0.0, "shared": 0.0, "population": 0.0}
    for path, g in leaves:
        k = group(path)
        sq[k] = sq[k] + (jnp.sum(g.reshape(g.shape[0], -1) ** 2, axis=1) if k == "population" else jnp.sum(g ** 2))
    scale = {k: jnp.minimum(1.0, max_norm / (jnp.sqrt(v) + 1e-6)) for k, v in sq.items()}

    def clip(path, g):
        s = scale[group(path)]
        return g * s.reshape(-1, *([1] * (g.ndim - 1))) if s.ndim else g * s

    return jax.tree_util.tree_map_with_path(clip, grads), jnp.sqrt(sq["population"])


def where_slot(cond, a, b, axis):
    """Per-slot select: cond has shape (lam,) and indexes axis `axis` of a and b."""
    shape = [1] * a.ndim
    shape[axis] = cond.shape[0]
    return jnp.where(cond.reshape(shape), a, b)
# endregion


# region PPO
def actor_loss(params, apply_fn, actor_batch, config):
    """PPO's clipped surrogate minus the entropy bonus, summed over the population.

    actor_batch = (obs, actions, log_probs, advantages), each (B, lam, ...). Returns the loss and the
    per-individual (l_clip, entropy).
    """
    obs, actions, log_probs, advantages = actor_batch
    clip_eps = config["clip_eps"]
    pi = apply_fn(params, obs, method=ActorCritic.policy)
    entropy = pi.entropy().mean(axis=0)  # (lam,)
    ratio = jnp.exp(pi.log_prob(actions) - log_probs)
    A = (advantages - advantages.mean(axis=0)) / (advantages.std(axis=0) + 1e-5)  # per individual
    l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean(axis=0)  # (lam,)
    return l_clip.sum() - config["entropy_coeff"] * entropy.sum(), (l_clip, entropy)


def ppo_update(rng, train_state, actor_batch, critic_batch, config):
    """PPO on the selected individuals, and a value update of the global critic on every rollout.

    actor_batch = (obs, actions, log_probs, advantages), each (N, lam, ...): the selected rollouts.
    critic_batch = (obs, values, targets), each (N, 4 * lam, ...): all four rollouts of every slot.
    Minibatches split the N axis and keep the population axis whole.
    """
    N = actor_batch[1].shape[0]
    n_minibatch, clip_eps = config["num_minibatches"], config["clip_eps"]

    def update_minibatch(train_state, minibatch):
        actor_minibatch, (c_obs, c_values, c_targets) = minibatch

        def loss_fn(params):
            l_actor, (l_clip, entropy) = actor_loss(params, train_state.apply_fn, actor_minibatch, config)
            values_pred = train_state.apply_fn(params, c_obs, method=ActorCritic.value)
            values_pred_clipped = c_values + (values_pred - c_values).clip(-clip_eps, clip_eps)
            l_vf = 0.5 * jnp.maximum((values_pred - c_targets) ** 2, (values_pred_clipped - c_targets) ** 2).mean()
            return l_actor + config["critic_coeff"] * l_vf, (l_vf, l_clip.mean(), entropy.mean())

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        grads, grad_norms = clip_grads_per_group(grads, config["max_grad_norm"])
        return train_state.apply_gradients(grads=grads), (loss, aux, grad_norms.mean())

    def update_epoch(carry, _):
        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, N)
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0).reshape(n_minibatch, -1, *x.shape[1:]),
            (actor_batch, critic_batch),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, config["epoch_ppo"])
# endregion


def main(config):
    N, lam, n = config["num_rollouts_per_pair"], config["pop_size"], config["n"]
    assert N % config["num_minibatches"] == 0, "num_rollouts_per_pair must be divisible by num_minibatches"
    assert config["num_generations"] % config["eval_freq"] == 0, "num_generations must be a multiple of eval_freq"
    assert config["chi"] > 0, "chi must be positive"
    p_truncated = (config["chi"] / (1.0 + config["chi"])) ** (config["max_edits"] + 1)  # P(k > max_edits)
    if p_truncated > 1e-4:
        print(f"warning: P(num_edits > max_edits) = {p_truncated:.2e}; raise --max_edits")
    assert n >= 2, "level mutation draws another index, so n >= 2"

    hidden_bits = jax.random.bernoulli(jax.random.PRNGKey(config["bits_seed"]), 0.5, (n,)).astype(jnp.int32)
    assert int(hidden_bits.sum()) < n, "hidden bitstring is all 1s; pick another --bits_seed"
    env = BinaryDefence(hidden_bits)
    env_params = env.default_params
    # over (N, lam)
    reset_env = jax.vmap(jax.vmap(env.reset_to_level, in_axes=(0, 0, None)), in_axes=(0, 0, None))
    step_env = jax.vmap(jax.vmap(env.step_env, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, None))

    # Rollout combinations (policy index, level index) per slot: (phi1,theta1), (phi2,theta1), (phi1,theta2), (phi2,theta2)
    pol_ids, lvl_ids = np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])
    C = 4

    network = ActorCritic(env.num_actions)

    @jax.jit
    def create_train_state(rng) -> TrainState:
        rng_init, rng_levels = jax.random.split(rng)
        params = network.init(rng_init, jnp.zeros((1, lam, n)))
        num_ppo_steps = config["num_generations"] * config["epoch_ppo"] * config["num_minibatches"]
        lr = optax.linear_schedule(config["lr"], 0.0, num_ppo_steps) if config["anneal_lr"] else config["lr"]
        tx = optax.adam(learning_rate=lr, eps=1e-5)  # ppo_update clips each individual's gradient separately
        levels = jax.random.randint(rng_levels, (lam,), 0, n)
        return TrainState.create(apply_fn=network.apply, params=params, tx=tx, levels=levels, num_generations=0)

    def rollout(rng, params, levels):
        """One episode (= one step) on each of N envs per slot. levels (lam,). Returns (N, lam, ...) arrays."""
        rng_reset, rng_action, rng_step = jax.random.split(rng, 3)
        obs, env_state = reset_env(jax.random.split(rng_reset, (N, lam)), jnp.broadcast_to(levels, (N, lam)), env_params)
        pi, value = network.apply(params, obs)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        _, _, reward, _, _ = step_env(jax.random.split(rng_step, (N, lam)), env_state, action, env_params)
        return obs, action, reward.astype(jnp.float32), log_prob, value

    def mutate_levels(rng, levels):
        """Each level gets k ~ Geometric(1 / (1 + chi)) - 1 edits (support {0, 1, ...}, mean chi), capped at max_edits.

        One edit moves the index to another one drawn uniformly from the other n - 1. The edits are a masked scan
        of max_edits steps, so k can differ between levels. Returns the new levels and the number of edits (lam,).
        """
        rng_num, rng_edit = jax.random.split(rng)
        num_edits = jax.random.geometric(rng_num, 1.0 / (1.0 + config["chi"]), shape=levels.shape) - 1
        num_edits = jnp.minimum(num_edits, config["max_edits"])

        def edit(levels, x):
            rng, i = x
            other = (levels + jax.random.randint(rng, levels.shape, 1, n)) % n
            return jnp.where(i < num_edits, other, levels), None

        steps = (jax.random.split(rng_edit, config["max_edits"]), jnp.arange(config["max_edits"]))
        new_levels, _ = jax.lax.scan(edit, levels, steps)
        return new_levels, num_edits

    def train_step(carry, _):
        """One PDCoEA generation."""
        rng, train_state = carry
        rng, rng_pol, rng_lvl, rng_roll, rng_mut, rng_ppo = jax.random.split(rng, 6)

        # --- Sample one tournament per slot: policies idx[i, j] and levels lidx[k, j]
        idx = jax.random.randint(rng_pol, (2, lam), 0, lam)
        lidx = jax.random.randint(rng_lvl, (2, lam), 0, lam)
        theta = train_state.levels[lidx]  # (2, lam)

        # --- One rollout per combination
        cand_params = jax.vmap(lambda i: gather_population(train_state.params, idx[i]))(pol_ids)
        obs, actions, rewards, log_probs, values = jax.vmap(rollout)(
            jax.random.split(rng_roll, C), cand_params, theta[lvl_ids]
        )  # (C, N, lam, ...)
        # One-step episodes: GAE is r - V(s) and the value target is r
        advantages, targets = rewards - values, rewards

        # --- Payoff: regret g = R* - R_hat = 1 - success rate, (C, lam)
        R_hat = rewards.mean(axis=1)
        F = 1.0 - R_hat
        F11, F21, F12, F22 = F[0], F[1], F[2], F[3]  # F(phi_i, theta_k) as F<i><k>
        # The policy minimises F and the level maximises it.
        win1 = (F12 <= F11) & (F11 <= F21)  # (lam,)
        all_tied = (F12 == F11) & (F11 == F21)

        # --- Selection
        parent = jnp.where(win1, idx[0], idx[1])
        level_parent = jnp.where(win1, lidx[0], lidx[1])
        choose = lambda a: where_slot(win1, a[0], a[3], axis=1)  # combo 0 = (phi1,theta1), combo 3 = (phi2,theta2)
        actor_batch = jax.tree_util.tree_map(choose, (obs, actions, log_probs, advantages))
        merge = lambda a: jnp.moveaxis(a, 0, 1).reshape(N, C * lam, *a.shape[3:])  # (C, N, lam, ...) -> (N, C*lam, ...)
        critic_batch = jax.tree_util.tree_map(merge, (obs, values, targets))
        theta_sel = jnp.where(win1, theta[0], theta[1])

        new_levels, num_edits = mutate_levels(rng_mut, theta_sel)
        train_state = train_state.replace(
            params=gather_population(train_state.params, parent),
            opt_state=gather_population(train_state.opt_state, parent),
            levels=new_levels,
        )

        # --- Variation: PPO from the selected parents
        (rng_ppo, train_state), losses = ppo_update(rng_ppo, train_state, actor_batch, critic_batch, config)
        train_state = train_state.replace(num_generations=train_state.num_generations + 1)

        loss, (l_vf, l_clip, entropy), grad_norm = jax.tree_util.tree_map(lambda x: x.mean(), losses)
        metrics = {
            "pdcoea/frac_phi1_theta1_selected": win1.mean(),
            "pdcoea/frac_slots_tied": all_tied.mean(),
            "pdcoea/num_distinct_parents": jnp.zeros(lam).at[parent].set(1).sum(),
            "pdcoea/num_distinct_level_parents": jnp.zeros(lam).at[level_parent].set(1).sum(),
            "pdcoea/regret_mean": F.mean(),
            "pdcoea/regret_selected_mean": jnp.where(win1, F11, F22).mean(),
            "levels/mean_num_edits": num_edits.mean(),
            "levels/max_num_edits": num_edits.max(),
            "levels/frac_unedited": (num_edits == 0).mean(),
            "agent/loss": loss,
            "agent/critic_loss": l_vf,
            "agent/actor_loss": l_clip,
            "agent/entropy": entropy,
            "agent/grad_norm_per_individual": grad_norm,
        }
        return (rng, train_state), metrics

    # Fixed held-out probe set: K distinct indices drawn once, evaluated every eval_freq generations to track forgetting.
    K = config["num_eval_levels"]
    assert 1 <= K <= n
    eval_levels = jax.random.choice(jax.random.PRNGKey(config["eval_levels_seed"]), n, (K,), replace=False)
    eval_levels_np = np.asarray(eval_levels)
    eval_chunk = min(n, config["eval_chunk"])
    assert n % eval_chunk == 0, "n must be divisible by eval_chunk"

    def evaluate(train_state):
        """Exact scores of every individual over all n indices, and statistics of the level population."""
        def probs_on(indices):  # (chunk,) -> (chunk, lam, 2); chunked so that large n fits in memory
            obs = jnp.broadcast_to(jax.nn.one_hot(indices, n)[:, None, :], (indices.shape[0], lam, n))
            return network.apply(train_state.params, obs, method=ActorCritic.policy).probs
        probs = jax.lax.map(probs_on, jnp.arange(n).reshape(-1, eval_chunk)).reshape(n, lam, 2)
        p_correct = jnp.take_along_axis(probs, hidden_bits[:, None, None].repeat(lam, 1), axis=-1)[..., 0]  # (n, lam)
        greedy_correct = probs.argmax(-1) == hidden_bits[:, None]  # (n, lam)
        expected = p_correct.mean(axis=0)  # (lam,) expected reward on a uniform index
        greedy = greedy_correct.mean(axis=0)  # (lam,) fraction of the bitstring the greedy policy gets right
        # How the level population does against the policy population: success rate of the mean policy on each level
        on_levels = p_correct[train_state.levels].mean(axis=1)  # (lam,)
        # Solve rate on the fixed eval levels: per level, the fraction of the population whose greedy policy solves it
        eval_solved = greedy_correct[eval_levels]  # (K, lam)
        return {
            "_eval_levels_solve_rate": eval_solved.mean(axis=1),  # (K,), turned into forgetting metrics on the host
            "eval_levels/solve_rate": eval_solved.mean(),
            "eval_levels/solve_rate_worst_individual": eval_solved.mean(axis=0).min(),
            "eval_levels/expected_accuracy": p_correct[eval_levels].mean(),
            "eval_levels/frac_individuals_solving_all": eval_solved.all(axis=0).mean(),
            "eval/expected_accuracy_mean": expected.mean(),
            "eval/expected_accuracy_best": expected.max(),
            "eval/expected_accuracy_worst": expected.min(),
            "eval/frac_individuals_perfect": (greedy == 1.0).mean(),
            # Solve rate of an individual: fraction of the n indices its greedy policy solves. Mean over the population.
            "eval/mean_solve_rate": greedy.mean(),
            "eval/solve_rate_best": greedy.max(),
            "eval/solve_rate_std": greedy.std(),
            "eval/worst_index_accuracy": p_correct.mean(axis=1).min(),
            "levels/num_distinct": jnp.zeros(n).at[train_state.levels].set(1).sum(),
            "levels/population_success_on_levels": on_levels.mean(),
        }

    @jax.jit
    def train_and_eval_step(runner_state):
        runner_state, metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])
        metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)  # average over the eval_freq generations (all scalars)
        metrics.update(evaluate(runner_state[1]))
        metrics["num_generations"] = runner_state[1].num_generations
        return runner_state, metrics

    rng = jax.random.PRNGKey(config["seed"])
    rng, rng_init = jax.random.split(rng)
    runner_state = (rng, create_train_state(rng_init))
    print(f"BinaryDefence n={n}, {int(hidden_bits.sum())} ones; pop_size {lam}, {N} rollouts per pair")
    run = wandb.init(
        entity=config["wandb_entity"], project=config["wandb_project"], name=config["wandb_name"],
        mode=config["wandb_mode"],
        config={**config, "shared_encoder": False, "num_ones": int(hidden_bits.sum()), "eval_levels": eval_levels_np.tolist()},
    )

    # Forgetting is tracked at the population level: individuals are replaced by copies of parents every generation,
    # so slot j has no stable identity. A level is "known" when at least half of the population solves it.
    prev = peak = None
    cum_forgotten = cum_learned = 0
    history = []
    for _ in range(config["num_generations"] // config["eval_freq"]):
        start = time.time()
        runner_state, metrics = train_and_eval_step(runner_state)
        metrics = jax.device_get(metrics)
        rate = np.asarray(metrics.pop("_eval_levels_solve_rate"))  # (K,)
        metrics = {k: float(v) for k, v in metrics.items()}
        known = rate >= 0.5
        if prev is None:
            prev, peak = rate, rate.copy()
        prev_known = prev >= 0.5
        forgotten, learned = int((prev_known & ~known).sum()), int((~prev_known & known).sum())
        cum_forgotten, cum_learned = cum_forgotten + forgotten, cum_learned + learned
        peak = np.maximum(peak, rate)
        metrics.update({
            "eval_levels/num_known": int(known.sum()),
            "eval_levels/num_solved_by_all": int((rate == 1.0).sum()),
            "eval_levels/num_forgotten": forgotten,  # known at the previous eval, not now
            "eval_levels/num_learned": learned,
            "eval_levels/cum_forgotten": cum_forgotten,
            "eval_levels/cum_learned": cum_learned,
            "eval_levels/num_known_before_not_now": int(((peak >= 0.5) & ~known).sum()),
            "eval_levels/forgetting_from_peak": float((peak - rate).mean()),  # mean drop below each level's best so far
            "eval_levels/max_forgetting_from_peak": float((peak - rate).max()),
            "eval_levels/solve_rate_change": float((rate - prev).mean()),
        })
        prev = rate
        history.append(metrics)
        dt = time.time() - start
        sps = config["eval_freq"] * C * N * lam / dt
        tracked = config["num_eval_levels_charted"]
        per_level = {f"eval_levels_per_index/idx_{i:07d}": float(r) for i, r in zip(eval_levels_np[:tracked], rate[:tracked])}
        run.log({**metrics, **per_level, "eval_levels/solve_rate_hist": wandb.Histogram(rate),
                 "timing/sps": sps, "timing/env_steps": metrics["num_generations"] * C * N * lam},
                step=int(metrics["num_generations"]))
        print(
            f"gen {int(metrics['num_generations']):6d} | "
            f"solve {metrics['eval/mean_solve_rate']:.3f} | eval lvls {metrics['eval_levels/solve_rate']:.3f} "
            f"known {metrics['eval_levels/num_known']}/{K} forgot {metrics['eval_levels/cum_forgotten']} | acc mean {metrics['eval/expected_accuracy_mean']:.3f} best {metrics['eval/expected_accuracy_best']:.3f} "
            f"perfect {metrics['eval/frac_individuals_perfect']:.2f} | "
            f"regret {metrics['pdcoea/regret_mean']:.3f} (sel {metrics['pdcoea/regret_selected_mean']:.3f}) "
            f"win1 {metrics['pdcoea/frac_phi1_theta1_selected']:.2f} | "
            f"levels distinct {int(metrics['levels/num_distinct'])} | "
            f"entropy {metrics['agent/entropy']:.3f} | "
            f"{sps:.0f} sps",
            flush=True,
        )
    run.finish()
    return runner_state, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    # === Env ===
    parser.add_argument("--n", type=int, default=100, help="Length of the hidden bitstring")
    parser.add_argument("--bits_seed", type=int, default=0, help="Seed of the hidden bitstring")
    # === Population ===
    parser.add_argument("--pop_size", type=int, default=64, help="lam: number of policies and of levels")
    parser.add_argument("--num_generations", type=int, default=3000)
    parser.add_argument("--chi", type=float, default=0.3,
                        help="Mutation rate: expected number of edits per offspring level (geometric on {0, 1, ...})")
    parser.add_argument("--max_edits", type=int, default=20, help="Cap on edits per offspring level")
    # === Eval ===
    parser.add_argument("--eval_freq", type=int, default=100, help="Generations between evaluations")
    parser.add_argument("--num_eval_levels", type=int, default=256, help="K: fixed random indices used to track forgetting")
    parser.add_argument("--num_eval_levels_charted", type=int, default=32,
                        help="How many of the eval levels get their own per-index wandb chart")
    parser.add_argument("--eval_levels_seed", type=int, default=1)
    parser.add_argument("--eval_chunk", type=int, default=128, help="Indices evaluated at once (memory bound for large n)")
    # === Logging ===
    parser.add_argument("--wandb_entity", type=str, default="ectheory-birmingham")
    parser.add_argument("--wandb_project", type=str, default="binary-defence-pdcoea")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    # === PPO ===
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--num_rollouts_per_pair", type=int, default=16,
                        help="N: rollouts per (policy, level) pair. Each slot plays 4 pairs, 4*N*lam env steps per generation")
    parser.add_argument("--num_minibatches", type=int, default=1)
    parser.add_argument("--epoch_ppo", type=int, default=4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coeff", type=float, default=1e-2)
    parser.add_argument("--critic_coeff", type=float, default=0.5)
    main(vars(parser.parse_args()))
