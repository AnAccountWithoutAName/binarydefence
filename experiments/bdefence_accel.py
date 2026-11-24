"""ACCEL on BinaryDefence: one PPO agent trained on a prioritised level buffer that grows by mutation.

The algorithm of jaxued/examples/maze_plr.py with --use_accel (robust PLR, no exploratory gradient
updates), with BinaryDefence as the environment and the same network, eval levels and eval metrics
as experiments/bdefence_pdcoea.py. Every update takes one of three branches:

    DR:      play num_train_envs fresh levels (uniform indices), score them, insert into the buffer. No gradient step.
    replay:  sample num_train_envs levels from the buffer (rank prioritisation + staleness), PPO update on them,
             re-score them.
    mutate:  (always right after a replay step) mutate the replayed levels, score the children, insert them.
             No gradient step.

The branch after a DR or mutate step is replay with probability replay_prob, once the buffer is at
least minimum_fill_ratio full, and DR otherwise.

A level is an index i in [0, n). Mutation is the one of bdefence_pdcoea.py: k ~ Geometric on {0, 1, ...}
with mean --chi edits, each moving the index to another one drawn uniformly. The buffer score is the
positive value loss (PVL), mean(max(A, 0)) over the level's rollouts (with one-step episodes A = r - V(s)), or with
--score_function regret the true regret 1 - success rate that bdefence_pdcoea.py selects on.

Matched to bdefence_pdcoea.py: the actor is one individual of its PopulationActor (one-hot index ->
Dense 64 tanh -> Dense 64 tanh -> Dense 2), the critic its GlobalCritic (Dense 64 tanh -> Dense 1), the
level buffer holds as many levels as PDCoEA's level population (64), and an update costs the same number
of env steps as a PDCoEA generation (num_train_envs * num_rollouts_per_level = 256 * 16 = 4 * 16 * 64).

With one agent the solve rate of an eval level is 0 or 1 (the greedy policy is right or not), so
eval/mean_solve_rate is the agent's fraction of the bitstring solved, the counterpart of one PDCoEA
individual's, and forgetting is "was solved, is not now". eval_levels/forgetting_from_peak_soft tracks
the same on the probability of the right bit, which moves before the greedy answer flips.

The LevelSampler is jaxued/src/jaxued/level_sampler.py, copied (jaxued is not installed in this venv).

Run from the BinaryDefence root:
    PYTHONPATH=. .venv/Scripts/python.exe experiments/bdefence_accel.py --n 2048
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

DR, REPLAY = 0, 1  # update_state: the branch last taken (mutate resets it to DR)


# region Level sampler (jaxued/src/jaxued/level_sampler.py)
class LevelSampler:
    """PLR level buffer: rank prioritisation with temperature, mixed with staleness. See jaxued for the docs."""

    def __init__(self, capacity, replay_prob=0.95, staleness_coeff=0.5, minimum_fill_ratio=1.0,
                 prioritization="rank", prioritization_params=None, duplicate_check=False):
        self.capacity = capacity
        self.replay_prob = replay_prob
        self.staleness_coeff = staleness_coeff
        self.minimum_fill_ratio = minimum_fill_ratio
        self.prioritization = prioritization
        self.prioritization_params = prioritization_params or ({"temperature": 1.0} if prioritization == "rank" else {"k": 1})
        self.duplicate_check = duplicate_check

    def initialize(self, pholder_level, pholder_level_extra=None):
        sampler = {
            "levels": jax.tree_util.tree_map(lambda x: jnp.array([x]).repeat(self.capacity, axis=0), pholder_level),
            "scores": jnp.full(self.capacity, -jnp.inf, dtype=jnp.float32),
            "timestamps": jnp.zeros(self.capacity, dtype=jnp.int32),
            "size": 0,
            "episode_count": 0,
        }
        if pholder_level_extra is not None:
            sampler["levels_extra"] = jax.tree_util.tree_map(lambda x: jnp.array([x]).repeat(self.capacity, axis=0), pholder_level_extra)
        return sampler

    def sample_replay_decision(self, sampler, rng):
        return (self._proportion_filled(sampler) >= self.minimum_fill_ratio) & (jax.random.uniform(rng) < self.replay_prob)

    def sample_replay_level(self, sampler, rng):
        idx = jax.random.choice(rng, self.capacity, p=self.level_weights(sampler))
        new_episode_count = sampler["episode_count"] + 1
        sampler = {**sampler, "timestamps": sampler["timestamps"].at[idx].set(new_episode_count), "episode_count": new_episode_count}
        return sampler, (idx, jax.tree_util.tree_map(lambda x: x[idx], sampler["levels"]))

    def sample_replay_levels(self, sampler, rng, num):
        return jax.lax.scan(self.sample_replay_level, sampler, jax.random.split(rng, num), length=num)

    def insert(self, sampler, level, score, level_extra=None):
        if self.duplicate_check:
            idx = self.find(sampler, level)
            return jax.lax.cond(
                idx == -1,
                lambda: self._insert_new(sampler, level, score, level_extra),
                lambda: ({
                    **self.update(sampler, idx, score, level_extra),
                    "timestamps": sampler["timestamps"].at[idx].set(sampler["episode_count"] + 1),
                    "episode_count": sampler["episode_count"] + 1,
                }, idx),
            )
        return self._insert_new(sampler, level, score, level_extra)

    def insert_batch(self, sampler, levels, scores, level_extras=None):
        def _insert(sampler, step):
            level, score, level_extra = step
            return self.insert(sampler, level, score, level_extra)
        return jax.lax.scan(_insert, sampler, (levels, scores, level_extras))

    def find(self, sampler, level):
        eq_tree = jax.tree_util.tree_map(lambda X, y: (X == y).reshape(self.capacity, -1).all(axis=-1), sampler["levels"], level)
        eq_mask = jnp.array(jax.tree_util.tree_flatten(eq_tree)[0]).all(axis=0) & (jnp.arange(self.capacity) < sampler["size"])
        return jax.lax.select(eq_mask.any(), eq_mask.argmax(), -1)

    def update(self, sampler, idx, score, level_extra=None):
        new_sampler = {**sampler, "scores": sampler["scores"].at[idx].set(score)}
        if level_extra is not None:
            new_sampler["levels_extra"] = jax.tree_util.tree_map(lambda x, y: x.at[idx].set(y), new_sampler["levels_extra"], level_extra)
        return new_sampler

    def update_batch(self, sampler, level_inds, scores, level_extras=None):
        def _update(sampler, step):
            level_idx, score, level_extra = step
            return self.update(sampler, level_idx, score, level_extra), None
        return jax.lax.scan(_update, sampler, (level_inds, scores, level_extras))[0]

    def level_weights(self, sampler):
        return (1 - self.staleness_coeff) * self.score_weights(sampler) + self.staleness_coeff * self.staleness_weights(sampler)

    def score_weights(self, sampler):
        mask = jnp.arange(self.capacity) < sampler["size"]
        ord = (-jnp.where(mask, sampler["scores"], -jnp.inf)).argsort()
        if self.prioritization == "rank":
            ranks = jnp.empty_like(ord).at[ord].set(jnp.arange(len(ord)) + 1)
            w_s = jnp.where(mask, 1 / ranks, 0) ** (1 / self.prioritization_params["temperature"])
            return w_s / w_s.sum()
        topk_mask = jnp.empty_like(ord).at[ord].set(jnp.arange(self.capacity) < jnp.minimum(sampler["size"], self.prioritization_params["k"]))
        return jax.nn.softmax(sampler["scores"], where=topk_mask, initial=0)

    def staleness_weights(self, sampler):
        mask = jnp.arange(self.capacity) < sampler["size"]
        w_c = jnp.where(mask, sampler["episode_count"] - sampler["timestamps"], 0)
        return jax.lax.select(w_c.sum() > 0, w_c / w_c.sum(), mask / sampler["size"])

    def _insert_new(self, sampler, level, score, level_extra):
        idx = self._get_next_idx(sampler)
        replace_cond = sampler["scores"][idx] < score

        def _replace():
            new_sampler = {
                **sampler,
                "levels": jax.tree_util.tree_map(lambda x, y: x.at[idx].set(y), sampler["levels"], level),
                "scores": sampler["scores"].at[idx].set(score),
                "timestamps": sampler["timestamps"].at[idx].set(sampler["episode_count"] + 1),
                "size": jnp.minimum(sampler["size"] + 1, self.capacity),
            }
            if level_extra is not None:
                new_sampler["levels_extra"] = jax.tree_util.tree_map(lambda x, y: x.at[idx].set(y), new_sampler["levels_extra"], level_extra)
            return new_sampler

        new_sampler = jax.lax.cond(replace_cond, _replace, lambda: sampler)
        new_sampler["episode_count"] += 1
        return new_sampler, jax.lax.select(replace_cond, idx, -1)

    def _proportion_filled(self, sampler):
        return sampler["size"] / self.capacity

    def _get_next_idx(self, sampler):
        return jax.lax.select(sampler["size"] < self.capacity, sampler["size"], self.level_weights(sampler).argmin())
# endregion


class TrainState(BaseTrainState):
    sampler: dict = struct.field(pytree_node=True)
    update_state: int = 0
    num_dr_updates: int = 0
    num_replay_updates: int = 0
    num_mutation_updates: int = 0
    replay_last_level_batch: chex.Array = struct.field(pytree_node=True, default=None)  # (num_train_envs,) int32


# region Network: one individual of bdefence_pdcoea.py's PopulationActor, and its GlobalCritic
class ActorCritic(nn.Module):
    action_dim: int
    width: int = 64

    @nn.compact
    def __call__(self, obs):
        dense = lambda w, scale, name: nn.Dense(w, kernel_init=orthogonal(scale), bias_init=constant(0.0), name=name)
        x = nn.tanh(dense(self.width, np.sqrt(2), "actor_embed")(obs))
        x = nn.tanh(dense(self.width, np.sqrt(2), "actor0")(x))
        logits = dense(self.action_dim, 0.01, "actor1")(x)
        v = dense(1, 1.0, "critic1")(nn.tanh(dense(self.width, np.sqrt(2), "critic0")(obs)))
        return distrax.Categorical(logits=logits), jnp.squeeze(v, axis=-1)
# endregion


def main(config):
    n, E, S = config["n"], config["num_train_envs"], config["num_rollouts_per_level"]
    assert (E * S) % config["num_minibatches"] == 0
    assert config["num_updates"] % config["eval_freq"] == 0, "num_updates must be a multiple of eval_freq"
    assert config["chi"] > 0 and n >= 2

    hidden_bits = jax.random.bernoulli(jax.random.PRNGKey(config["bits_seed"]), 0.5, (n,)).astype(jnp.int32)
    assert int(hidden_bits.sum()) < n, "hidden bitstring is all 1s; pick another --bits_seed"
    env = BinaryDefence(hidden_bits)
    env_params = env.default_params
    # over (S, E): S rollouts of each of the E levels
    reset_env = jax.vmap(jax.vmap(env.reset_to_level, in_axes=(0, 0, None)), in_axes=(0, 0, None))
    step_env = jax.vmap(jax.vmap(env.step_env, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, None))

    network = ActorCritic(env.num_actions)
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config["topk_k"]},
        duplicate_check=config["buffer_duplicate_check"],
    )

    @jax.jit
    def create_train_state(rng) -> TrainState:
        # Only replay updates step the optimiser. With --num_replay_updates the schedule spans exactly those steps,
        # so the learning rate reaches 0 at the end as in bdefence_pdcoea.py; otherwise it spans all updates (jaxued).
        target = config["num_replay_updates"] or config["num_updates"]
        num_steps = target * config["epoch_ppo"] * config["num_minibatches"]
        lr = optax.linear_schedule(config["lr"], 0.0, num_steps) if config["anneal_lr"] else config["lr"]
        tx = optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]), optax.adam(learning_rate=lr, eps=1e-5))
        return TrainState.create(
            apply_fn=network.apply,
            params=network.init(rng, jnp.zeros((1, n))),
            tx=tx,
            sampler=level_sampler.initialize(jnp.array(0, dtype=jnp.int32)),
            update_state=jnp.int32(DR),
            replay_last_level_batch=jnp.zeros(E, dtype=jnp.int32),
        )

    # region Rollouts, scores and PPO
    def rollout(rng, params, levels):
        """S one-step episodes on each of the E levels. Returns (S, E, ...) arrays."""
        rng_reset, rng_action, rng_step = jax.random.split(rng, 3)
        obs, env_state = reset_env(jax.random.split(rng_reset, (S, E)), jnp.broadcast_to(levels, (S, E)), env_params)
        pi, value = network.apply(params, obs)
        action = pi.sample(seed=rng_action)
        _, _, reward, _, _ = step_env(jax.random.split(rng_step, (S, E)), env_state, action, env_params)
        reward = reward.astype(jnp.float32)
        # One-step episodes: GAE is r - V(s) and the value target is r
        return (obs, action, pi.log_prob(action), value, reward - value, reward), reward

    def compute_score(advantages, rewards, values):
        """Per level (E,). PVL: mean positive advantage. MaxMC: best return seen minus the mean value.
        regret: the true regret R* - R_hat = 1 - success rate, the payoff bdefence_pdcoea.py selects on (an oracle score)."""
        if config["score_function"] == "pvl":
            return jnp.maximum(advantages, 0.0).mean(axis=0)
        if config["score_function"] == "regret":
            return 1.0 - rewards.mean(axis=0)
        return rewards.max(axis=0) - values.mean(axis=0)

    def ppo_update(rng, train_state, batch, update_grad):
        """PPO on the (S, E) batch, flattened. With update_grad False, the losses are computed and the params kept."""
        batch = jax.tree_util.tree_map(lambda x: x.reshape(S * E, *x.shape[2:]), batch)
        n_mb = config["num_minibatches"]

        def update_minibatch(train_state, minibatch):
            obs, actions, log_probs, values, advantages, targets = minibatch

            def loss_fn(params):
                pi, values_pred = train_state.apply_fn(params, obs)
                ratio = jnp.exp(pi.log_prob(actions) - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                eps = config["clip_eps"]
                l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - eps, 1 + eps) * A)).mean()
                values_pred_clipped = values + (values_pred - values).clip(-eps, eps)
                l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()
                entropy = pi.entropy().mean()
                loss = l_clip + config["critic_coeff"] * l_vf - config["entropy_coeff"] * entropy
                return loss, (l_vf, l_clip, entropy)

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, (loss, aux, optax.global_norm(grads))

        def update_epoch(carry, _):
            rng, train_state = carry
            rng, rng_perm = jax.random.split(rng)
            perm = jax.random.permutation(rng_perm, S * E)
            minibatches = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0).reshape(n_mb, -1, *x.shape[1:]), batch)
            train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
            return (rng, train_state), losses

        (rng, train_state), losses = jax.lax.scan(update_epoch, (rng, train_state), None, config["epoch_ppo"])
        loss, (l_vf, l_clip, entropy), grad_norm = jax.tree_util.tree_map(lambda x: x.mean(), losses)
        return train_state, {"agent/loss": loss, "agent/critic_loss": l_vf, "agent/actor_loss": l_clip,
                             "agent/entropy": entropy, "agent/grad_norm": grad_norm}

    def mutate_levels(rng, levels):
        """As in bdefence_pdcoea.py: k ~ Geometric(1 / (1 + chi)) - 1 edits per level, each to another uniform index."""
        rng_num, rng_edit = jax.random.split(rng)
        num_edits = jnp.minimum(jax.random.geometric(rng_num, 1.0 / (1.0 + config["chi"]), shape=levels.shape) - 1,
                                config["max_edits"])

        def edit(levels, x):
            rng, i = x
            other = (levels + jax.random.randint(rng, levels.shape, 1, n)) % n
            return jnp.where(i < num_edits, other, levels), None

        steps = (jax.random.split(rng_edit, config["max_edits"]), jnp.arange(config["max_edits"]))
        return jax.lax.scan(edit, levels, steps)[0], num_edits
    # endregion

    def train_step(carry, _):
        """One ACCEL update: DR, replay or mutate."""
        rng, train_state = carry
        rng, rng_branch, rng_levels, rng_roll, rng_ppo = jax.random.split(rng, 5)

        def branch_metrics(levels, rewards, scores, losses, num_edits, branch):
            num_edits = num_edits.astype(jnp.float32)
            return {
                **losses,
                "levels/mean_score": scores.mean(),
                "levels/batch_success_rate": rewards.mean(),
                "levels/batch_num_distinct": jnp.zeros(n).at[levels].set(1).sum(),
                "levels/mean_num_edits": num_edits.mean(),
                "levels/frac_unedited": (num_edits == 0).mean(),
                "accel/frac_dr": jnp.float32(branch == 0),
                "accel/frac_replay": jnp.float32(branch == 1),
                "accel/frac_mutate": jnp.float32(branch == 2),
            }

        def on_new_levels(train_state):
            levels = jax.random.randint(rng_levels, (E,), 0, n)
            batch, rewards = rollout(rng_roll, train_state.params, levels)
            scores = compute_score(batch[4], rewards, batch[3])
            sampler, _ = level_sampler.insert_batch(train_state.sampler, levels, scores)
            train_state, losses = ppo_update(rng_ppo, train_state, batch, update_grad=config["exploratory_grad_updates"])
            train_state = train_state.replace(sampler=sampler, update_state=jnp.int32(DR), num_dr_updates=train_state.num_dr_updates + 1)
            return train_state, branch_metrics(levels, rewards, scores, losses, jnp.zeros(E), 0)

        def on_replay_levels(train_state):
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(train_state.sampler, rng_levels, E)
            batch, rewards = rollout(rng_roll, train_state.params, levels)
            scores = compute_score(batch[4], rewards, batch[3])
            sampler = level_sampler.update_batch(sampler, level_inds, scores)
            train_state, losses = ppo_update(rng_ppo, train_state, batch, update_grad=True)
            train_state = train_state.replace(sampler=sampler, update_state=jnp.int32(REPLAY), replay_last_level_batch=levels,
                                              num_replay_updates=train_state.num_replay_updates + 1)
            return train_state, branch_metrics(levels, rewards, scores, losses, jnp.zeros(E), 1)

        def on_mutate_levels(train_state):
            rng_mut, rng_l = jax.random.split(rng_levels)
            levels, num_edits = mutate_levels(rng_mut, train_state.replay_last_level_batch)
            batch, rewards = rollout(rng_roll, train_state.params, levels)
            scores = compute_score(batch[4], rewards, batch[3])
            sampler, _ = level_sampler.insert_batch(train_state.sampler, levels, scores)
            train_state, losses = ppo_update(rng_ppo, train_state, batch, update_grad=config["exploratory_grad_updates"])
            train_state = train_state.replace(sampler=sampler, update_state=jnp.int32(DR),
                                              num_mutation_updates=train_state.num_mutation_updates + 1)
            return train_state, branch_metrics(levels, rewards, scores, losses, num_edits, 2)

        # As in maze_plr.py: after a replay step always mutate; otherwise replay with prob. replay_prob (once filled), else DR
        s = train_state.update_state
        replay = level_sampler.sample_replay_decision(train_state.sampler, rng_branch).astype(jnp.int32)
        branch = (1 - s) * replay + 2 * s
        train_state, metrics = jax.lax.switch(branch, [on_new_levels, on_replay_levels, on_mutate_levels], train_state)
        return (rng, train_state), metrics

    # region Eval: same probe set and metrics as bdefence_pdcoea.py, for a population of one
    K = config["num_eval_levels"]
    assert 1 <= K <= n
    eval_levels = jax.random.choice(jax.random.PRNGKey(config["eval_levels_seed"]), n, (K,), replace=False)
    eval_levels_np = np.asarray(eval_levels)
    eval_chunk = min(n, config["eval_chunk"])
    assert n % eval_chunk == 0, "n must be divisible by eval_chunk"
    cap = config["level_buffer_capacity"]

    def evaluate(train_state):
        def probs_on(indices):
            return network.apply(train_state.params, jax.nn.one_hot(indices, n))[0].probs
        probs = jax.lax.map(probs_on, jnp.arange(n).reshape(-1, eval_chunk)).reshape(n, 2)
        p_correct = jnp.take_along_axis(probs, hidden_bits[:, None], axis=-1)[:, 0]  # (n,)
        greedy_correct = probs.argmax(-1) == hidden_bits  # (n,)
        sampler = train_state.sampler
        in_buffer = jnp.arange(cap) < sampler["size"]
        buf = sampler["levels"]
        return {
            "_eval_levels_solved": greedy_correct[eval_levels],  # (K,) 0/1
            "_eval_levels_p_correct": p_correct[eval_levels],  # (K,)
            "eval/mean_solve_rate": greedy_correct.mean(),  # fraction of the bitstring the greedy policy solves
            "eval/expected_accuracy_mean": p_correct.mean(),
            "eval/frac_individuals_perfect": greedy_correct.all().astype(jnp.float32),
            "eval/worst_index_accuracy": p_correct.min(),
            "eval_levels/solve_rate": greedy_correct[eval_levels].mean(),
            "eval_levels/expected_accuracy": p_correct[eval_levels].mean(),
            "level_sampler/size": sampler["size"],
            "level_sampler/num_distinct": jnp.zeros(n).at[jnp.where(in_buffer, buf, n)].set(1, mode="drop").sum(),
            "level_sampler/mean_score": jnp.where(in_buffer, sampler["scores"], 0.0).sum() / jnp.maximum(sampler["size"], 1),
            "level_sampler/weighted_score": (level_sampler.level_weights(sampler) * jnp.where(in_buffer, sampler["scores"], 0.0)).sum(),
            # Counterpart of PDCoEA's levels/population_success_on_levels: the agent's success on the buffer's levels
            "level_sampler/success_on_buffer": jnp.where(in_buffer, p_correct[buf], 0.0).sum() / jnp.maximum(sampler["size"], 1),
        }

    @jax.jit
    def train_and_eval_step(runner_state):
        runner_state, metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])
        metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)  # averaged over the eval_freq updates
        ts = runner_state[1]
        metrics.update(evaluate(ts))
        metrics.update({"num_updates": ts.num_dr_updates + ts.num_replay_updates + ts.num_mutation_updates,
                        "accel/num_replay_updates": ts.num_replay_updates})
        return runner_state, metrics
    # endregion

    rng = jax.random.PRNGKey(config["seed"])
    rng, rng_init = jax.random.split(rng)
    runner_state = (rng, create_train_state(rng_init))
    print(f"BinaryDefence n={n}, {int(hidden_bits.sum())} ones; ACCEL, buffer {cap}, {E} levels x {S} rollouts per update")
    run = wandb.init(
        entity=config["wandb_entity"], project=config["wandb_project"], name=config["wandb_name"], mode=config["wandb_mode"],
        tags=["ACCEL"],
        config={**config, "algorithm": "accel", "pop_size": 1, "shared_encoder": False,
                "num_ones": int(hidden_bits.sum()), "eval_levels": eval_levels_np.tolist()},
    )

    # One agent: an eval level is "known" when it is solved, so forgetting is "was solved, is not now".
    prev = peak = peak_soft = None
    cum_forgotten = cum_learned = 0
    history = []
    # With --num_replay_updates, train until that many replay (gradient) updates are done, checked every eval_freq
    # updates; otherwise for num_updates updates.
    def done(metrics, i):
        if config["num_replay_updates"]:
            return metrics is not None and metrics["accel/num_replay_updates"] >= config["num_replay_updates"]
        return i >= config["num_updates"] // config["eval_freq"]

    i, metrics = 0, None
    while not done(metrics, i):
        i += 1
        start = time.time()
        runner_state, metrics = train_and_eval_step(runner_state)
        metrics = jax.device_get(metrics)
        solved = np.asarray(metrics.pop("_eval_levels_solved")).astype(np.float64)  # (K,)
        p = np.asarray(metrics.pop("_eval_levels_p_correct")).astype(np.float64)
        metrics = {k: float(v) for k, v in metrics.items()}
        known = solved >= 0.5
        if prev is None:
            prev, peak, peak_soft = solved, solved.copy(), p.copy()
        prev_known = prev >= 0.5
        forgotten, learned = int((prev_known & ~known).sum()), int((~prev_known & known).sum())
        cum_forgotten, cum_learned = cum_forgotten + forgotten, cum_learned + learned
        peak, peak_soft = np.maximum(peak, solved), np.maximum(peak_soft, p)
        metrics.update({
            "eval_levels/num_known": int(known.sum()),
            "eval_levels/num_forgotten": forgotten,
            "eval_levels/num_learned": learned,
            "eval_levels/cum_forgotten": cum_forgotten,
            "eval_levels/cum_learned": cum_learned,
            "eval_levels/num_known_before_not_now": int(((peak >= 0.5) & ~known).sum()),
            "eval_levels/forgetting_from_peak": float((peak - solved).mean()),
            "eval_levels/forgetting_from_peak_soft": float((peak_soft - p).mean()),
            "eval_levels/max_forgetting_from_peak_soft": float((peak_soft - p).max()),
            "eval_levels/solve_rate_change": float((solved - prev).mean()),
        })
        prev = solved
        history.append(metrics)
        dt = time.time() - start
        sps = config["eval_freq"] * E * S / dt
        tracked = config["num_eval_levels_charted"]
        per_level = {f"eval_levels_per_index/idx_{i:07d}": float(r) for i, r in zip(eval_levels_np[:tracked], p[:tracked])}
        step = int(metrics["num_updates"])
        run.log({**metrics, **per_level, "eval_levels/p_correct_hist": wandb.Histogram(p),
                 "timing/sps": sps, "timing/env_steps": step * E * S}, step=step)
        print(
            f"upd {step:6d} | solve {metrics['eval/mean_solve_rate']:.3f} | eval lvls {metrics['eval_levels/solve_rate']:.3f} "
            f"known {metrics['eval_levels/num_known']}/{K} forgot {cum_forgotten} | acc {metrics['eval/expected_accuracy_mean']:.3f} | "
            f"buffer {int(metrics['level_sampler/size'])} distinct {int(metrics['level_sampler/num_distinct'])} "
            f"success {metrics['level_sampler/success_on_buffer']:.3f} | replay {metrics['accel/frac_replay']:.2f} | "
            f"entropy {metrics['agent/entropy']:.3f} | {sps:.0f} sps",
            flush=True,
        )
    run.finish()
    return runner_state, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    # === Env ===
    parser.add_argument("--n", type=int, default=2048, help="Length of the hidden bitstring")
    parser.add_argument("--bits_seed", type=int, default=0, help="Seed of the hidden bitstring")
    # === ACCEL ===
    parser.add_argument("--num_updates", type=int, default=12000, help="Updates; each costs as many env steps as a PDCoEA generation")
    parser.add_argument("--num_replay_updates", type=int, default=None,
                        help="If set, train until this many replay (gradient) updates, ignoring --num_updates, and anneal "
                             "the learning rate over them. 12000 gives as many PPO updates as one PDCoEA lineage")
    parser.add_argument("--num_train_envs", type=int, default=256, help="E: levels per update")
    parser.add_argument("--num_rollouts_per_level", type=int, default=16, help="S: one-step episodes per level per update")
    parser.add_argument("--score_function", type=str, default="pvl", choices=["pvl", "MaxMC", "regret"],
                        help="regret: 1 - success rate, the true regret PDCoEA uses. pvl and MaxMC are 0 on levels a "
                             "deterministic policy always fails, so they cannot find those levels in this game")
    parser.add_argument("--exploratory_grad_updates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--level_buffer_capacity", type=int, default=64, help="Matches PDCoEA's level population")
    parser.add_argument("--replay_prob", type=float, default=0.8)
    parser.add_argument("--staleness_coeff", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--topk_k", type=int, default=4)
    parser.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    parser.add_argument("--prioritization", type=str, default="rank", choices=["rank", "topk"])
    parser.add_argument("--buffer_duplicate_check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chi", type=float, default=0.3, help="Mean number of edits per mutated level (geometric on {0, 1, ...})")
    parser.add_argument("--max_edits", type=int, default=20)
    # === Eval ===
    parser.add_argument("--eval_freq", type=int, default=100, help="Updates between evaluations")
    parser.add_argument("--num_eval_levels", type=int, default=256)
    parser.add_argument("--num_eval_levels_charted", type=int, default=32)
    parser.add_argument("--eval_levels_seed", type=int, default=1)
    parser.add_argument("--eval_chunk", type=int, default=128)
    # === PPO ===
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--num_minibatches", type=int, default=1)
    parser.add_argument("--epoch_ppo", type=int, default=4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coeff", type=float, default=1e-2)
    parser.add_argument("--critic_coeff", type=float, default=0.5)
    # === Logging ===
    parser.add_argument("--wandb_entity", type=str, default="ectheory-birmingham")
    parser.add_argument("--wandb_project", type=str, default="binary-defence")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    main(vars(parser.parse_args()))
