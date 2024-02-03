import random

import gymnasium
import pytest
from grammar_synthesis.envs import GrammarSynthesisEnv
from grammar_synthesis.examples import LOL
from grammar_synthesis.policy import (
    ParsedPlayback,
    UniformLengthSampler,
    UniformRandomSampler,
    WeightedRandomSampler,
)
from gymnasium.utils.env_checker import check_env
from gymnasium.utils.env_match import check_environments_match
from gymnasium.utils.performance import benchmark_init, benchmark_render, benchmark_step


@pytest.fixture(scope="module")
def env():
    env = gymnasium.make(
        "GrammarSynthesisEnv-v1",
        grammar=LOL,
        reward_fn=lambda program_text, mdp_config: len(program_text),
        max_len=70,
        derivation_dir="right",
    )
    yield env
    env.close()


def test_env(env):
    assert env is not None


def test_check_env(env):
    check_env(env.unwrapped)


def test_benchmarks(env):
    benchmark_init(lambda: env)
    benchmark_step(env)
    benchmark_render(env)


def test_env_match(env):
    env2 = GrammarSynthesisEnv(
        grammar=LOL,
        reward_fn=lambda program_text, mdp_config: len(program_text),
        max_len=70,
    )
    check_environments_match(env, env2, 100)


def test_uniform_random(env):
    agent = UniformRandomSampler(env.unwrapped)
    num_episodes = 10
    for i in range(num_episodes):
        obs, info = env.reset(seed=1)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            mask = info["action_mask"]
            action = agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()


def test_weighted_random(env):
    weights = [0.1, 0.2, 0.3, 0.4]
    agent = WeightedRandomSampler(env.unwrapped, weights)
    num_episodes = 10
    for i in range(num_episodes):
        obs, info = env.reset(seed=1)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            mask = info["action_mask"]
            action = agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)


def test_uniform_length(env):
    agent = UniformLengthSampler(env.unwrapped, 70, seed=1)
    num_episodes = 10
    for i in range(num_episodes):
        n = random.randint(3, 70)
        agent.generate_actions(n)

        obs, info = env.reset(seed=1)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            mask = info["action_mask"]
            action = agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()


def test_parsed_playback(env):
    agent = ParsedPlayback(env.unwrapped)
    program_text = """lollol"""
    agent.build_actions(program_text)
    num_episodes = 10
    for i in range(num_episodes):
        obs, info = env.reset(seed=1)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            mask = info["action_mask"]
            action = agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()
