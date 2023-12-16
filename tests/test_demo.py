import random

import gymnasium
import pytest
from grammar_synthesis.envs import GrammarSynthesisEnv
from grammar_synthesis.envs.examples import LOL
from grammar_synthesis.policy import ParsedPlayback, RandomSampler, UniformRandomSampler
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


def test_random(env):
    random_agent = RandomSampler(env.unwrapped)
    num_episodes = 10
    for i in range(num_episodes):
        obs, info = env.reset(seed=1)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            mask = info["action_mask"]
            action = random_agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()


def test_uniform_random(env):
    uniform_agent = UniformRandomSampler(env.unwrapped, 70, seed=1)
    num_episodes = 10
    for i in range(num_episodes):
        n = random.randint(3, 70)
        uniform_agent.generate_actions(n)

        obs, info = env.reset(seed=1)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            mask = info["action_mask"]
            action = uniform_agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()


def test_parsed_playback(env):
    playback_agent = ParsedPlayback(env.unwrapped)
    program_text = """lollol"""
    playback_agent.build_actions(program_text)
    num_episodes = 10
    for i in range(num_episodes):
        obs, info = env.reset(seed=1)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            mask = info["action_mask"]
            action = playback_agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()
