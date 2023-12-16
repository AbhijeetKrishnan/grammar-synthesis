import argparse
import random

from grammar_synthesis import GrammarSynthesisEnv
from grammar_synthesis.envs.examples import LOL
from grammar_synthesis.policy import ParsedPlayback, RandomSampler, UniformRandomSampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_episodes", type=int, default=5)
    parser.add_argument("-l", "--max_len", type=int, default=70)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = GrammarSynthesisEnv(
        grammar=LOL,
        reward_fn=lambda program_text, mdp_config: len(program_text),
        max_len=args.max_len,
    )

    random.seed(args.seed)
    num_episodes = args.num_episodes

    random_agent = RandomSampler(env)
    print("=" * 5, "Random Agent", "=" * 5)
    for i in range(num_episodes):
        obs, info = env.reset(seed=args.seed + i)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            mask = info["action_mask"]
            action = random_agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    playback_agent = ParsedPlayback(env)
    program_text = """lollol"""
    playback_agent.build_actions(program_text)
    print("=" * 5, f"Playback Agent ({program_text})", "=" * 5)
    for i in range(num_episodes):
        obs, info = env.reset(seed=args.seed + i)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            mask = info["action_mask"]
            action = playback_agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    uniform_agent = UniformRandomSampler(env, args.max_len, seed=args.seed)
    print("=" * 5, "Uniform Random Agent", "=" * 5)
    for i in range(num_episodes):
        n = random.randint(3, args.max_len)
        uniform_agent.generate_actions(n)

        obs, info = env.reset(seed=args.seed + i)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            mask = info["action_mask"]
            action = uniform_agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    env.close()


if __name__ == "__main__":
    main()
