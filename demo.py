import argparse
import random

import grammar_synthesis
import gymnasium
from grammar_synthesis.policy import RandomSampler, UniformRandomSampler, ParsedPlayback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_episodes', type=int, default=5)
    parser.add_argument('-l', '--max_len', type=int, default=70)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    with open('grammar_synthesis/envs/assets/example.pg') as grammar_file:
        env = gymnasium.make(
            'GrammarSynthesisEnv-v0',
            grammar=grammar_file.read(),
            reward_fn=lambda program_text, mdp_config: len(program_text),
            max_len=args.max_len)

    num_episodes = args.num_episodes
    env.action_space.seed(args.seed)

    random_agent = RandomSampler(env)
    print('=' * 5, 'Random Agent', '=' * 5)
    for _ in range(num_episodes):
        obs, info, terminated, truncated = *env.reset(), False, False
        while not terminated and not truncated:
            mask = info['action_mask']
            action = random_agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    playback_agent = ParsedPlayback(env)
    program_text = """lollol"""
    playback_agent.build_actions(program_text)
    print('=' * 5, f'Playback Agent ({program_text})', '=' * 5)
    for _ in range(num_episodes):
        obs, info, terminated, truncated = *env.reset(), False, False
        while not terminated and not truncated:
            mask = info['action_mask']
            action = playback_agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    uniform_agent = UniformRandomSampler(env, args.max_len)
    print('=' * 5, 'Uniform Random Agent', '=' * 5)
    for _ in range(num_episodes):
        n = random.randint(3, args.max_len)
        uniform_agent.generate_actions(n)

        obs, info, terminated, truncated = *env.reset(), False, False
        while not terminated and not truncated:
            mask = info['action_mask']
            action = uniform_agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    env.close()


if __name__ == '__main__':
    main()
