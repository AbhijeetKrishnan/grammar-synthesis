import argparse

import grammar_synthesis
import gymnasium
from grammar_synthesis.policy import RandomSampler, UniformRandomSampler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_episodes', type=int, default=5)
    parser.add_argument('-l', '--max_len', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    env = gymnasium.make(
        'GrammarSynthesisEnv-v0', 
        grammar=open('grammar_synthesis/envs/assets/example.lark').read(), 
        start_symbol='s', 
        reward_fn=lambda program_text, mdp_config: len(program_text),
        max_len=args.max_len)
    
    num_episodes = args.num_episodes
    env.action_space.seed(args.seed)
    max_len = args.max_len

    uniform_agent = UniformRandomSampler(env.parser, max_len)
    random_agent = RandomSampler(env)

    for _ in range(num_episodes):
        obs, info, terminated, truncated = *env.reset(), False, False
        while not terminated and not truncated:
            mask = info['action_mask']
            action = random_agent.get_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    env.close()

    # TODO: fix usage of uniform agent once implementation is finished
    program = uniform_agent.generate_string(5)
    print(program)