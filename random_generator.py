import gymnasium

import grammar_synthesis

def reward_fn(symbol_list):
    return len(symbol_list)

env = gymnasium.make(
    'GrammarSynthesisEnv-v0', 
    grammar=open('grammar_synthesis/envs/assets/example.lark').read(), 
    start_symbol='s', 
    reward_fn=lambda symbol_list: len(symbol_list),
    max_len=20)

num_episodes = 5
env.action_space.seed(1)

for _ in range(num_episodes):
    obs, info, terminated, truncated = *env.reset(), False, False
    while not terminated and not truncated:
        mask = info['action_mask']
        action = env.action_space.sample(mask=mask)
        obs, terminated, reward, truncated, info = env.step(action)
    env.render()
env.close()