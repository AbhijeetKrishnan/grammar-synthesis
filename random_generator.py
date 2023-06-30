import gymnasium

import grammar_synthesis

if __name__ == '__main__':

    with open('grammar_synthesis/envs/assets/microrts-dsl.lark') as dsl_file: 
        env = gymnasium.make('GrammarSynthesisEnv-v0', grammar=dsl_file.read(), start='program', reward_fn = lambda symbols: len(symbols))

    num_runs = 100
    for _ in range(num_runs):
        obs, info = env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            # env.render()
            mask = info["action_mask"]
            action = env.action_space.sample(mask=mask)
            obs, terminated, reward, truncated, info = env.step(action)
        env.render()

    env.close()