from gymnasium.envs.registration import register

register(
    id="GrammarSynthesisEnv-v0",
    entry_point="grammar_synthesis.envs.synthesis_env:GrammarSynthesisEnv",
    max_episode_steps=200
)