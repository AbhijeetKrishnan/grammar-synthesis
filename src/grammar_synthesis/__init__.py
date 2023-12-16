from gymnasium.envs.registration import register

from .envs import (
    GrammarSynthesisEnv as GrammarSynthesisEnv,
)

register(
    id="GrammarSynthesisEnv-v1",
    entry_point="grammar_synthesis.envs.synthesis_env:GrammarSynthesisEnv",
)
