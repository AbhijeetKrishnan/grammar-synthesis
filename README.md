# grammar-synthesis

A [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)-based custom environment that uses [`parglare`](http://www.igordejanovic.net/parglare) to represent tasks for program synthesis using a context-free grammar (CFG).

Program synthesis is the task of writing a program in a (programming) language that fits some specification.

A CFG $G = (V, \Sigma, P, S)$ is used to define the space of all programs. A sample grammar is shown below

$$
\begin{align*}
    S &\rightarrow S\ S\ |\ l\ A\ l \\
    A &\rightarrow o\ A\ |\ o \\
\end{align*}
$$

The grammar synthesis task is to find a string $s$ belonging to the language $L(G)$ of $G$ that optimizes a function, generally the reward function of a secondary MDP (Markov Decision Process).

## Installation

`grammar_synthesis` is installed manually as a Python package using `setuptools`.

```bash
git clone https://github.com/AbhijeetKrishnan/grammar-synthesis.git
cd grammar-synthesis
python3 setup.py install
```

## Usage

The grammar must be specified using [`parglare`](https://github.com/lark-parser/lark)'s flavor of the BNF syntax.

```ebnf
s: s s | "l" a "l";

a: "o" a | "o";
```

This sample grammar and others are provided under `grammar_synthesis/envs/assets/`.

The below snippet implements a random policy for generating a string from a grammar. The reward function used is simply
the length of the synthesized program.

```python
import gymnasium

import grammar_synthesis

def reward_fn(program_text: str, mdp_config: dict):
    return len(program_text)

env = gymnasium.make(
    'GrammarSynthesisEnv-v0',
    grammar=open('grammar_synthesis/envs/assets/example.lark').read(),
    start_symbol='s',
    reward_fn=reward_fn,
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
```

```
l o l l o o o l
l o o o l
l o o l l o o l l o l l o l _s_ _s_ _s_ _s_ l _a_
l o l l o o o o o o o l l o o l l o l
l o l
```

More policies are provided under `grammar_synthesis/policy/`. The current list of included policies are -

1. **Random**: samples a random non-terminal and valid rule at each step
2. **Uniform Random**: implements the algorithm described in [McKenzie, B. (1997)](http://hdl.handle.net/10092/11231) to synthesize strings at random from a CFG
3. **Parsed Playback**: repeats the actions required to synthesize an input program

Their use is demonstrated in `demo.py`.

## Observation Space

The observation space is a fixed length `numpy` array of token IDs representing the current expansion of the string. $l_{\text{max}}$ is the fixed, maximum length of a string. The token IDs represent terminals and non-terminals in the grammar, and begin from $1$. The array is padded out to the fixed length if shorter than it.

## Action Space

The action space is an integer in $[0, |P| \times l_{\text{max}})$ representing the action of first choosing a non-terminal to expand, and the production rule to use to expand it. The index of the non-terminal being expanded $s_i$ and the index of the production rule used $p_i$ are encoded in the action $a$ as -

$
\begin{align}
p_i &= \left\lfloor\frac{a}{l*{\text{max}}}\right\rfloor \\
s_i &= a \bmod l*{\text{max}}
\end{align}
$

The implementation provides an action mask returned in the `info` dict as `info['action_mask']` that masks out invalid actions. Thus, we can sample valid actions repeatedly using `env.action_space.sample(mask=mask)`.

## Theory

The grammar synthesis task $\text{Synthesis}(M, G)$ can be framed as an MDP $M' = (S', A', T', R')$ as follows -

- $S'$: the currently expanded partial program (partial parse tree), represented as a list $[s_0, s_1, ...], s_i \in V \cup \Sigma$
- $A'$: the action of choosing a particular non-terminal in the current state to expand, and the production rule to use
- $T'$: a deterministic transition function that is $1$ for every legal expansion, with the subsequent state being one with the selected
  non-terminal replaced with the rule body of the selected rule, and $0$ otherwise
- $R'$: a reward function that uses the secondary MDP's reward function when we have a complete program (i.e., all symbols in the current state
  are terminals), and $0$ otherwise.
