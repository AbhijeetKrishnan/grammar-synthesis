# grammar-synthesis

A [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)-based custom environment that uses [Lark](https://github.com/lark-parser/lark) to represent tasks for program synthesis using a context-free grammar (CFG).

Program synthesis is the task of writing a program in a (programming) language that fits some specification.

A CFG $(V, \Sigma, P, S)$ is used to define the space of all programs. A sample grammar is shown below

$$
\begin{align*}
    S &\rightarrow A\ S\ B\ |\ B S \\
    A &\rightarrow a\ A\ |\ B\ C \\
    B &\rightarrow b\ B\ |\ C \\
    C &\rightarrow c \\
\end{align*}
$$

The grammar synthesis task is to find a string belonging to the language of the CFG that optimizes a function, generally the reward function of a secondary MDP (Markov Decision Process).

## Installation

`grammar_synthesis` is installed manually as a Python package using `setuptools`.

```bash
git clone https://github.com/AbhijeetKrishnan/grammar-synthesis.git
cd grammar-synthesis
python3 setup.py install
```

## Usage

The grammar must be specified using [Lark](https://github.com/lark-parser/lark)'s flavor of the EBNF syntax.

```ebnf
s: s s | "l" a "l"

a: "o" a | "o"
```

This sample grammar and others are provided under `grammar_synthesis/envs/assets/`.

The below snippet implements a random policy for generating a string from a grammar. The reward function used is simply
the length of the synthesized program.

```python
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
env.close()
```

```bash
> python random_generator.py
L O L L O O O L
L O O O L
L O O L L O O L L O L L O L s s s s L a L
L O L L O O O O O O O L L O O L L O L
L O L
```

## Observation Space

The observation space is a fixed length `numpy` array of token IDs representing the current expansion of the string. $\text{max\_len}$ is the fixed, maximum length of a string. The token IDs represent terminals and non-terminals in the grammar, and begin from $1$. The array is padded out to the fixed length if shorter than it.

## Action Space

The action space is an integer in $[0, |P| \times \text{max\_len})$ representing the action of first choosing a non-terminal to expand, and the production rule to use to expand it. The index of the non-terminal being expanded $s_i$ and the index of the production rule used $p_i$ are encoded in the action $a$ as -

$$
\begin{align}
    p_i &= \left\lfloor\frac{a}{\text{max\_len}}\right\rfloor \\
    s_i &= a \bmod \text{max\_len}
\end{align}
$$

The implementation provides an action mask returned in the `info` dict as `info['action_mask']` that masks out invalid actions. Thus, we can sample valid actions repeatedly using `env.action_space.sample(mask=mask)`.