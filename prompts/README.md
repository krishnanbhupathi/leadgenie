# prompts/

`current.json` is the prompt configuration (few-shot lessons + addendum on top of the
base system prompt in `src/leadgenie/agent.py`) that `leadgenie improve` kept because it
beat the incumbent on the dev split. `run` and `eval` load it by default (`--prompt`).
If it is absent, the base prompt is used.
