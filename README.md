This is a clone of original repo: https://github.com/IINemo/thinkbooster

## Setup

```
./setup.sh
pip install vllm==0.15.1
pip install "git+https://github.com/vllm-project/speculators.git@fda222eae200c017f941c5b2388f9d65d3b610b2"
```

## Run experiments

- 5 datasets (Math500, Olympiadbench, ScienceQA, CommonsenseQA, Knights&Knaves)
- 3 LLMs (Qwen2.5-Math-1.5B-Instruct, Gemma3-4b-it, Llama3.2-3B-Instruct)
- 2 verifiers (PRM: Qwen/Qwen2.5-Math-PRM-7B, UHead=ReProbe: TODO)

Example SLURM jobs: `llm_tts/example_slurm_runs` (this is only beam-search with PRM verifier)

## How to add new strategy?

1. Implement your strategy in `llm_tts/strategies/strategy_X.py`
    - for LLM generation **only** use `step_generator` argument
    - for verifier scoring, **only** use `scorer` argument
    - implement `def generate_trajectories_batch` with same interface for fast evaluation
    - important: in the end of `generate_trajectories_batch`, report total generated tokens and average tokens per sample (format: `[X] TOTAL generated tokens: A | avg per sample: B`)
2. Include it in `llm_tts/strategies/__init__.py`
3. In `llm_tts/run_tts_eval.py`:
   - include strategy
   - create separate branch for it in `def create_strategy`
4. In `configs/my_exps`, add separate `.yaml` config for strategy
5. Run on 5 datasets x 3 LLMs x 2 verifiers using examples in `llm_tts/example_slurm_runs`. For parameters:
   - use default decoding width of 25, e.g.: beam-search - 5 chains x 5 candidates; self-consistency - 25 chains
   - `max_steps=50`
   - `generation.temperature=1.0 generation.top_k=1000 generation.top_p=1.0`
