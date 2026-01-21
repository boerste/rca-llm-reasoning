# Grounded or Guessing? An Empirical Evaluation of LLM Reasoning in Agentic Workflows for Root Cause Analysis in Cloud-based Systems

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)&ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)&ensp;

> 📣 **Our paper has been accepted to FORGE 2026!**

We contribute a systematic evaluation framework designed to isolate and assess the diagnostic reasoning capabilities of Large Language Models (LLMs) for Root Cause Analysis (RCA).

While LLMs are increasingly adopted for RCA, current approaches often embed them within complex, multi-agent workflows. This entanglement makes it difficult to assess individual agent contributions, obscuring whether a correct diagnosis results from genuine reasoning or auxiliary task heuristics.

We address this by providing a controlled RCA setting that isolates the LLM from confounding factors (e.g., noisy telemetry, inter-agent chatter). By employing simple agent architectures, deterministic tools, and explicit typed knowledge graphs, this framework foregrounds reasoning behavior, allowing for the targeted analysis of both final outputs and intermediate reasoning traces.

<div align="center"> <img src="./.assets/method-overview.png" alt="Method Overview" width="800"/> </div>

## Quick Start

### Installation

The [`install.bash`](./install.bash) script installs micromamba, a new micromamba environment, and all the dependencies requires for this project. 
Use the following commands:

```bash
# Clone the repository
git clone https://github.com/boerste/rca-llm-reasoning.git
cd rca-llm-reasoning

# Install and activate environment
bash install.bash --micromamba # if not already installed
bash install.bash --env-name=llm-rca --deps
micromamba activate llm-rca

# Install ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull ollama models
ollama pull llama3.2:3b
ollama pull qwen3:4b
ollama pull qwen3:32b
ollama pull llama3.3
ollama pull command-r-plus
```

For our experiments, we used the following:
- [`ollama == 0.11.4`](https://github.com/ollama/ollama/releases/tag/v0.11.4)
- `langchain == 0.3.27`
- `langchain-core == 0.3.74`
- `langchain-ollama == 0.3.6`
- `langgraph == 0.6.4`

And Ollama models:
- [`llama3.2:3b`: ID=a80c4f17acd5](https://ollama.com/library/llama3.2:3b)
- [`qwen3:4b`: ID=e55aed6fe643](https://ollama.com/library/qwen3:4b)
- [`qwen3:32b`: ID=030ee887880f](https://ollama.com/library/qwen3:32b)
- [`llama3.3`: ID=a6eb4748fd29](https://ollama.com/library/llama3.3:70b)
- [`command-r-plus`: ID=e61b6b184f38](https://ollama.com/library/command-r-plus:104b)

## 📚 Citation

If you use our work in your research, please cite our paper:

```bibtex
@inproceedings{
    riddell2026,
    title={{Grounded Or Guessing? An Empirical Evaluation of LLM Reasoning in Agentic Workflows for Root Cause Analysis in Cloud-based Systems}},
    author={Evelien Riddell and James Riddell and Gengyi Sun and Michal Antkiewicz and Krzysztof Czarnecki},
    booktitle={The ACM International Conference on AI Foundation Models and Software Engineering (FORGE)},
    year={2026}
}
```
