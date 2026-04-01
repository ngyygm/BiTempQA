# BiTempQA: A Diagnostic Benchmark for Bitemporal Reasoning in LLM Agent Memory Systems

**BiTempQA** is the first diagnostic benchmark explicitly designed to evaluate bitemporal reasoning — reasoning about *when events occurred* vs. *when the system learned about them* — in LLM agent memory systems.

## Key Features

- **308 Chinese QA pairs** across 10 scenario types and 9 question types at 3 difficulty levels
- **Dual-timestamp annotations**: every memory entry carries explicit `event_time` and `record_time`
- **56.5% of questions require reasoning about both timestamps simultaneously**
- **5 memory backend evaluations**: FAISS, BM25, Naive RAG, Simple KG, ChromaDB

## Dataset

The BiTempQA dataset is available on HuggingFace: [BiTempQA](https://huggingface.co/datasets/ngyygm/BiTempQA)

## Quick Start

### Installation

```bash
pip install -r benchmark/requirements.txt
```

### Run Evaluation

```bash
# Configure your API keys in benchmark/configs/eval_config.yaml
python benchmark/scripts/16_full_evaluation.py
```

### Generate Paper Figures

```bash
python benchmark/scripts/30_paper_figures.py
```

## Project Structure

```
├── paper/                    # ACL 2026 paper (LaTeX source + PDF)
│   ├── main.tex
│   ├── sections/             # Paper sections
│   ├── figures/              # Paper figures
│   └── references.bib
├── benchmark/
│   ├── src/                  # Core library
│   │   ├── generation/       # Scenario & QA generation
│   │   ├── evaluation/       # Evaluation pipeline
│   │   ├── systems/          # Memory backend implementations
│   │   ├── benchmarks/       # Benchmark loaders
│   │   └── schemas.py        # Data models
│   ├── scripts/              # Experiment scripts
│   ├── configs/              # Configuration files
│   ├── tests/                # Unit tests
│   └── data/
│       ├── raw/              # Scenario templates & seed prompts
│       ├── validated/        # Final dataset (train/dev/test)
│       └── eval_results/     # Evaluation reports & statistics
```

## Key Results

| System | Accuracy | F1 | 95% CI |
|--------|----------|-----|--------|
| FAISS | 75.6% | 0.870 | [70.8, 80.2] |
| Naive RAG | 74.0% | 0.867 | [69.2, 78.9] |
| BM25 | 73.4% | 0.867 | [68.2, 78.2] |
| Simple KG | 67.2% | 0.841 | [61.7, 72.4] |

**Key findings:**
- Record-time reasoning is genuinely harder than event-time reasoning (p=0.023)
- ~55% of failures stem from retrieval limitations, not reasoning errors
- System confidence scores are anti-calibrated (AUROC < 0.5)
- Oracle ensemble of all systems achieves 85.4% (+9.8pp)

## Citation

```bibtex
@inproceedings{bitempqa2026,
  title={BiTempQA: A Diagnostic Benchmark for Bitemporal Reasoning in LLM Agent Memory Systems},
  author={Anonymous},
  booktitle={Proceedings of ACL 2026},
  year={2026}
}
```

## License

MIT License
