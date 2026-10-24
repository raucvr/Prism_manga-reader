# Evaluation set

How well does a manga teach its paper? Each quiz in `quizzes/` asks 8 to 10 questions that together cover every
sentence of the paper's abstract, one fact per question, with the abstract's own numbers and qualifiers as the gold
answers. The questions were written from the abstract before any manga existed, and they never change, so scores from
different versions of Prism compare.

| Paper | arXiv | Evaluated in | Known to language models |
|---|---|---|---|
| Attention Is All You Need | 1706.03762v7 | English | yes |
| BERT | 1810.04805v2 | Chinese | yes |
| Chain-of-Thought Prompting | 2201.11903v6 | Chinese | yes |
| Batch Normalization | 1502.03167v3 | English | yes |
| LongZero | 2610.30132v1 | English | no (October 2026) |
| Right On Q | 2610.29378v1 | Chinese | no (October 2026) |
| When Better Opportunities Backfire | 2610.25603v1 | Chinese | no (October 2026) |
| Don't Fake It If You Can't Make It | 2609.29080v1 | English | no (September 2026) |
| Artificial Hivemind (NeurIPS 2025 best paper) | 2510.22954v2 | Chinese | perhaps |
| Gated Attention for LLMs (NeurIPS 2025 best paper) | 2505.06708v1 | English | perhaps |
| 1000 Layer Networks for Self-Supervised RL (NeurIPS 2025 best paper) | 2503.14858v5 | Chinese | perhaps |
| Why Diffusion Models Don't Memorize (NeurIPS 2025 best paper) | 2505.17638v2 | English | perhaps |
| The Flexibility Trap (ICML 2026 outstanding paper) | 2601.15165v4 | Chinese | unlikely (January 2026) |
| High-accuracy Sampling for Diffusion Models (ICML 2026 outstanding paper) | 2602.01338v2 | English | unlikely (February 2026) |

The four famous papers are part of every language model's training data, so a reader can answer much of their quiz
without the manga. The recent ones are not, or much less: their scores show what the manga itself teaches. Every
evaluation also asks the reader without the panels (closed book), which measures what it knew before.

## Run it

```bash
.venv/bin/python -I backend/tools/fetch_papers.py            # pinned PDFs into eval/papers/ (checksums verified)

# a finished manga (made in the app or through the API): a vision model reads only the lettered panels
.venv/bin/python -I backend/tools/eval_quiz.py output/<job_id> --quiz eval/quizzes/bert.json --language zh-CN

# or a script-only dry run (no images; the text cost of a job, up to about $0.80 per paper with the default DeepSeek
# checker): the story layer with mock panels and the same quiz
.venv/bin/python -I backend/tools/dry_run.py eval/papers/bert.pdf --out eval/results/bert --languages en-US,zh-CN --mock \
    --quiz eval/quizzes/bert.json
```
On Windows the interpreter is `.venv\Scripts\python`.

`eval_quiz.py` reports `score` (the share of points from the panels alone), `closed_book` (the same reader without
the panels: what it already knew), `from_charts` (answers taken from a drawn chart or table) and the cost (about two
cents). Defaults: the reader is `core_engine.vision_model` and the grader `core_engine.text_model` (Seed 2.0 Pro in
the example config, as in the published numbers); pass `--reader` / `--grader` to change them, and keep them fixed
when you compare versions, since readers differ in how much they find. The Seed numbers in `README.md` and
`docs/v4-report.md` come from this tool with Seed 2.0 Pro as reader and grader; the Claude numbers there come from
separate, stricter readers.
