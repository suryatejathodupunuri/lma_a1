# RAG: Baseline Retrieval vs. Re-Ranking

Compare two RAG pipelines on the 32-page *Spirit Island* core rulebook:

- **Baseline:** dense retrieval of ten chunks → original top three → LLM answer.
- **Re-ranking:** same ten chunks → Cross-Encoder → re-ranked top three → LLM answer.

Both pipelines use the same corpus, prompt template, and generation settings. Three selected failure queries demonstrate rank shifts and answer differences; retrieval latency is measured over five trials per question.

## Files

- `rag_assignment.ipynb`: implementation and saved outputs.
- `report.pdf`: report covering architecture, failure analysis, latency, and scaling.
- `data/`: source rulebook PDF.
- `artifacts/`: extracted text, chunks, embeddings, ranks, answers, and latency results.
- `requirements.txt`: dependencies; `requirements-lock.txt`: exact versions used.
- `.env.example`: API-key configuration template.

## Setup

Python 3.12 was used. Embeddings and re-ranking run on CPU.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-lock.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env`. Open the notebook, select `.venv/bin/python` as its kernel, and run cells in order. Saved outputs can be inspected without execution. Matching cached responses are reused; new prompts require Gemini API calls.

## Models and Data

- Bi-Encoder: `sentence-transformers/all-MiniLM-L6-v2`.
- Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L6-v2`.
- LLM: `gemini-3.8-flash`.
- Dataset: [Spirit Island core rulebook, linked by Greater Than Games](https://shop.greaterthangames.com/products/spirit-island).

Chunk size is 220 tokens with 40-token overlap, producing 144 chunks. Gemini uses temperature 0, a 2048-token output limit, and low thinking level. Selected stress-test results are not an estimate of overall accuracy.
