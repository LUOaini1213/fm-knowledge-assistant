# FM Knowledge Assistant

Cited search and question answering over Singapore building and facilities-management codes. The system retrieves clauses and checks recognized generated quantities against locally cited source statements: value, unit, comparator and a conservative lexical subject signature. Detected unsupported or ambiguous quantity statements are withheld. These bounded checks do not establish general semantic entailment or applicability. The system also extracts quantitative requirements and evaluates passage classification.

**Status (measured 2026-10-14 with qwen2.5:3b):** the guard withheld every generated answer. 0 of 40 answerable questions got a shown answer, although 33 and 34 of the 40 drafts (two runs) stated the expected values with a gold citation; 8 of 8 unanswerable questions were refused and no wrong answer was shown. In practice the system is cited-clause search with abstention: hybrid retrieval ranks a gold clause or page first for 36 of the 40 answerable questions (Hit@1 0.900, MRR@20 0.946). The requirement extractor scores F1 0.815 on a 20-passage test set. Main limits: eight documents, 48 questions written by me, and an answer path that needs a model quoting the source more closely or a guard that accepts paraphrased subjects. The result tables and the numbers in this status line are checked against the committed result files by `python eval/check_readme.py` in CI.

- **Retrieval:** hybrid. BM25 keyword search is combined with dense vectors (bge-small) through reciprocal-rank fusion. The vectors are held in DuckDB with an HNSW index (the `vss` extension). Document restrictions apply before each candidate limit, so an allowed passage is not lost behind more highly ranked documents outside the filter. Omitted or empty `docs` means all documents; a nonempty filter with no matching documents returns no hits.
- **Grounded answers:** a draft is shown only if all of these hold:
  - it cites the supplied passages;
  - recognized numeric values appear in cited passages, with no unsupported numeric expressions;
  - each supported quantitative statement has local citations and matches the source quantity's value, unit, comparator and lexical subject; an unbound or ambiguous quantity is unverified. A citation binds to the one sentence next to it, whether it is written after the sentence ("X [1].", "X. [1]") or before it ("[1] X.", "According to [1], X.");
  - the question is close enough to the corpus.
  Low similarity or `NOT_FOUND` yields an abstention. A draft that fails validation is withheld with `unverified` status, which means the draft failed the checks, not that the corpus lacks the answer. CLI/API do not publish that draft; the Python `Answer.text` retains it for internal evaluation.
- **Document intelligence:**
  - extracts requirements such as "not less than 850 mm" as (comparator, value, unit, subject);
  - classifies passages into their code and chapter.
- **Serving:** a FastAPI service (`/search`, `/ask`, `/extract`) with a Dockerfile. CI runs the tests (Linux and Windows), the mutation check and the README number check, and builds the image.
- **LLM:** any OpenAI-compatible endpoint. The evaluation uses a local 3B model (qwen2.5:3b through Ollama on a 4 GB GPU), so no data leaves the machine and no key is needed. DeepSeek or Azure OpenAI can be used through the same client, but neither was tested here.

## Corpus

Eight official documents, all publicly downloadable:

| id | Document | Agency |
|---|---|---|
| firecode2023 | Code of Practice for Fire Precautions in Buildings 2023 (285 pages) | SCDF |
| accessibility2025 | Code on Accessibility in the Built Environment 2025 v1.1 | BCA |
| copeh2024 | Code of Practice on Environmental Health (2024) | NEA |
| lift_obc2024 | Guide on Specifications for Outcome-Based Lift Maintenance Contracts (2024) | BCA |
| psi_owner2024, psi_engineer2024 | Periodic Structural Inspection of Existing Buildings: owner's guide and engineers' guidelines (2024) | BCA |
| gm_maintainability_eb | Green Mark 2021 Maintainability (Existing Buildings) Technical Guide v2.1 | BCA |
| gm_energy | Green Mark 2021 Energy Efficiency Technical Guide R2.1 | BCA |

**Licensing.** SCDF's and BCA's terms of use do not allow the contents to be republished, so this repository contains no document text, chunks or vectors. `sources.json` lists the official URLs and the SHA-256 of the tested editions, and `python -m fmka.cli fetch` downloads them for local use. PUB's codes were left out because PUB's terms forbid storing its material in an information retrieval system. The evaluation questions are written in my own words and point to clause numbers rather than quoting the text.

Parsing cuts numbered documents at clause numbers (for example "2.3.9", "CLAUSE 1.1", "4.2.1.2"), giving 2,180 chunks. Three safeguards keep the numbering clean:

- Contents pages are skipped.
- A new chapter must start at N.1, so a table value such as "12.5" inside chapter 3 is not read as a clause.
- Running headers and footers are removed.

## Evaluation results

Retrieval and requirement extraction were rerun on 2026-10-14 at the code of that date and reproduce the tables below exactly. Generated answers were measured again on the same day, in two runs; see [Grounded answering](#grounded-answering-48-questions). The earlier answer table (v1–v4) was measured before the quantity/citation guard was added on 2026-10-10. It is kept as history and does not describe the current system. Corrected classification and the synthetic checker comparison use separate result files.

### Retrieval (40 answerable questions)

A hit means a chunk with the gold clause or page is ranked within the top k.

| Mode | Hit@1 | Hit@5 | MRR@20 |
|---|---|---|---|
| BM25 | 0.875 | 1.000 | 0.933 |
| Dense (bge-small, HNSW) | 0.825 | 1.000 | 0.894 |
| **Hybrid (RRF)** | **0.900** | **1.000** | **0.946** |

### Grounded answering (48 questions)

The set has 40 answerable questions and 8 that the corpus does not cover. Each unanswerable question was checked by keyword search to have no answer in the corpus, and several are deliberate near misses: the codes have cooling-tower siting rules but no cleaning interval, and toilet illuminance but not office illuminance.

The questions are split into a development set (q01–q20, u01–u04) and a test set (q21–q40, u05–u08). The abstention threshold (0.725 cosine) and the archived v2–v4 fixes were chosen from development errors. The 2026-10-14 citation-placement fix came from an independent test of the app and from development drafts.

An answer counts as correct only if it states the expected values **and** cites a gold clause.

#### Guard at `e091712`, measured 2026-10-14

The model is qwen2.5:3b-16k through Ollama at temperature 0, on a GTX 1650 that another job was also using. Requests that timed out were sent again unchanged. The table also scores each model draft, whether or not it was shown. Per-question records, without draft text, are in [`eval/results/qa_runs_2026-10-14.json`](eval/results/qa_runs_2026-10-14.json).

| Guard | Run | Shown and correct (dev / test) | Shown but wrong | Withheld as unverified (dev / test) | Answerable abstained | Drafts correct with gold citation (dev / test) | Unanswerable abstained / 8 |
|---|---|---|---|---|---|---|---|
| `13a209d`, before the citation-placement fix | 1 | 0 / 0 | 0 | 19 / 19 | 2 | 18 / 16 | 8 |
| `e091712` (measured 2026-10-14) | 1 | 0 / 0 | 0 | 18 / 19 | 3 | 17 / 16 | 8 |
| `e091712` (measured 2026-10-14) | 2 | 0 / 0 | 0 | 18 / 20 | 2 | 17 / 17 | 8 |

Later code was not re-measured. The only guard change after `e091712` (in `427472b`) also rejects any remaining mathematical symbol such as ≠, ± or ≈, in answers and sources. It can only withhold more drafts, so "0 shown" still holds; the withheld and draft counts belong to `e091712`.

What the measurement shows:

- **The measured guard shows no generated answer.** In both runs, none of the 40 answerable questions got a shown answer. All 8 unanswerable questions were refused and no wrong answer was shown. When a draft is withheld, `/ask` returns the withheld message with references to the cited and retrieved clauses.
- **The model's drafts are mostly right.** 33 and 34 of the 40 drafts state the expected values and cite a gold clause. The guard withholds all of them.
- **Citation placement was one cause, and it is fixed.** The model usually writes the citation before the sentence ("[1] A spiral staircase ..."), while the guard bound a citation only to the text before it. Before the fix, 29 of 40 drafts had a statement with no valid local citation. A citation now binds to the one sentence next to it, on either side, and sentence-initial phrases such as "According to [1]," count as citations. After the fix, 12 and 13 of 40 drafts still have an uncited statement. In most of these the model cites only the first sentence of a multi-sentence answer, and the guard refuses the rest by design.
- **The quantity grammar is now the limiting check.** As a diagnostic, each run's drafts were rescored with every sentence credited with every citation in its draft. This binding is deliberately too permissive and is not used by the system. Even so, only 1 of the 33 or 34 correct drafts passes, and 1 wrong draft also passes. Most correct drafts paraphrase the source, and the subject check requires the source clause's own wording. Others state two quantities in one sentence ("1500 mm by 1750 mm") or a quantity the grammar does not parse. These checks are what keep subject-transfer errors out, so they have not been loosened to raise the answer rate.
- **Run-to-run variation.** The two runs differ in the text of 10 of the 48 drafts, in the status of 1 question (q29 abstained as NOT_FOUND in run 1 and was withheld in run 2), and in the correctness of 1 draft.

#### History: before the quantity/citation guard (archived, measured before 2026-10-10)

These runs used the earlier numeric-membership check, which accepted any number found in a cited passage. [`eval/results/qa_runs.json`](eval/results/qa_runs.json) keeps them unchanged.

| Run | Change | Test: correct / 20 | Test: shown but wrong | Unanswerable abstained / 8 |
|---|---|---|---|---|
| v1 | first run | 14 | 3 | 8 |
| v2 | numbers in the passage header count as sourced; retry once when the model forgets to cite | 16 | 1 | 8 |
| v3 | the retry prompt shows "[1]" instead of the literal "[n]" that the model had copied | 18 | 2 | 8 |
| v4 | tokenizer splits clause numbers from words ("cl.2.3.9" → "cl", "2.3.9") | 17 | 2 | 8 |

Notes on the archived runs:

- **q36 gold widened after v1.** The fix in v2 also added a second gold passage to q36, because page 70 states the same 1.45 baseline.
- **Run-to-run variation.** v3 and v4 differ by one test question with the same model and near-identical code.
- **What the checks caught** in v4:
  - q09: the model invented "1:8". The number check withheld it.
  - q39: an extra number, "5000", appeared that is not in the cited passage. The answer was withheld.
- **The two wrong answers** in v4:
  - q38 is incomplete: it gives the ±5% tolerance but not the 80% condition.
  - q33 describes the evaluation framework without the literal "KPI" that the metric looks for.

**An archived failure of numeric membership.** Asked "How often must cooling towers be cleaned?" at a looser threshold (0.70), the model answered "every 12 months" and cited NEA clause 3.4. That clause gives 12 months for kitchen exhaust ducts, not for cooling towers. The old number check passed because "12" was in the cited passage. At the default threshold of 0.725 the question was refused, because its best match scored 0.712. This failure is why the quantity/citation guard exists. The guard is tested with synthetic examples of subject transfer and subjectless quantity fragments.

I tried two further checks against this failure:

- **Word overlap between the question and the sentence holding the number** could not separate right from wrong. Some correct answers scored 0.08, while this wrong answer scored 0.25.
- **A second LLM check** is available as `strict=True`. In the earlier whole-answer experiment, the 3B model rejected this case, but also rejected 7 of 34 correct answers and neither of the two wrong ones, so it remains off by default. The current implementation checks locally cited claims separately and requires every claim to be covered. Its model accuracy has not been measured again.

### Offline quantity-binding challenge cases

[The authored suite](eval/claim_guard_cases.jsonl) contains 34 incorrect bindings, 24 correct examples and two unscored policy boundaries. It covers unit and object transfers, changed comparisons, local citation swaps, metadata borrowing, multiple quantities, supported number spellings and unsupported forms. Since 2026-10-14 it also covers citations written before their sentence ("[1] X.", bullets, "According to [1], X."), with swapped, borrowed and doubly used leading citations among the incorrect bindings. Examples were written for checker review, not sampled from the corpus or produced by a model.

`python eval/eval_claim_guard.py` compares the frozen `8b8591f` numeric-membership checker with the current checker. [The report](eval/results/claim_guard.json) records every acceptance/refusal and both source fingerprints. `--check` reproduces the report without writing and fails if any incorrect challenge example is accepted. Correct examples refused by the conservative policy are recorded separately; neither this suite nor its rejection rate is a new LLM quality score.

| Checker | Incorrect examples rejected / 34 | Correct examples accepted / 24 | Correct examples refused / 24 |
|---|---|---|---|
| Frozen numeric membership (`8b8591f`) | 3 | 24 | 0 |
| Current quantity/citation guard | 34 | 24 | 0 |

Both unscored boundaries are conservatively refused: an omitted applicability condition and a logically weaker comparator. Zero refusals in these authored positive examples does not establish a production refusal rate; unfamiliar but valid paraphrases can still be withheld.

### Requirement extraction

The extractor was scored on hand-labelled passages. A match needs the same comparator, value and unit, and matching is one-to-one.

| Split | Passages | Precision | Recall | F1 |
|---|---|---|---|---|
| Dev, first rules | 30 | 1.000 | 0.452 | 0.622 |
| Dev, after writing rules from its errors | 30 | 1.000 | 0.984 | 0.992 |
| **Test** (sampled and labelled only after the rules were fixed) | 20 | **0.943** | **0.717** | **0.815** |

Gold rows are keyed by the chunk id of a locally rebuilt index. `eval/eval_extraction.py` therefore refuses to score when a PDF in `data/raw` is not the tested edition from `sources.json`, when a gold chunk id is missing, or when a row pinned with `text_sha256` points at different text (`--pin-gold` writes those hashes). New runs record commit, index, gold and source fingerprints in the result file. The committed `extraction.json` was produced before this check and has no provenance block; it needs a rerun on the tested editions to gain one.

Recall on the test set is what to expect on new text. The misses are:

- one comparator shared by several values ("at least 900 mm wide, 480 mm deep");
- "from 900 mm to 1500 mm";
- sizes stated without a comparator word;
- symbols such as ">500mm".

### Passage routing

Stratified 5-fold cross-validation. The labels come from the documents' own structure, so no hand labelling was involved. Both feature families encode the same cleaned heading and body. Clause references are removed and no document-title or clause metadata is prepended. The pretrained sentence encoder is frozen; TF-IDF vocabulary and both classifiers are fitted separately within each training fold. `python -m fmka.classify` writes `eval/results/classification_corrected.json`.

The run on the existing 2,180-chunk local index produced the following results. The index fingerprint, cached model revision and actual package versions are recorded in [the run provenance](eval/results/classification_corrected_provenance.json).

| Task | TF-IDF + LR acc / macro-F1 | Clean-content bge-small + LR acc / macro-F1 |
|---|---|---|
| Source document (8 classes) | 0.988 / 0.958 | 0.972 / 0.923 |
| Fire Code chapter (11 classes) | 0.852 / 0.771 | 0.804 / 0.773 |
| Accessibility chapter (9 classes) | 0.819 / 0.778 | 0.836 / 0.809 |

TF-IDF has the higher accuracy on source document and Fire Code chapter (Fire Code macro-F1 differs by 0.002); the embedding is ahead on both measures for the Accessibility chapter. These are within-corpus, chunk-level folds, not an evaluation on unseen documents; adjacent or similarly worded clauses can still make that task easier.

**Archived, with label leakage (not a valid comparison).** The earlier code reused retrieval vectors whose inputs included the document title and clause number; only the TF-IDF branch received cleaned text. [`classification.json`](eval/results/classification.json) keeps that run for audit. F1 is macro-F1.

| Task | Passages | Classes | Majority | TF-IDF + LR | bge-small + LR (leaky) |
|---|---|---|---|---|---|
| Source document | 2,180 | 8 | acc 0.638 | acc 0.988 / F1 0.958 | acc 0.996 / F1 0.975 |
| Fire Code chapter | 1,390 | 11 | acc 0.187 | acc 0.852 / F1 0.771 | acc 0.895 / F1 0.864 |
| Accessibility Code chapter | 371 | 9 | acc 0.307 | acc 0.819 / F1 0.778 | acc 0.852 / F1 0.824 |

TF-IDF results are identical in both runs. Without the leaked metadata the embedding no longer has uniformly higher accuracy.

## Tests

The tests cover:

- extraction patterns;
- BM25, fusion and the tokenizer;
- every answer check;
- the clause-numbering rules;
- an end-to-end run on a generated PDF with a hashing embedder and a scripted LLM, so no model download is needed;
- the API endpoints, including withholding failed drafts and providing readable extractive passages;
- complete English and digit numeric expressions, cited-source boundaries, and unsupported syntax;
- local quantity/citation binding, unit aliases, exact comparisons, subject mismatches, withheld subjectless answers, and per-claim optional judging;
- unsupported mathematical qualifiers such as ≠, ± and ≈ in answers or sources cannot silently become exact quantities;
- citation placement: trailing, leading, bulleted and reporting-phrase citations each bind to one sentence, and swapped, borrowed or doubly used citations are refused;
- restricted retrieval with matching documents beyond the global candidate pool;
- overlapping HTTP searches and answers keep their DuckDB result cursors, document filters and source references isolated;
- failed and successful index publication, preserving the previous database until the replacement is complete;
- classification feature inputs and real cross-validation training/validation isolation;
- the extraction evaluation refusing other PDF editions, missing gold chunks and changed passage text;
- every number in this README's result tables and status lines against `eval/results` (`eval/check_readme.py`), including that a changed cell or extra row is reported;
- the Docker start command with fake bootstrap programs in a real POSIX `sh` (on Windows the one from Git for Windows; WSL only if `FMKA_TEST_ALLOW_WSL=1`).

`tests/mutate.py` plants deliberate bugs, and every one must fail at least one test. It first checks the full unmodified suite, requires unique and compilable mutations, and only counts actual JUnit test failures as caught defects. Syntax errors, collection/setup errors, process failures and timeouts are not kills. Target files are restored byte-for-byte even on failure. Run `python tests/mutate.py --self-test` to check the harness itself. Mutation runs temporarily edit source: run them exclusively, with no concurrent editor, test, build or application process using the checkout.

## Answer interface and limits

`POST /ask` returns `answer` only for `status="answered"`. Other states return `answer: null` with a `message`. In `extractive` mode, `extracts` contains the readable passage text and its document, clause and page reference. `sources_considered`, `citations` and `problems` remain available for diagnosis. A client should use the status rather than treating any nonempty field as a verified answer.

The shared numeric scanner recognizes decimal and scientific literals, grouped thousands, complete multi-level clause IDs, and English cardinals through millions with digit words after `point`. It normalizes equivalent spellings without authorizing component words of a compound number. The additional guard binds supported quantities to local citations, canonical unit aliases, exact comparators and ordered lexical subjects/predicates. It does not convert units, infer arbitrary synonyms, resolve pronouns or reconstruct an omitted subject from the question. Reference-only metadata numbers are checked against their local citations; a page number or year cannot supply a physical quantity. Detected unsupported quantitative grammar is withheld, and no-client extractive mode still returns readable original passages. Arbitrary languages, unrecognized number spellings and general factual assertions remain outside this bounded parser.

Matching a quantity tuple does not prove that a source's conditions, exceptions or broader context apply to the question. Non-quantitative factual entailment is outside the deterministic guard. Exact comparator matching can conservatively refuse a logically weaker paraphrase. The two explicit challenge boundaries make these limits visible; optional model judging also provides no proof of correctness.

## Run it

```bash
# Python 3.11, from the repository root (bash, or Git Bash on Windows)
pip install -r requirements.txt   # or requirements-ci.txt: no torch, enough for the tests and checks
pip install -e . --no-deps        # makes the fmka package importable (alternative: export PYTHONPATH=src)
python eval/check_readme.py       # every README number against eval/results; no model or corpus
python -m fmka.cli fetch      # official PDFs into data/raw (local use only)
python -m fmka.cli build      # parse, embed, index into data/processed/fmka.duckdb
python -m fmka.cli search "minimum clear width of an exit door"
FMKA_LLM_BASE_URL=http://localhost:11434/v1 FMKA_LLM_MODEL=qwen2.5:3b-16k python -m fmka.cli ask "What is the minimum clear width of an exit door?"
uvicorn fmka.api:app --port 8000                         # or: docker build -t fmka . && docker run -p 8000:8000 fmka
python -m pytest -q tests && python tests/mutate.py
python eval/eval_claim_guard.py --check  # offline synthetic checker comparison; no model or corpus
FMKA_LLM_BASE_URL=... python eval/run_eval.py && python eval/eval_extraction.py
python -m fmka.classify      # corrected classification features; separate result file
```

In PowerShell, set the LLM variables with `$env:FMKA_LLM_BASE_URL = "http://localhost:11434/v1"` (and likewise `FMKA_LLM_MODEL`) instead of the inline `VAR=value` prefix. The Docker image fetches and indexes the documents on first start, and stops if either step fails. Rebuilding the index constructs a separate database and replaces the previous one only after all tables and vectors are written and closed. On Windows an open reader may prevent replacement; close readers before rebuilding and restart long-running API processes to use the new index.

## Limits

- The corpus is eight documents. Tables and drawings are read as flattened text, so answers that live only in a table cell or a figure are weak.
- With qwen2.5:3b, the quantity/citation guard currently withholds every generated answer in the evaluation set (see above). In practice the system is a cited-clause search with abstention. Its answer path needs a model that quotes the source more closely, or a guard that recognises paraphrased subjects without letting subject transfer through.
- Abstention relies on a similarity threshold chosen on 24 development questions, and the answerable and unanswerable similarity ranges overlap (0.689–0.893 against 0.683–0.741).
- The evaluation set is small (48 questions), and I wrote it myself.
