# Product Recommendation Agent

A content-based product recommendation agent built for the Rooman AI Challenge
(Category 3 — Customer & Growth).

> **One-sentence job:** My agent takes a user's stated preferences, budget, and
> optional purchase history, and produces a ranked list of products from a
> catalogue, each with a plain-English reason.

---

## What it does

1. Loads a product catalogue (`catalogue.json`) — 24 products across 6
   categories, each with `category`, `price`, `rating`, `popularity`, `tags`,
   and a `description`.
2. Loads user profiles (`profiles.json`) — 4 sample users, one of which is a
   deliberate **cold-start** case (no preferences, no history, no category
   hints).
3. For a normal user: builds a TF-IDF vector for every product and a matching
   "query" vector from the user's preference text + preferred categories +
   the tags/category of anything in their purchase history, then ranks
   products by cosine similarity (blended 85/15 with normalized rating so
   ties favor better-reviewed items).
4. Applies the user's budget as a filter. If fewer than 3 items would survive
   the filter, the agent **relaxes it and says so explicitly** rather than
   returning an empty or misleadingly short list.
5. For the cold-start user: skips content matching entirely (there is nothing
   to match against) and falls back to a transparent popularity + rating
   ranking, clearly labelled `cold_start_popularity_fallback` in the output.
6. Every single recommendation gets a rationale generated from the actual
   matched tags, category match, budget fit, and rating — never a static
   template sentence with no product-specific content.

This is the "Input → Think → Act → Output" loop: user profile in, TF-IDF /
similarity computation as the thinking step, catalogue filtering as the
action, ranked JSON + printed table as the output.

---

## Project structure

```
product-recommendation-agent/
├── agent.py              # the agent
├── catalogue.json        # product catalogue (deliverable)
├── profiles.json         # 4 sample user profiles (deliverable)
├── requirements.txt
├── README.md
└── output/                       # generated on run (deliverable: recommendation output)
    ├── user_1_recommendations.json
    ├── user_2_recommendations.json
    ├── user_3_recommendations.json
    └── user_4_recommendations.json
```

---

## Setup

Requires Python 3.9+.

```bash
cd product-recommendation-agent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No API key is required to run the agent — the core recommendation logic
(TF-IDF + cosine similarity) is fully local and deterministic.

### Optional: LLM-polished rationale

An optional step can ask Claude to rephrase the rationale into a single
smoother sentence, without changing any of the underlying facts. This is
**off by default** so the project is foolproof to run out of the box. To try
it:

```bash
export ANTHROPIC_API_KEY=your_key_here     # Windows: set ANTHROPIC_API_KEY=your_key_here
python3 agent.py --user user_1 --polish
```

If the key isn't set, `--polish` is silently ignored and the template
rationale (already complete and specific) is used instead.

---

## Running the agent

Run for one user:

```bash
python3 agent.py --user user_1
python3 agent.py --user user_2 --top 3
python3 agent.py --user user_3
python3 agent.py --user user_4          # the cold-start profile
```

Run for every profile at once (this is what generated the sample output in
`output/`):

```bash
python3 agent.py --all --top 5
```

Each run prints a ranked table to the console and writes
`output/<user_id>_recommendations.json`.

### CLI options

| Flag | Description |
|---|---|
| `--user <id>` | run for a single user (e.g. `user_3`) |
| `--all` | run for every profile in `profiles.json` |
| `--top <n>` | number of recommendations to return (default 5) |
| `--polish` | optionally rephrase rationale via Claude if `ANTHROPIC_API_KEY` is set |
| `--catalogue <path>` | override catalogue file (default `catalogue.json`) |
| `--profiles <path>` | override profiles file (default `profiles.json`) |
| `--outdir <path>` | override output directory (default `output/`) |

---

## Sample profiles included

| user_id | Scenario |
|---|---|
| `user_1` | Priya — clear fitness/running preferences, has 1 past purchase, ₹5,000 budget |
| `user_2` | Ahmed — tech/audio/camera preferences, has 1 past purchase, ₹25,000 budget |
| `user_3` | Sara — budget skincare for sensitive skin, no purchase history, ₹1,500 budget |
| `user_4` | Rahul — **pure cold start**: no preferences, no categories, no history, no budget |

Sample generated output for all four is committed in `output/`.

---

## Design choices & why

- **Content-based over collaborative filtering**: with a brand-new catalogue
  and no real interaction logs across many users, collaborative filtering
  has nothing to learn from. Content-based (TF-IDF + cosine similarity)
  works from day one for any single user and is fully explainable — a hard
  requirement given the "rationale for every recommendation" deliverable.
- **TF-IDF over embeddings/LLM-only matching**: TF-IDF is deterministic,
  needs no API key, runs in milliseconds, and its output is directly
  inspectable (you can see exactly which words drove the score). An LLM
  could also do this matching, but it would be slower, non-deterministic,
  harder to unit test, and not meaningfully more accurate on a 24-item
  catalogue.
- **85/15 similarity/rating blend**: pure similarity ranking can surface a
  poorly-rated product purely because it shares a lot of vocabulary with the
  query. A small rating-based nudge breaks ties toward products that are
  actually good, without letting rating override genuine relevance.
- **Explicit budget relaxation instead of silent failure**: an agent that
  quietly returns 1 result (or 0) when a budget is tight looks broken. The
  agent relaxes the filter and **labels the output** `budget_relaxed: true`
  so it's honest about what happened.
- **Separate, clearly-labelled cold-start path**: guessing at a user's taste
  from nothing produces recommendations that look personalised but aren't.
  Instead, the agent detects this case and switches to a transparent
  popularity/rating fallback, which is a defensible strategy the user can
  understand and reviewers can verify (`mode` field in the output makes this
  explicit).
- **Optional LLM polish, not required**: the assignment asks the agent to
  "use an AI language model." The LLM step exists and can be exercised with
  `--polish`, but making it mandatory would break reproducibility for anyone
  without an API key — so it's an enhancement, not a dependency.

---

## Tradeoffs and what I'd improve with more time

- **Catalogue size (24 items)**: enough to demonstrate the pipeline across 6
  categories and a cold-start case, but too small to show TF-IDF's real
  strength. With more time I'd load a larger public product dataset (e.g. an
  Amazon or Flipkart product export) and add pagination.
- **No real collaborative signal**: purchase history currently only widens
  the content query (adds the bought item's tags/category). With more users
  and real transaction logs, a hybrid content + collaborative model
  (e.g. matrix factorization or item-item similarity from co-purchases)
  would likely outperform pure content-based matching, especially for users
  with rich history.
- **Tag matching is lexical, not semantic**: "trainers" wouldn't match
  "shoes" even though they mean the same thing. Swapping the TF-IDF
  vectorizer for a sentence-embedding model (e.g. `all-MiniLM-L6-v2`) would
  catch synonyms at the cost of needing to ship/download a model.
- **Budget relaxation is currently all-or-nothing**: it either respects the
  budget strictly or drops the filter entirely. A softer version would rank
  in-budget items first and only pull in over-budget items to fill remaining
  slots, rather than mixing them by score.
- **No explicit negative feedback loop**: the agent doesn't yet handle "I
  don't want X" signals or thumbs-down feedback that could down-rank similar
  items in future runs.
- **Single-shot CLI, no persistence**: preferences and history are read from
  a static JSON file per run. A production version would read/write a
  small SQLite table so a user's history updates automatically after each
  interaction.

---

## Example: recommendations for `user_3` (Sara, budget skincare, cold-start-free)

```
1. HydraCloud Daily Moisturizer (₹649, 4.3/5) - score 0.7412
   Recommended because it matches preference terms: skincare, beauty, budget,
   sensitive skin; falls in a preferred category (Beauty); fits the stated
   budget of ₹1500; has a strong rating of 4.3/5.
2. PureMinerale SPF 50 Sunscreen (₹549, 4.6/5) - score 0.6035
   ...
```

Full output for all 4 profiles is in `output/*.json`.
