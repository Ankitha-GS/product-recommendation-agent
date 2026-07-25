#!/usr/bin/env python3
"""
Product Recommendation Agent

Takes a user profile and a product catalogue, and returns a ranked, explained list of recommendations.

Approach (content-based filtering):
  1. Build a TF-IDF vector for every product from its category + tags + description.
  2. Build a "query" vector for the user from their stated preferences, preferred
     categories, and the tags/categories of anything they've already bought.
  3. Rank products by cosine similarity to that query, blended with a small
     rating/popularity signal so well-reviewed items get a slight boost on ties.
  4. Filter by budget when one is given; relax the filter (and say so) if it
     would leave nothing to recommend.
  5. Cold start (no preferences AND no history AND no category hints): fall back
     to a popularity + rating ranked list across the whole catalogue instead of
     guessing, and say clearly that this is a fallback.
  6. Every recommendation ships with a plain-English rationale built from the
     actual matched tags/category/price/rating -- never a canned sentence.

Optional LLM polish:
  If ANTHROPIC_API_KEY is set in the environment, the agent will ask Claude to
  rephrase the rationale into a smoother sentence. This is OFF by default and
  fully optional -- the agent produces complete, correct output with zero API
  keys configured, so reviewers can run it with no setup.

Usage:
  python agent.py --user user_1
  python agent.py --all
  python agent.py --user user_3 --top 3 --polish
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def product_text(p):
    """Flatten a product's attributes into one text blob for TF-IDF."""
    return " ".join([
        p["category"],
        p["category"],  # weight category a bit higher
        " ".join(p["tags"]),
        p["description"],
    ])


def is_cold_start(profile):
    return (
        not profile.get("preferences_text", "").strip()
        and not profile.get("preferred_categories")
        and not profile.get("purchase_history")
    )


def build_query_text(profile, catalogue_by_id):
    parts = [profile.get("preferences_text", "")]
    parts += profile.get("preferred_categories", []) * 2  # weight explicit category prefs

    for pid in profile.get("purchase_history", []):
        prod = catalogue_by_id.get(pid)
        if prod:
            parts.append(prod["category"])
            parts += prod["tags"]

    return " ".join(parts)


def popularity_fallback(catalogue, top_n, exclude_ids):
    """Cold-start ranking: normalized rating + normalized popularity, 50/50."""
    max_pop = max(p["popularity"] for p in catalogue) or 1
    ranked = []
    for p in catalogue:
        if p["id"] in exclude_ids:
            continue
        score = 0.5 * (p["rating"] / 5.0) + 0.5 * (p["popularity"] / max_pop)
        ranked.append((p, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    results = []
    for p, score in ranked[:top_n]:
        rationale = (
            f"No stated preferences or purchase history were available for this user, "
            f"so this is a popularity/rating fallback rather than a personalised match. "
            f"'{p['name']}' is one of the catalogue's top performers "
            f"(rating {p['rating']}/5, popularity index {p['popularity']})."
        )
        results.append({
            "product_id": p["id"],
            "name": p["name"],
            "category": p["category"],
            "price": p["price"],
            "rating": p["rating"],
            "score": round(score, 4),
            "method": "cold_start_popularity_fallback",
            "matched_tags": [],
            "rationale": rationale,
        })
    return results


def recommend_for_profile(profile, catalogue, vectorizer, product_vectors, top_n=5):
    catalogue_by_id = {p["id"]: p for p in catalogue}
    exclude_ids = set(profile.get("purchase_history", []))

    if is_cold_start(profile):
        return popularity_fallback(catalogue, top_n, exclude_ids), {
            "mode": "cold_start_popularity_fallback",
            "budget_relaxed": False,
        }

    query_text = build_query_text(profile, catalogue_by_id)
    query_vec = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vec, product_vectors).flatten()

    budget = profile.get("budget")
    budget_relaxed = False

    def in_budget(p):
        return budget is None or p["price"] <= budget

    candidates = [
        (p, sim) for p, sim in zip(catalogue, similarities)
        if p["id"] not in exclude_ids
    ]

    filtered = [(p, sim) for p, sim in candidates if in_budget(p)]
    if budget is not None and len(filtered) < min(top_n, 3):
        # Not enough affordable matches -- relax the budget filter and flag it.
        budget_relaxed = True
        filtered = candidates

    # Blend similarity with a small quality signal so ties favor better-rated items.
    scored = [
        (p, sim, 0.85 * sim + 0.15 * (p["rating"] / 5.0))
        for p, sim in filtered
    ]
    scored.sort(key=lambda x: x[2], reverse=True)

    query_tokens = set(w.lower() for w in query_text.replace(",", " ").split())

    results = []
    for p, sim, final_score in scored[:top_n]:
        matched_tags = [t for t in p["tags"] if t.lower() in query_tokens
                        or any(t.lower() in tok or tok in t.lower() for tok in query_tokens if len(tok) > 3)]
        matched_tags = list(dict.fromkeys(matched_tags))  # dedupe, preserve order

        category_match = p["category"] in profile.get("preferred_categories", []) \
            or p["category"].lower() in query_text.lower()

        budget_note = ""
        if budget is not None:
            if p["price"] <= budget:
                budget_note = f"fits the stated budget of ₹{budget}"
            else:
                budget_note = f"exceeds the stated budget of ₹{budget}, shown because too few items matched within budget"

        reasons = []
        if matched_tags:
            reasons.append(f"matches preference terms: {', '.join(matched_tags[:4])}")
        if category_match:
            reasons.append(f"falls in a preferred category ({p['category']})")
        if budget_note:
            reasons.append(budget_note)
        reasons.append(f"has a strong rating of {p['rating']}/5")
        if p["id"] in [h for h in profile.get("purchase_history", [])]:
            reasons.append("similar to a past purchase")

        rationale = f"Recommended because it {'; '.join(reasons)}."

        results.append({
            "product_id": p["id"],
            "name": p["name"],
            "category": p["category"],
            "price": p["price"],
            "rating": p["rating"],
            "similarity_score": round(float(sim), 4),
            "final_score": round(float(final_score), 4),
            "method": "content_based_tfidf_cosine",
            "matched_tags": matched_tags,
            "rationale": rationale,
        })

    return results, {"mode": "content_based", "budget_relaxed": budget_relaxed}


def maybe_polish_with_llm(results, profile):
    """Optional: rewrite rationales more fluently using Claude, if a key is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return results  # no key configured -- template rationale stands, agent still fully works

    try:
        import anthropic  # only imported if we actually have a key
        client = anthropic.Anthropic(api_key=api_key)
        for r in results:
            prompt = (
                f"Rewrite this product recommendation reason as one natural, friendly sentence "
                f"(max 30 words), keeping every factual detail: {r['rationale']}"
            )
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
            if text:
                r["rationale"] = text
    except Exception as e:
        print(f"[warn] LLM polishing skipped ({e}); using template rationale instead.", file=sys.stderr)

    return results


def run(profile_id, catalogue, profiles, top_n, polish, outdir):
    catalogue_texts = [product_text(p) for p in catalogue]
    vectorizer = TfidfVectorizer(stop_words="english")
    product_vectors = vectorizer.fit_transform(catalogue_texts)

    profiles_by_id = {p["user_id"]: p for p in profiles}
    targets = profiles if profile_id == "all" else [profiles_by_id[profile_id]]

    all_output = {}
    for profile in targets:
        results, meta = recommend_for_profile(profile, catalogue, vectorizer, product_vectors, top_n)
        if polish:
            results = maybe_polish_with_llm(results, profile)

        output = {
            "user_id": profile["user_id"],
            "name": profile["name"],
            "input_preferences": profile.get("preferences_text", ""),
            "budget": profile.get("budget"),
            "mode": meta["mode"],
            "budget_relaxed": meta["budget_relaxed"],
            "recommendations": results,
        }
        all_output[profile["user_id"]] = output

        print(f"\n=== Recommendations for {profile['name']} ({profile['user_id']}) ===")
        print(f"Mode: {meta['mode']}" + (" [budget relaxed]" if meta["budget_relaxed"] else ""))
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['name']} (₹{r['price']}, {r['rating']}/5) - score {r.get('final_score', r.get('score'))}")
            print(f"   {r['rationale']}")

        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / f"{profile['user_id']}_recommendations.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

    return all_output


def main():
    parser = argparse.ArgumentParser(description="Content-based Product Recommendation Agent")
    parser.add_argument("--catalogue", default=str(BASE_DIR / "catalogue.json"))
    parser.add_argument("--profiles", default=str(BASE_DIR / "profiles.json"))
    parser.add_argument("--user", default=None, help="user_id to recommend for, e.g. user_1")
    parser.add_argument("--all", action="store_true", help="run for every profile in profiles.json")
    parser.add_argument("--top", type=int, default=5, help="number of recommendations to return")
    parser.add_argument("--polish", action="store_true", help="use Claude to polish rationale text if ANTHROPIC_API_KEY is set")
    parser.add_argument("--outdir", default=str(BASE_DIR / "output"))
    args = parser.parse_args()

    if not args.user and not args.all:
        parser.error("specify --user <user_id> or --all")

    catalogue = load_json(args.catalogue)
    profiles = load_json(args.profiles)

    profile_id = "all" if args.all else args.user
    run(profile_id, catalogue, profiles, args.top, args.polish, Path(args.outdir))


if __name__ == "__main__":
    main()
