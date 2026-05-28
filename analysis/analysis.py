# =============================================================================
# Bachelor Thesis — Complete Analysis Implementation
#
# Appendix A: Dataset Analysis (Section 3.3 / Chapter 4)
#   Generates all statistical analyses, tables, and figures for Chapter 4.
#
# Appendix B: Surface Marker Validation Test (Chapter 5)
#   Validates whether lightweight regex markers can predict high-context
#   ambiguity tier before full semantic analysis. Uses an 80/20 held-out
#   split to avoid circularity and report unbiased precision/recall.
#
# Dependencies: pandas, matplotlib, numpy (Python 3.8+)
# Input:        data/Cornelius_2025_user_story_ambiguity_dataset.xlsx
# Output:       results/ (CSV files), figures/ (PNG files)
#               Filenames correspond to thesis section numbers.
# =============================================================================

import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# =============================================================================
# APPENDIX A — Dataset Analysis
# =============================================================================

# -----------------------------------------------------------------------
# 4.1 Dataset Overview
#
# Load the Cornelius (2025) User Story Ambiguity Dataset and prepare it
# for analysis. The dataset contains user stories annotated with seven
# binary ambiguity labels and a pre-computed AmbiguityCount column.
#
# The seven ambiguity types are:
#   Semantic    — vague or ambiguous word meaning
#   Scope       — unclear system boundary or feature extent
#   Actor       — undefined or generic role/stakeholder
#   Acceptance  — subjective or unmeasurable acceptance criteria
#   Dependency  — implicit dependency on external system or team
#   Priority    — missing or implicit priority/urgency signal
#   Technical   — vague non-functional requirement
#
# All seven columns are cast to boolean to ensure consistent behaviour
# in sum() and logical operations used throughout the analysis.
# -----------------------------------------------------------------------

df = pd.read_excel("data/Cornelius_2025_user_story_ambiguity_dataset.xlsx")

ambiguity_columns = [
    "SemanticAmbiguity",
    "ScopeAmbiguity",
    "ActorAmbiguity",
    "AcceptanceAmbiguity",
    "DependencyAmbiguity",
    "PriorityAmbiguity",
    "TechnicalAmbiguity",
]

df[ambiguity_columns] = df[ambiguity_columns].astype(bool)

# --- Data Validation ---
# Before any analysis, verify that the dataset is internally consistent.
# Two checks are performed:
#   Check 1: HasAmbiguity must equal (AmbiguityCount > 0) for every row.
#            A mismatch would mean the aggregate flag contradicts the count.
#   Check 2: AmbiguityCount must equal the row-wise sum of the seven binary
#            columns. A mismatch would mean the count was computed incorrectly.
# If either check fails the script raises an error — all downstream numbers
# depend on the integrity of these columns.

computed_has_ambiguity = (df["AmbiguityCount"] > 0)
mismatch_flag = (computed_has_ambiguity != df["HasAmbiguity"]).sum()

computed_count = df[ambiguity_columns].sum(axis=1)
mismatch_count = (computed_count != df["AmbiguityCount"]).sum()

validation_df = pd.DataFrame({
    "Check": [
        "HasAmbiguity consistent with AmbiguityCount > 0",
        "AmbiguityCount consistent with sum of individual columns",
    ],
    "Mismatches": [mismatch_flag, mismatch_count],
    "Status": [
        "PASS" if mismatch_flag == 0 else "FAIL",
        "PASS" if mismatch_count == 0 else "FAIL",
    ],
})

validation_df.to_csv("results/0_validation.csv", index=False)
print("\n=== 4.1 Dataset Overview ===")
print(validation_df.to_string(index=False))
print(f"Dataset loaded: {len(df):,} user stories, {len(ambiguity_columns)} ambiguity label columns.")

if mismatch_flag > 0 or mismatch_count > 0:
    raise ValueError("Dataset failed validation. Fix before proceeding.")

# -----------------------------------------------------------------------
# 4.2 User story-Level Ambiguity Prevalence
#
# Answers: what proportion of the 12,847 user stories carry at least one
# annotated ambiguity label?
#
# HasAmbiguity is the pre-computed aggregate flag in the dataset (True if
# any of the seven labels is True). Summing a boolean column gives the
# count of True values, which is then expressed as a percentage of total.
# -----------------------------------------------------------------------

total_user_stories        = len(df)
stories_with_ambiguity    = df["HasAmbiguity"].sum()
stories_without_ambiguity = total_user_stories - stories_with_ambiguity

prevalence_df = pd.DataFrame({
    "Category": ["With Ambiguity", "Without Ambiguity"],
    "Count":    [stories_with_ambiguity, stories_without_ambiguity],
})
prevalence_df["Percentage"] = (
    prevalence_df["Count"] / total_user_stories * 100
).round(2)

prevalence_df.to_csv("results/4_2_story_level_prevalence.csv", index=False)

print("\n=== 4.2 User story-Level Ambiguity Prevalence ===")
print(prevalence_df.to_string(index=False))
print(f"Total stories: {total_user_stories:,}")

# -----------------------------------------------------------------------
# 4.3 Distribution of Ambiguity Types (Label-Level)
#
# Answers: across all annotated ambiguity instances, how are they
# distributed across the seven types?
#
# Because a single story can carry multiple labels, the denominator here
# is total_ambiguity_instances (sum of AmbiguityCount across all stories),
# not total_user_stories. This gives the share of each type among all
# individual ambiguity occurrences — a label-level perspective.
#
# The bar chart (Figure 4.3) visualises this distribution.
# -----------------------------------------------------------------------

ambiguity_counts = df[ambiguity_columns].sum().reset_index()
ambiguity_counts.columns = ["Ambiguity Type", "Frequency"]

total_ambiguity_instances = df["AmbiguityCount"].sum()

ambiguity_counts["Percentage"] = (
    ambiguity_counts["Frequency"] / total_ambiguity_instances * 100
).round(2)

ambiguity_counts.to_csv("results/4_3_ambiguity_type_distribution.csv", index=False)

print("\n=== 4.3 Distribution of Ambiguity Types (Label-Level) ===")
print(ambiguity_counts.to_string(index=False))
print(f"Total ambiguity instances across all stories: {int(total_ambiguity_instances):,}")

plt.figure(figsize=(8, 5))
plt.bar(ambiguity_counts["Ambiguity Type"], ambiguity_counts["Frequency"])
plt.xticks(rotation=45, ha="right")
plt.xlabel("Ambiguity Type")
plt.ylabel("Number of Ambiguity Instances")
plt.title("Distribution of Ambiguity Types")
plt.tight_layout()
plt.savefig("figures/4_3_ambiguity_type_distribution.png")
plt.close()

# -----------------------------------------------------------------------
# 4.4 Structural Characteristics of Ambiguous Requirements
#
# 4.4.1 Ambiguity Count Distribution
# Answers: how many ambiguity labels does a typical story carry?
#
# value_counts() on AmbiguityCount produces a frequency table showing
# how many stories have exactly 0, 1, 2, 3, ... labels. Sorting by index
# (not by frequency) preserves the natural count order for the bar chart.
# -----------------------------------------------------------------------

ambiguity_distribution = (
    df["AmbiguityCount"]
    .value_counts()
    .sort_index()
    .reset_index()
)
ambiguity_distribution.columns = ["AmbiguityCount", "Number of Stories"]
ambiguity_distribution.to_csv("results/4_4_ambiguity_count_distribution.csv", index=False)

print("\n=== 4.4.1 Ambiguity Count Distribution ===")
print(ambiguity_distribution.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.bar(ambiguity_distribution["AmbiguityCount"], ambiguity_distribution["Number of Stories"])
plt.xlabel("Number of Ambiguities per Story")
plt.ylabel("Number of Stories")
plt.title("Ambiguity Count Distribution")
plt.tight_layout()
plt.savefig("figures/4_4_ambiguity_count_distribution.png")
plt.close()

# -----------------------------------------------------------------------
# 4.4.2 Composition of Single and Multiple Ambiguity
#
# Splits ambiguous stories into two groups and asks: within each group,
# which ambiguity types appear most often?
#
# Single-ambiguity stories (AmbiguityCount == 1):
#   Exactly one label is True. The frequency of each type equals the
#   number of stories where that type is the sole ambiguity. Percentages
#   are out of n_single (the group size), so they sum to 100%.
#
# Multi-ambiguity stories (AmbiguityCount > 1):
#   Multiple labels may be True per story. The frequency of each type is
#   the count of stories in this group that carry that type. Because one
#   story can contribute to multiple types, percentages do NOT sum to 100%
#   — they express "X% of multi-ambiguity stories contain this type".
# -----------------------------------------------------------------------

single_stories   = df[df["AmbiguityCount"] == 1]
multiple_stories = df[df["AmbiguityCount"] > 1]

n_single   = len(single_stories)
n_multiple = len(multiple_stories)

single_composition = pd.DataFrame({
    "Ambiguity Type": ambiguity_columns,
    "Frequency":      single_stories[ambiguity_columns].sum().values,
})
single_composition["Percentage"] = (
    single_composition["Frequency"] / n_single * 100
).round(2)
single_composition["Ambiguity Type"] = (
    single_composition["Ambiguity Type"].str.replace("Ambiguity", "")
)
single_composition.to_csv("results/4_4_2_single_ambiguity_composition.csv", index=False)

multiple_composition = pd.DataFrame({
    "Ambiguity Type": ambiguity_columns,
    "Frequency":      multiple_stories[ambiguity_columns].sum().values,
})
multiple_composition["Percentage"] = (
    multiple_composition["Frequency"] / n_multiple * 100
).round(2)
multiple_composition["Ambiguity Type"] = (
    multiple_composition["Ambiguity Type"].str.replace("Ambiguity", "")
)
multiple_composition.to_csv("results/4_4_2_multiple_ambiguity_composition.csv", index=False)

print(f"\n=== 4.4.2 Single-Ambiguity Story Composition (n={n_single}) ===")
print(single_composition.to_string(index=False))
print(f"\n=== 4.4.2 Multi-Ambiguity Story Composition (n={n_multiple}) ===")
print("(Percentages show story-presence rate; do not sum to 100%)")
print(multiple_composition.to_string(index=False))

# -----------------------------------------------------------------------
# 4.4.3 Co-occurrence Patterns Between Ambiguity Types
#
# Answers: when two ambiguity types appear in the same story, which pairs
# co-occur most frequently?
#
# Three outputs are produced:
#
# (a) Full symmetric matrix — cell [A, B] = number of stories where both
#     A and B are True. The diagonal is zero (a type cannot co-occur with
#     itself). Saved as 4_4_3_cooccurrence_matrix.csv.
#
# (b) Upper-triangle matrix — lower triangle and diagonal replaced with
#     "—" to match the presentation format of Table 5 in the thesis.
#     Saved as 4_4_3_cooccurrence_upper_triangle.csv.
#
# (c) Ranked pairs — all 21 unique pairs sorted by co-occurrence count,
#     useful for identifying the most common combinations.
#     Saved as 4_4_3_cooccurrence_ranked_pairs.csv.
#
# (d) Normalised co-occurrence rates — cell [A, B] = P(B present | A
#     present) = count(A AND B) / count(A). The diagonal is 1.0 because
#     P(A present | A present) = 1, and rows with no occurrences stay 0.
#     Answers "given that type A is present, how likely is type B to also
#     appear?"
#     Saved as 4_4_3_cooccurrence_rate.csv.
# -----------------------------------------------------------------------

cooccurrence = pd.DataFrame(0, index=ambiguity_columns, columns=ambiguity_columns)

for col_a, col_b in combinations(ambiguity_columns, 2):
    count = ((df[col_a]) & (df[col_b])).sum()
    cooccurrence.loc[col_a, col_b] = count
    cooccurrence.loc[col_b, col_a] = count  # matrix is symmetric

short_labels         = [col.replace("Ambiguity", "") for col in ambiguity_columns]
cooccurrence.index   = short_labels
cooccurrence.columns = short_labels

cooccurrence.to_csv("results/4_4_3_cooccurrence_matrix.csv")

# Upper triangle — lower triangle and diagonal replaced with "—"
upper = cooccurrence.copy().astype(object)
for i in range(len(short_labels)):
    upper.iloc[i, i] = "—"
    for j in range(i):
        upper.iloc[i, j] = "—"

upper.to_csv("results/4_4_3_cooccurrence_upper_triangle.csv")

print("\n=== 4.4.3 Co-occurrence Matrix (upper triangle) ===")
print(upper.to_string())

# Ranked pairs — all unique type combinations sorted by frequency
pairs = []
for col_a, col_b in combinations(short_labels, 2):
    pairs.append({
        "Type A":              col_a,
        "Type B":              col_b,
        "Co-occurrence Count": cooccurrence.loc[col_a, col_b],
    })

pairs_df = pd.DataFrame(pairs).sort_values(
    "Co-occurrence Count", ascending=False
).reset_index(drop=True)
pairs_df.to_csv("results/4_4_3_cooccurrence_ranked_pairs.csv", index=False)

print("\n=== 4.4.3 Top Co-occurring Pairs ===")
print(pairs_df.head(10).to_string(index=False))

# Normalised rates: P(Type B present | Type A present)
# Row = Type A (the given condition), Column = Type B (the observed outcome)
cooccurrence_rate = pd.DataFrame(0.0, index=short_labels, columns=short_labels)

for col_a, label_a in zip(ambiguity_columns, short_labels):
    total_a = df[col_a].sum()  # total stories where Type A is present
    if total_a == 0:
        continue
    cooccurrence_rate.loc[label_a, label_a] = 1.0
    for col_b, label_b in zip(ambiguity_columns, short_labels):
        if col_a != col_b:
            both = ((df[col_a]) & (df[col_b])).sum()
            cooccurrence_rate.loc[label_a, label_b] = round(both / total_a, 4)

cooccurrence_rate.to_csv("results/4_4_3_cooccurrence_rate.csv")

print("\n=== 4.4.3 Normalised Co-occurrence Rates — P(B present | A present) ===")
print(cooccurrence_rate.to_string())

# -----------------------------------------------------------------------
# 4.5 Context-dependent and Structural Layering
#
# Classifies the seven ambiguity types into two tiers based on the degree
# to which resolving them requires information beyond the story text itself:
#
#   High-context types (5): Actor, Acceptance, Dependency, Priority, Technical
#     These require knowledge of the system context, team agreements, or
#     external constraints to resolve — they cannot be fixed by rewriting
#     the story alone.
#
#   Lower-context types (2): Semantic, Scope
#     These can often be resolved through clearer wording within the story.
#
# Two derived columns are added to df:
#   HasHighContext   — True if the story carries any high-context label
#   HighContextCount — count of high-context labels present in the story
#
# Prevalence metrics answer: how common are high-context ambiguities
# relative to all stories and to ambiguous stories specifically?
# -----------------------------------------------------------------------

high_context_cols = [
    "ActorAmbiguity",
    "AcceptanceAmbiguity",
    "DependencyAmbiguity",
    "PriorityAmbiguity",
    "TechnicalAmbiguity",
]

df["HasHighContext"]   = df[high_context_cols].any(axis=1)
df["HighContextCount"] = df[high_context_cols].sum(axis=1)

high_context_stories = df["HasHighContext"].sum()

high_context_prevalence_df = pd.DataFrame({
    "Metric": [
        "High-context stories (count)",
        "High-context stories (% of total)",
        "High-context stories (% of ambiguous)",
    ],
    "Value": [
        high_context_stories,
        round(high_context_stories / total_user_stories * 100, 2),
        round(high_context_stories / stories_with_ambiguity * 100, 2),
    ],
})
high_context_prevalence_df.to_csv("results/4_5_high_context_prevalence.csv", index=False)

print("\n=== 4.5.1 Degree of Context-dependent Grouping ===")
print(high_context_prevalence_df.to_string(index=False))

# Conditional probabilities between Semantic and High-context presence.
# Two directions are computed to capture both perspectives:
#   P(Semantic | HighContext) — of stories that are high-context, what
#     fraction also carry Semantic ambiguity? Indicates co-occurrence rate.
#   P(HighContext | Semantic) — of stories with Semantic ambiguity, what
#     fraction also carry a high-context type? Indicates structural layering.
both_semantic_high = df[
    (df["SemanticAmbiguity"]) & (df["HasHighContext"])
].shape[0]

conditional_df = pd.DataFrame({
    "Metric": [
        "P(Semantic | HighContext)",
        "P(HighContext | Semantic)",
    ],
    "Value": [
        round(both_semantic_high / high_context_stories, 4),
        round(both_semantic_high / df["SemanticAmbiguity"].sum(), 4),
    ],
})
conditional_df.to_csv("results/4_5_conditional_probabilities.csv", index=False)

print("\n=== 4.5.2 Conditional Probabilities ===")
print(conditional_df.to_string(index=False))

# -----------------------------------------------------------------------
# 4.5.3 Structural Layering of High-tier Ambiguity
#
# Answers: is ambiguity concentrated in a small number of stories, or is
# it evenly spread? A Pareto-style analysis identifies whether the "top
# 20% most ambiguous stories" account for a disproportionate share of
# total ambiguity instances (the 80/20 principle).
#
# Stories are ranked by AmbiguityCount descending. The top 20% block
# (top_20) is then used to compute:
#   - what % of total ambiguity instances it contains
#   - what % of total high-context instances it contains
#
# hc_single / hc_multi additionally show whether high-context ambiguity
# tends to appear alongside other types (multi) or in isolation (single).
# -----------------------------------------------------------------------

sorted_df    = df.sort_values(by="AmbiguityCount", ascending=False)
top_20_count = int(len(df) * 0.20)
top_20       = sorted_df.head(top_20_count)

ambiguity_in_top_20    = top_20["AmbiguityCount"].sum()
high_context_total     = df["HighContextCount"].sum()
high_context_in_top_20 = top_20["HighContextCount"].sum()

hc_single = df[(df["AmbiguityCount"] == 1) & (df["HasHighContext"])].shape[0]
hc_multi  = df[(df["AmbiguityCount"] > 1)  & (df["HasHighContext"])].shape[0]

concentration_df = pd.DataFrame({
    "Metric": [
        "Total ambiguity in top 20% by AmbiguityCount (%)",
        "High-context ambiguity in top 20% by AmbiguityCount (%)",
        "High-context stories with single ambiguity (count)",
        "High-context stories with multiple ambiguities (count)",
        "High-context stories with multiple ambiguities (% of HC stories)",
    ],
    "Value": [
        round(ambiguity_in_top_20 / total_ambiguity_instances * 100, 2),
        round(high_context_in_top_20 / high_context_total * 100, 2),
        hc_single,
        hc_multi,
        round(hc_multi / (hc_single + hc_multi) * 100, 2),
    ],
})
concentration_df.to_csv("results/4_5_concentration.csv", index=False)

print("\n=== 4.5.3 Structural Layering of High-tier Ambiguity ===")
print(f"Top 20% stories (n={top_20_count:,}) sorted by AmbiguityCount:")
print(concentration_df.to_string(index=False))

# -----------------------------------------------------------------------
# 4.5.3 (continued) — Gini Coefficient & Lorenz Curve
#
# The Gini coefficient measures inequality in the distribution of ambiguity
# across stories. A value of 0 means every story has exactly the same
# AmbiguityCount (perfect equality); a value of 1 means all ambiguity is
# concentrated in a single story (maximum inequality).
#
# Two variants are reported:
#   (a) All stories including zeros — captures the full dataset inequality,
#       where most stories have no ambiguity at all.
#   (b) Ambiguous stories only — isolates inequality among stories that do
#       carry ambiguity, showing how unevenly it is spread within that group.
#
# The Lorenz curve (Figure 4.5) provides the visual counterpart: the x-axis
# is the cumulative share of stories (sorted by count ascending) and the
# y-axis is the cumulative share of total ambiguity instances. The gap
# between the curve and the 45° equality line visualises the Gini value.
# -----------------------------------------------------------------------

def gini(array):
    """Compute the Gini coefficient using the standard rank-weighted formula."""
    # Keep the thesis calculation reproducible by using the same standard form.
    array = np.array(array, dtype=float)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = np.sort(array)
    n     = array.shape[0]
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


n_all       = total_user_stories
n_ambiguous = int(stories_with_ambiguity)

gini_all       = gini(df["AmbiguityCount"])
gini_ambiguous = gini(df[df["AmbiguityCount"] > 0]["AmbiguityCount"])

gini_df = pd.DataFrame({
    "Metric": [
        f"Gini (all {n_all:,} stories, incl. zeros)",
        f"Gini ({n_ambiguous:,} ambiguous stories only)",
    ],
    "Value": [round(gini_all, 4), round(gini_ambiguous, 4)],
})
gini_df.to_csv("results/4_5_gini.csv", index=False)

print("\n=== 4.5.3 Gini Coefficient & Lorenz Curve ===")
print(gini_df.to_string(index=False))

# Lorenz curve data: sort stories by AmbiguityCount ascending, then compute
# the running share of total ambiguity and the running share of stories.
sorted_counts      = np.sort(df["AmbiguityCount"])
cumulative_counts  = np.cumsum(sorted_counts) / np.sum(sorted_counts)
cumulative_stories = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)

plt.figure(figsize=(6, 6))
plt.plot(cumulative_stories, cumulative_counts, label="Lorenz Curve")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Equality")
plt.xlabel("Cumulative Share of Stories")
plt.ylabel("Cumulative Share of Ambiguity")
plt.title("Lorenz Curve of Ambiguity Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("figures/4_5_lorenz_curve_ambiguity.png")
plt.close()

# -----------------------------------------------------------------------
# 4.5.4 Context-dependent Across Ambiguity Counts
#
# Produces one row per exact AmbiguityCount value (0, 1, 2, 3, …) showing
# how stories are distributed across density levels and what proportion of
# each level carries high-context ambiguity.
#
# This allows the thesis to discuss, for example, whether stories with
# higher ambiguity density are also more likely to involve high-context
# types — which would suggest compounding contextual risk.
#
# The dominant-type loop identifies, for each count level, which ambiguity
# type appears in the highest percentage of stories at that level.
# -----------------------------------------------------------------------

density_rows = []
for count in sorted(df["AmbiguityCount"].unique()):
    group       = df[df["AmbiguityCount"] == count]
    n_stories   = len(group)
    n_instances = int(group["AmbiguityCount"].sum())
    n_hc        = int(group["HasHighContext"].sum())

    density_rows.append({
        "AmbiguityCount":            count,
        "Stories (n)":               n_stories,
        "Stories (% of all)":        round(n_stories / total_user_stories * 100, 2),
        "Ambiguity instances":        n_instances,
        "Instances (% of total)":    round(n_instances / total_ambiguity_instances * 100, 2),
        "Stories with high-context": n_hc,
        "High-context (% of group)": round(n_hc / n_stories * 100, 2) if n_stories > 0 else 0,
    })

density_df = pd.DataFrame(density_rows)
density_df.to_csv("results/4_5_4_density_segmentation.csv", index=False)

print("\n=== 4.5.4 Context-dependent Across Ambiguity Counts ===")
print(density_df.to_string(index=False))

print("\nDominant ambiguity type per AmbiguityCount level:")
for count in sorted(df[df["AmbiguityCount"] > 0]["AmbiguityCount"].unique()):
    group        = df[df["AmbiguityCount"] == count]
    type_counts  = group[ambiguity_columns].sum()
    dominant     = type_counts.idxmax().replace("Ambiguity", "")
    dominant_pct = round(type_counts.max() / len(group) * 100, 2)
    print(f"  Count={count}: {dominant} ({dominant_pct}% of stories in this group)")

# -----------------------------------------------------------------------
# Master Summary — all Chapter 4 headline numbers in one table
#
# Consolidates every key metric reported in Chapter 4 into a single CSV
# (0_master_summary.csv) for easy cross-referencing and thesis writing.
# Numbers are grouped by the chapter section they support.
# -----------------------------------------------------------------------

ambiguous_stories = int(df["HasAmbiguity"].sum())
total_stories     = total_user_stories
n_single_sum      = int((df["AmbiguityCount"] == 1).sum())
n_multi_sum       = int((df["AmbiguityCount"] > 1).sum())
hc_stories        = int(df["HasHighContext"].sum())
hc_single_sum     = int(df[(df["AmbiguityCount"] == 1) & df["HasHighContext"]].shape[0])
hc_multi_stories  = int(df[(df["AmbiguityCount"] > 1)  & df["HasHighContext"]].shape[0])

summary = pd.DataFrame([
    # 4.2 User story-Level Ambiguity Prevalence
    ("Total user stories",                      total_stories),
    ("Ambiguous stories (n)",                    ambiguous_stories),
    ("Ambiguous stories (%)",                    round(ambiguous_stories / total_stories * 100, 2)),
    ("Non-ambiguous stories (%)",                round((total_stories - ambiguous_stories) / total_stories * 100, 2)),
    # 4.3 Distribution of Ambiguity Types (Label-Level)
    ("Total ambiguity instances",                int(total_ambiguity_instances)),
    ("Semantic instances (%)",                   round(df["SemanticAmbiguity"].sum() / total_ambiguity_instances * 100, 2)),
    ("Scope instances (%)",                      round(df["ScopeAmbiguity"].sum() / total_ambiguity_instances * 100, 2)),
    ("Actor instances (%)",                      round(df["ActorAmbiguity"].sum() / total_ambiguity_instances * 100, 2)),
    ("High-context instances (%)",               round(df[high_context_cols].sum().sum() / total_ambiguity_instances * 100, 2)),
    # 4.4.2 Composition of Single and Multiple Ambiguity
    ("Single-ambiguity stories (%)",             round(n_single_sum / ambiguous_stories * 100, 2)),
    ("Multi-ambiguity stories (%)",              round(n_multi_sum  / ambiguous_stories * 100, 2)),
    # 4.5.1 Degree of Context-dependent Grouping
    ("HC stories (% of total)",                 round(hc_stories / total_stories * 100, 2)),
    ("HC stories (% of ambiguous)",             round(hc_stories / ambiguous_stories * 100, 2)),
    ("HC stories with single ambiguity (n)",     hc_single_sum),
    ("HC stories with multiple ambiguities (n)", hc_multi_stories),
    ("HC stories in multi-ambiguity (%)",        round(hc_multi_stories / hc_stories * 100, 2)),
    # 4.5.3 Structural Layering of High-tier Ambiguity
    ("Top 20% stories (n)",                      top_20_count),
    ("Ambiguity in top 20% (%)",                 round(top_20["AmbiguityCount"].sum() / total_ambiguity_instances * 100, 2)),
    ("HC ambiguity in top 20% by AmbiguityCount (%)",
                                                 round(top_20["HighContextCount"].sum() / df["HighContextCount"].sum() * 100, 2)),
    # 4.5.3 Gini Coefficient & Lorenz Curve
    ("Gini — all stories",                       round(gini_all, 4)),
    ("Gini — ambiguous stories only",            round(gini_ambiguous, 4)),
], columns=["Metric", "Value"])

summary.to_csv("results/0_master_summary.csv", index=False)

print("\n=== MASTER SUMMARY — All Chapter 4 Headline Numbers ===")
print(summary.to_string(index=False))
print("\nAppendix A complete.")

# =============================================================================
# APPENDIX B — Surface Marker Validation Test
# =============================================================================
#
# Purpose: test whether lightweight regex-based surface markers can identify
# high-context ambiguity types (Actor, Acceptance, Dependency, Priority,
# Technical) without requiring full semantic analysis. This informs the
# discussion of automated detection limitations in Chapter 5.
#
# Scientific design: to avoid circularity (markers defined on the same data
# they are tested on), the ambiguous stories are split 80/20. Markers are
# characterised on the definition group (80%) and evaluated only on the
# independent test group (20%) that was never inspected during rule design.

# -----------------------------------------------------------------------
# B.1 Marker Definitions
# Each key maps to a regex pattern targeting linguistic signals associated
# with one ambiguity type. All matches are case-insensitive.
#
#   vague_verbs  — broad action verbs linked to SemanticAmbiguity
#   scope        — scope-expanding words linked to ScopeAmbiguity
#   actor        — generic "As a user/role" phrasing linked to ActorAmbiguity
#   acceptance   — subjective quality terms linked to AcceptanceAmbiguity
#   dependency   — integration/connection verbs linked to DependencyAmbiguity
#   priority     — urgency language linked to PriorityAmbiguity
#   technical    — non-functional requirement terms linked to TechnicalAmbiguity
# -----------------------------------------------------------------------

markers = {
    "vague_verbs":  r"\b(manage|handle|process|maintain|deal with|"
                    r"take care|ensure|facilitate|support|address|"
                    r"utilize|leverage|implement|optimize|enhance)\b",
    "scope":        r"\b(system|all|any|every|various|multiple|everything|"
                    r"overall|comprehensive|entire|general|whole)\b",
    "actor":        r"^As a user,",
    "acceptance":   r"\b(better|faster|improved|enhanced|efficient|"
                    r"effective|good|great|smooth|optimal|adequate)\b",
    "dependency":   r"\b(integrate|connect|link|sync|interface|"
                    r"communicate|interact|data flows)\b",
    "priority":     r"\b(as soon as possible|asap|urgent|immediately|"
                    r"priority|critical|important)\b",
    "technical":    r"\b(fast performance|high performance|scalable|"
                    r"reliable|secure|robust|responsive)\b",
}

# The five ambiguity label columns that constitute "high-context" ambiguity.
# A story is classified as high-context if it carries at least one of these.
high_labels = [
    "ActorAmbiguity",
    "AcceptanceAmbiguity",
    "DependencyAmbiguity",
    "PriorityAmbiguity",
    "TechnicalAmbiguity",
]

# Restrict to ambiguous stories only — the marker test targets the subset
# of stories that already carry at least one annotated ambiguity label.
amb = df[df["HasAmbiguity"] == True].copy().reset_index(drop=True)

# Ground-truth label: True if the story has any high-context ambiguity type
amb["actual_high"] = amb[high_labels].any(axis=1)

n_amb      = len(amb)
n_hc_total = amb["actual_high"].sum()
print(f"\n--- Appendix B: Surface Marker Validation ---")
print(f"Ambiguous stories in scope : {n_amb}")
print(f"Of which high-context      : {n_hc_total} ({round(n_hc_total / n_amb * 100, 1)}%)")

# -----------------------------------------------------------------------
# B.2 80/20 Held-Out Split
#
# IMPORTANT — this split uses random sampling (pandas DataFrame.sample),
# NOT a sequential slice of the first 80% of rows. The random shuffle
# means the group composition will differ from any split that takes rows
# in dataset order (e.g. head/tail or positional slicing).
#
# To reproduce this exact split, random_state=42 must be used with the
# same version of pandas and the same input DataFrame row order.
# Any change to filtering, sorting, or pandas version will alter the split.
#
# The index is saved once from the first sample() call and reused for
# both halves to guarantee they are complementary and non-overlapping.
# -----------------------------------------------------------------------

SEED             = 42
_def_idx         = amb.sample(frac=0.80, random_state=SEED).index
definition_group = amb.loc[_def_idx].reset_index(drop=True)
test_group       = amb.drop(_def_idx).reset_index(drop=True)

# Row counts are derived directly from the split — NOT from marker results.
# TP + FN for each group will always equal these high-context counts,
# which confirms the marker evaluation is internally consistent.
print(f"\nDefinition group (80%) : n={len(definition_group)}")
print(f"Test group       (20%) : n={len(test_group)}")
print(f"  High-context stories in definition group (row count) : {definition_group['actual_high'].sum()}")
print(f"  High-context stories in test group       (row count) : {test_group['actual_high'].sum()}")
print(f"  Note: differs from a sequential 80/20 slice because sample() shuffles randomly (seed={SEED}).")

# -----------------------------------------------------------------------
# B.3 Definition Group — Per-Marker Coverage
#
# For each marker pattern, compute:
#   - how many stories in the definition group it matches
#   - what proportion of those matches are actually high-context
#     (individual marker precision)
#
# This table justifies why each marker was retained in the final rule set.
# A marker with very low precision would indicate it fires on many
# non-high-context stories and may not be diagnostic.
# -----------------------------------------------------------------------

coverage_rows = []
for name, pattern in markers.items():
    matched     = definition_group["StoryText"].str.contains(pattern, case=False, regex=True)
    n_matched   = matched.sum()
    n_hc        = (matched & definition_group["actual_high"]).sum()
    precision_m = n_hc / n_matched if n_matched > 0 else 0

    coverage_rows.append({
        "Marker":                   name,
        "Stories matched (n)":      int(n_matched),
        "Matched (% of def group)": round(n_matched / len(definition_group) * 100, 2),
        "High-context matches (n)": int(n_hc),
        "Marker precision (%)":     round(precision_m * 100, 2),
    })

coverage_df = pd.DataFrame(coverage_rows)
coverage_df.to_csv("results/surface_marker_definition_coverage.csv", index=False)

print("\n=== B.3 Definition Group — Per-Marker Coverage (80%) ===")
print(coverage_df.to_string(index=False))

# -----------------------------------------------------------------------
# B.4 Combined Precision / Recall
#
# Only the five HIGH-CONTEXT markers (actor, acceptance, dependency,
# priority, technical) are used for prediction — vague_verbs and scope
# are associated with Semantic/Scope ambiguity, not high-context types.
#
# A story is flagged as high-context if ANY high-context marker fires.
# Metrics are computed on:
#   (a) the full ambiguous dataset  — baseline / reference
#   (b) the independent test group  — unbiased generalisation estimate
#
# Definitions:
#   TP — story flagged by markers AND actually high-context
#   FP — story flagged by markers but NOT high-context
#   FN — story NOT flagged by markers but IS high-context (missed)
#   Precision = TP / (TP + FP)  → of all flagged, how many were correct
#   Recall    = TP / (TP + FN)  → of all high-context, how many were found
#   F1        = harmonic mean of precision and recall
# -----------------------------------------------------------------------

def precision_recall(subset):
    """Flag stories using combined high-context markers; return TP/FP/FN/P/R/F1."""
    has_high_marker = (
        subset["StoryText"].str.contains(markers["actor"],       case=False, regex=True) |
        subset["StoryText"].str.contains(markers["acceptance"],  case=False, regex=True) |
        subset["StoryText"].str.contains(markers["dependency"],  case=False, regex=True) |
        subset["StoryText"].str.contains(markers["priority"],    case=False, regex=True) |
        subset["StoryText"].str.contains(markers["technical"],   case=False, regex=True)
    )
    tp = (has_high_marker &  subset["actual_high"]).sum()
    fp = (has_high_marker & ~subset["actual_high"]).sum()
    fn = (~has_high_marker & subset["actual_high"]).sum()
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return int(tp), int(fp), int(fn), round(p, 4), round(r, 4), round(f1, 4)


# Definition group (80%) is the baseline — markers were characterised here.
# Test group (20%) was never seen during marker design — gives unbiased result.
# Comparing the two tells us whether precision holds on truly unseen stories.
def_tp,  def_fp,  def_fn,  def_p,  def_r,  def_f1  = precision_recall(definition_group)
test_tp, test_fp, test_fn, test_p, test_r, test_f1 = precision_recall(test_group)

comparison_df = pd.DataFrame([
    {"Metric": "True Positives",  "Definition group (80%)": def_tp,  "Test group (20%)": test_tp},
    {"Metric": "False Positives", "Definition group (80%)": def_fp,  "Test group (20%)": test_fp},
    {"Metric": "False Negatives", "Definition group (80%)": def_fn,  "Test group (20%)": test_fn},
    {"Metric": "Precision",       "Definition group (80%)": def_p,   "Test group (20%)": test_p},
    {"Metric": "Recall",          "Definition group (80%)": def_r,   "Test group (20%)": test_r},
    {"Metric": "F1 Score",        "Definition group (80%)": def_f1,  "Test group (20%)": test_f1},
])

comparison_df.to_csv("results/surface_marker_precision_recall.csv", index=False)

print("\n=== B.4 Precision / Recall — Definition Group (80%) vs Independent Test Group (20%) ===")
print(comparison_df.to_string(index=False))

# -----------------------------------------------------------------------
# B.5 Generalisation Verdict
# Compare precision on the definition group against the held-out test group.
# The definition group is the correct baseline because the markers were
# designed by inspecting only those stories. The test group was completely
# isolated, so any drop in precision there reflects true generalisation loss
# rather than overfitting to the marker-design population.
# A drop of more than 5 percentage points warrants marker refinement.
# -----------------------------------------------------------------------

precision_diff = abs(def_p - test_p)
print(f"\n--- Generalisation Check ---")
print(f"Definition group precision : {def_p:.2%}")
print(f"Test group precision       : {test_p:.2%}")
print(f"Difference                 : {precision_diff:.2%}")

if precision_diff <= 0.05:
    print("=> PASS: precision difference <=5pp — markers generalise well to unseen stories.")
else:
    print("=> REVIEW: precision dropped >5pp on test group — markers may need refinement.")

print("\nAppendix B complete.")
