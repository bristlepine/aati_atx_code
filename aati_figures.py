# aati_figures.py
# .\venv\Scripts\Activate

"""
Author: zk
Date Updated: 10 June 2025

Main script for codifying and visualizing AATI SWOT data.

Workflow:
1. Load raw matrix from CSV
2. Codify SWOT fields (split entries into rows, clean text)
3. Optionally use NLP (spaCy or Gemini) to enrich codification
4. Save cleaned/codified data to CSV
5. Generate visualizations:
   - Bar charts
   - Word clouds
   - Radar charts
   - Heatmaps
"""

# ─────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
import spacy

# Optional: Gemini integration (if enabled)
# from your_gemini_module import query_gemini  # Replace with real implementation

# ─────────────────────────────────────────────────────────────
# 2. Configuration
# ─────────────────────────────────────────────────────────────

RAW_CSV = "Desk Research Matrix - Combined Matrix.csv"
CODIFIED_CSV = "codified_matrix.csv"
OUTPUT_DIR = "outputs"

USE_GEMINI = False  # Set to True if using Gemini API

# ─────────────────────────────────────────────────────────────
# 3. Load Raw Data
# ─────────────────────────────────────────────────────────────

def load_raw_matrix(path):
    """Load the raw SWOT matrix from CSV."""
    return pd.read_csv(path)

# ─────────────────────────────────────────────────────────────
# 4. Codify Matrix
# ─────────────────────────────────────────────────────────────

def codify_swot(df):
    """
    Split multiline SWOT fields into one row per item,
    tagged with Source and Dimension.
    """
    codified_rows = []

    for _, row in df.iterrows():
        for category in ["Strengths", "Weakness", "Opportunity", "Threat"]:
            items = str(row[category]).split("\n")
            for item in items:
                if item.strip():
                    codified_rows.append({
                        "Source": row["Source"],
                        "Dimension": row["Dimension"],
                        "SWOT": category,
                        "Text": item.strip()
                    })

    return pd.DataFrame(codified_rows)

# ─────────────────────────────────────────────────────────────
# 5. Optional NLP Enhancement (spaCy or Gemini)
# ─────────────────────────────────────────────────────────────

def enrich_text_nlp(text):
    """Placeholder for NLP enrichment (entity tagging, summarization, etc.)."""
    # Example: return query_gemini(prompt=text) if USE_GEMINI else text
    return text  # For now, return unchanged

def apply_nlp_enrichment(df):
    """Apply NLP enrichment to codified text (optional)."""
    df["Text_Enhanced"] = df["Text"].apply(enrich_text_nlp)
    return df

# ─────────────────────────────────────────────────────────────
# 6. Save Codified Data
# ─────────────────────────────────────────────────────────────

def save_codified_matrix(df, path):
    df.to_csv(path, index=False)

# ─────────────────────────────────────────────────────────────
# 7. Generate Visualizations
# ─────────────────────────────────────────────────────────────

def plot_bar_chart(df):
    """Bar chart: count of SWOT items by Dimension and Category."""
    pivot = df.groupby(["Dimension", "SWOT"]).size().unstack().fillna(0)
    pivot.plot(kind="bar", stacked=True)
    plt.title("SWOT Items by Dimension")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/strengths_by_dimension_bar.png")
    plt.close()

def generate_wordcloud(df, swot_type):
    """Generate and save word cloud for a given SWOT type."""
    text = " ".join(df[df["SWOT"] == swot_type]["Text"])
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    wc.to_file(f"{OUTPUT_DIR}/{swot_type.lower()}_wordcloud.png")

# Additional visualization stubs:
# - plot_radar_chart()
# - plot_heatmap()

# ─────────────────────────────────────────────────────────────
# 8. Main Execution
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading raw matrix...")
    raw_df = load_raw_matrix(RAW_CSV)

    print("Codifying SWOT entries...")
    codified_df = codify_swot(raw_df)

    if USE_GEMINI:
        print("Enriching text via NLP...")
        codified_df = apply_nlp_enrichment(codified_df)

    print("Saving codified matrix...")
    save_codified_matrix(codified_df, CODIFIED_CSV)

    print("Generating visualizations...")
    plot_bar_chart(codified_df)
    for swot in ["Strengths", "Weakness", "Opportunity", "Threat"]:
        generate_wordcloud(codified_df, swot)

    print("Done! Visualizations saved to /outputs.")

if __name__ == "__main__":
    # main()
    print("Hello")