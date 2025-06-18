# aati_figures.py (Updated Version)

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import networkx as nx
import textwrap
from collections import Counter
from io import StringIO
import re
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration and Setup ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable.")
genai.configure(api_key=API_KEY)

RAW_CSV = "NotebookLM_Combined_Matrix.csv"
OUTPUT_DIR = "outputs"

# --- Core Data Functions ---
def load_raw_matrix(path):
    return pd.read_csv(path)

def extract_bracketed_codes(text):
    return re.findall(r"\[(.*?)\]", text or "")

def codify_swot(df):
    codified_rows = []
    for _, row in df.iterrows():
        for category in ["Strengths", "Weakness", "Opportunity", "Threat"]:
            items = str(row.get(category, "")).split("\n")
            for item in items:
                if item.strip():
                    codified_rows.append({
                        "Source": row["Source"],
                        "Dimension": row["Dimension"],
                        "SWOT": category,
                        "Text": item.strip(),
                        "DocSources": row.get("DocSources", "")
                    })
    return pd.DataFrame(codified_rows)

# --- NLP-Powered Code Consolidation ---
def consolidate_codes_with_gemini(codebook_df, save_path=None):
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    codes_to_process = codebook_df.sort_values(by="Count", ascending=False)['Code'].tolist()
    prompt = f"""
You are a data analyst. Your task is to group the following list of codes based on semantic similarity.
For each group of similar codes, list them together on a single line, separated by a pipe character (|).
If a code is unique and has no synonyms in the list, list it by itself on its own line.
EXAMPLE:
If the input list is: ['Lack of funds', 'Funding issues', 'Clear Mandate', 'Political Risk']
The correct output would be:
Lack of funds|Funding issues
Clear Mandate
Political Risk
---
Here is the list of codes to process:
{chr(10).join(codes_to_process)}
"""
    print("Asking Gemini to group similar codes... (This may take a moment)")
    response = model.generate_content(prompt)
    print("Processing AI groups and assigning canonical names...")
    code_counts = codebook_df.set_index('Code')['Count'].to_dict()
    final_mapping = {}
    grouped_codes_text = response.text.strip()
    for line in grouped_codes_text.splitlines():
        synonyms = [code.strip() for code in line.split('|')]
        if not synonyms: continue
        canonical_name = max(synonyms, key=lambda code: code_counts.get(code, 0))
        for synonym in synonyms:
            final_mapping[synonym] = canonical_name
    for code in codes_to_process:
        if code not in final_mapping:
            final_mapping[code] = code
    mapping_df = pd.DataFrame(list(final_mapping.items()), columns=['Original_Code', 'Consolidated_Code'])
    if save_path:
        mapping_df.sort_values(by=['Consolidated_Code', 'Original_Code']).to_csv(save_path, index=False)
        print(f"✓ Saved final, verified code consolidation map to {save_path}")
    return mapping_df

# --- Data Processing Helpers ---
def report_code_changes(consolidation_map_df, save_path=None):
    changed_codes_df = consolidation_map_df[consolidation_map_df['Original_Code'] != consolidation_map_df['Consolidated_Code']]
    if changed_codes_df.empty:
        print("✓ No codes were consolidated. All original codes were unique.")
        return None
    target_consolidated_codes = changed_codes_df['Consolidated_Code'].unique()
    final_report_df = consolidation_map_df[consolidation_map_df['Consolidated_Code'].isin(target_consolidated_codes)].sort_values(by=['Consolidated_Code', 'Original_Code'])
    if save_path:
        final_report_df.to_csv(save_path, index=False)
        print(f"✓ Report of consolidated groups saved to: {os.path.basename(save_path)}")
    return final_report_df

def apply_consolidation_map(codified_df, consolidation_map_df):
    consolidation_map_df = consolidation_map_df.rename(columns={'Code': 'Original_Code'}, errors='ignore')
    final_df = codified_df.merge(consolidation_map_df, left_on='Code', right_on='Original_Code', how='left')
    final_df['Code'] = final_df['Consolidated_Code'].fillna(final_df['Code'])
    return final_df[['Source', 'Dimension', 'SWOT', 'Text', 'Code','DocSources']]

# --- Visualization Functions ---
def plot_swot_quadrant_chart(df_dimension, dimension_name, filename):
    """
    Generates an improved SWOT quadrant chart with better title placement
    to avoid overlapping the central axes.
    """
    print(f"Generating enhanced SWOT quadrant chart for: {dimension_name}")

    # --- Helper function to make colormaps darker ---
    def truncate_colormap(cmap_name, minval=0.25, maxval=0.95, n=100):
        cmap = plt.get_cmap(cmap_name)
        return LinearSegmentedColormap.from_list(f'trunc({cmap_name})', cmap(np.linspace(minval, maxval, n)))

    # 1. Prepare frequency dictionaries
    def prepare_frequencies(codes_series):
        if codes_series.empty: return {}
        return Counter([code.replace('[', '').replace(']', '').strip() for code in codes_series])
    strengths_freqs = prepare_frequencies(df_dimension[df_dimension['SWOT'] == 'Strengths']['Code'].dropna())
    weaknesses_freqs = prepare_frequencies(df_dimension[df_dimension['SWOT'] == 'Weakness']['Code'].dropna())
    opportunities_freqs = prepare_frequencies(df_dimension[df_dimension['SWOT'] == 'Opportunity']['Code'].dropna())
    threats_freqs = prepare_frequencies(df_dimension[df_dimension['SWOT'] == 'Threat']['Code'].dropna())

    # 2. Set up the 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle(f"SWOT Analysis for Dimension: {dimension_name}", fontsize=26, y=0.98, fontweight='bold')

    # 3. Helper function to generate and plot a word cloud
    def plot_wc_in_quadrant(ax, frequencies, base_colormap):
        ax.axis('off') # Turn off all decorations on the subplot axis
        if not frequencies:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, fontsize=16, alpha=0.5)
            return
        padding = 0.05
        inset_ax = ax.inset_axes([padding, padding, 1 - padding*2, 1 - padding*2])
        darker_cmap = truncate_colormap(base_colormap)
        wc = WordCloud(background_color="white", colormap=darker_cmap, width=800, height=500, max_words=100, prefer_horizontal=0.95, relative_scaling=0.5).generate_from_frequencies(frequencies)
        inset_ax.imshow(wc, interpolation='bilinear')
        inset_ax.axis('off')

    # 4. Plot each quadrant's word cloud
    plot_wc_in_quadrant(axes[0, 0], strengths_freqs, 'Greens')
    plot_wc_in_quadrant(axes[0, 1], weaknesses_freqs, 'Reds')
    plot_wc_in_quadrant(axes[1, 0], opportunities_freqs, 'Blues')
    plot_wc_in_quadrant(axes[1, 1], threats_freqs, 'Oranges')
    
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    
    # --- NEW: Manually place titles in the corners of each quadrant ---
    label_font_props = {'fontsize': 22, 'fontweight': 'bold', 'color': '#333333'}
    # Top-Left Quadrant
    axes[0, 0].text(0.03, 0.97, 'Strengths', transform=axes[0, 0].transAxes, ha='left', va='top', **label_font_props)
    # Top-Right Quadrant
    axes[0, 1].text(0.97, 0.97, 'Weaknesses', transform=axes[0, 1].transAxes, ha='right', va='top', **label_font_props)
    # Bottom-Left Quadrant
    axes[1, 0].text(0.03, 0.03, 'Opportunities', transform=axes[1, 0].transAxes, ha='left', va='bottom', **label_font_props)
    # Bottom-Right Quadrant
    axes[1, 1].text(0.97, 0.03, 'Threats', transform=axes[1, 1].transAxes, ha='right', va='bottom', **label_font_props)

    # 5. Add central axis lines after layout calculation
    pos_tl = axes[0, 0].get_position()
    pos_br = axes[1, 1].get_position()
    x_mid = (pos_tl.x0 + pos_br.x1) / 2
    y_mid = (pos_br.y0 + pos_tl.y1) / 2
    horiz_line = plt.Line2D([pos_tl.x0, pos_br.x1], [y_mid, y_mid], transform=fig.transFigure, color='#AAAAAA', linewidth=2, zorder=10)
    vert_line = plt.Line2D([x_mid, x_mid], [pos_br.y0, pos_tl.y1], transform=fig.transFigure, color='#AAAAAA', linewidth=2, zorder=10)
    fig.add_artist(horiz_line)
    fig.add_artist(vert_line)
    
    # 6. Save the figure
    try:
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"✓ Saved final SWOT quadrant chart to: {os.path.basename(filename)}")
    except Exception as e:
        print(f"Error saving final quadrant chart: {e}")

def plot_swot_summary_grid_by_source(df, filename):
    swot_order = ["Strengths", "Weakness", "Opportunity", "Threat"]
    pivot_tables = {}

    # Always include 'All Sources'
    sources = {"All Sources": df}
    unique_sources = df['Source'].dropna().unique()
    for src in unique_sources:
        filtered_df = df[df['Source'] == src]
        sources[src] = filtered_df

    max_y = 0
    for title, source_df in sources.items():
        if not source_df.empty:
            pivot = source_df.groupby(['Dimension', 'SWOT']).size().unstack(fill_value=0)
            pivot = pivot.reindex(columns=swot_order, fill_value=0)
            if not pivot.empty:
                current_max = pivot.sum(axis=1).max()
                if current_max > max_y: max_y = current_max
            pivot_tables[title] = pivot
        else:
            pivot_tables[title] = pd.DataFrame()

    y_limit = max_y * 1.15 if max_y > 0 else 10
    n_sources = len(pivot_tables)
    n_cols = 2
    n_rows = (n_sources + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8 * n_rows), sharey=True)
    axes = axes.flatten()

    swot_colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728']

    for i, (title, pivot_data) in enumerate(pivot_tables.items()):
        ax = axes[i]

        # Get the matching source_df *before* plotting
        source_df = sources[title]

        # 🆕 Count number of unique DocSources (split by ';')
        all_docs = set()
        if 'DocSources' in source_df.columns:
            for docs in source_df['DocSources'].dropna():
                for d in str(docs).split(';'):
                    d = d.strip()
                    if d:
                        all_docs.add(d)

        doc_count = len(all_docs)
        count = source_df.shape[0]

        if not pivot_data.empty:
            pivot_data.plot(kind='bar', stacked=True, ax=ax, color=swot_colors, edgecolor='white', linewidth=0.5, legend=False)
            ax.set_ylabel('Number of Codes', fontsize=12)
            totals = pivot_data.sum(axis=1)
            for j, total in enumerate(totals):
                if total > 0:
                    ax.text(j, total, f'{total:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Data Found', ha='center', va='center', fontsize=16, alpha=0.5, transform=ax.transAxes)
            ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])

        ax.set_title(f"{title}", fontsize=16, pad=10)
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), ha='right', rotation=45)
        ax.set_ylim(0, y_limit)

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Legend
    handles, labels = axes[0].get_legend_handles_labels() if not pivot_tables["All Sources"].empty else ([], [])
    if handles:
        fig.legend(handles, labels, title='SWOT Category', fontsize=12, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    fig.suptitle('SWOT Item Count by Dimension and Source', fontsize=22, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if filename:
        plt.savefig(filename, dpi=300)
    plt.close()

def plot_combined_swot_heatmap(df, filename, top_n=50):
    """
    Generates a single heatmap showing the top N most frequent codes
    versus the four SWOT categories for all dimensions combined.
    """
    print("Generating combined SWOT heatmap...")
    
    # To make the heatmap readable, we'll focus on the most frequent codes
    top_codes = df['Code'].value_counts().nlargest(top_n).index
    df_filtered = df[df['Code'].isin(top_codes)]

    swot_order = ["Strengths", "Weakness", "Opportunity", "Threat"]
    
    pivot_data = df_filtered.groupby(['Code', 'SWOT']).size().unstack(fill_value=0)
    pivot_data = pivot_data.reindex(columns=swot_order, fill_value=0)
    
    # Sort the codes by total frequency for a cleaner look
    pivot_data = pivot_data.loc[top_codes]

    if pivot_data.empty:
        print("  -> No data for combined heatmap, skipping.")
        return

    fig_height = max(10, len(pivot_data) * 0.3)
    plt.figure(figsize=(12, fig_height))
    
    sns.heatmap(pivot_data, annot=True, fmt="d", cmap="viridis", linewidths=.5)
    
    plt.title(f"Top {top_n} Code Frequencies by SWOT Category (All Dimensions)", fontsize=16, pad=20)
    plt.xlabel("SWOT Category", fontsize=12)
    plt.ylabel("Consolidated Code", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    try:
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"✓ Saved combined heatmap to: {os.path.basename(filename)}")
    except Exception as e:
        print(f"Error saving combined heatmap: {e}")

def plot_swot_heatmap_by_dimension(df_dimension, dimension_name, filename):
    """
    Generates a heatmap for a single dimension, showing code frequency
    across SWOT categories.
    """
    print(f"Generating SWOT heatmap for dimension: {dimension_name}")
    
    swot_order = ["Strengths", "Weakness", "Opportunity", "Threat"]
    
    pivot_data = df_dimension.groupby(['Code', 'SWOT']).size().unstack(fill_value=0)
    pivot_data = pivot_data.reindex(columns=swot_order, fill_value=0)
    pivot_data = pivot_data.loc[pivot_data.sum(axis=1) > 0]

    if pivot_data.empty:
        print(f"  -> No data for dimension '{dimension_name}', skipping heatmap.")
        return

    fig_height = max(8, len(pivot_data) * 0.4)
    plt.figure(figsize=(12, fig_height))
    
    sns.heatmap(pivot_data, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5)
    
    plt.title(f"SWOT Code Hotspots for Dimension: {dimension_name}", fontsize=16, pad=20)
    plt.xlabel("SWOT Category", fontsize=12)
    plt.ylabel("Consolidated Code", fontsize=12)
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    try:
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"✓ Saved heatmap to: {os.path.basename(filename)}")
    except Exception as e:
        print(f"Error saving heatmap for {dimension_name}: {e}")