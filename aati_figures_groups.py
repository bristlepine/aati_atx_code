# aati_figures.py
# .\venv\Scripts\Activate
# gcloud auth application-default login
# pip freeze > requirements.txt

"""
Author: zk
Date Updated: 11 June 2025

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
# One-time setup (run these in terminal before first use):
#
# python -m nltk.downloader stopwords
# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_lg
# ─────────────────────────────────────────────────────────────

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
import re
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import networkx as nx # <--- ENSURE THIS LINE IS PRESENT HERE
import textwrap # <-- Make sure this is imported at the top of the file
from collections import Counter # Make sure Counter is imported at the top of your file
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable.")
genai.configure(api_key=API_KEY)

# Optional: Gemini integration (if enabled)
# from your_gemini_module import query_gemini  # Replace with real implementation

# ─────────────────────────────────────────────────────────────
# 2. Configuration
# ─────────────────────────────────────────────────────────────

RAW_CSV = "Desk Research Matrix - Combined Matrix.csv"
OUTPUT_DIR = "outputs"
CODIFIED_CSV = os.path.join(OUTPUT_DIR, "00_codified_matrix.csv")
CODEBOOK_MAPPING_CSV = os.path.join(OUTPUT_DIR, "00_codebook_mapping.csv")

USE_GEMINI = True  # Set to True if using Gemini API

# ─────────────────────────────────────────────────────────────
# 3. Utility Functions
# ─────────────────────────────────────────────────────────────

def extract_bracketed_codes(text):
    """Return a list of codes inside brackets [like this]."""
    return re.findall(r"\[(.*?)\]", text or "")


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

def extract_codebook(df, save_path=None):
    """
    Extract bracketed codes from SWOT fields.
    Returns a frequency DataFrame and optionally saves to CSV.
    """
    pattern = re.compile(r"\[(.*?)\]")
    all_codes = []

    for column in ["Strengths", "Weakness", "Opportunity", "Threat"]:
        df[column] = df[column].fillna("")
        for entry in df[column]:
            matches = extract_bracketed_codes(entry)
            all_codes.extend(code.strip() for code in matches)

    code_counts = Counter(all_codes)
    codebook_df = pd.DataFrame(code_counts.items(), columns=["Code", "Count"]).sort_values(by="Count", ascending=False)

    if save_path:
        codebook_df.to_csv(save_path, index=False)
        print(f"Saved raw codebook to {save_path}")

    return codebook_df

def auto_cluster_codes(codebook_df, num_clusters=40):
    """
    Automatically cluster codes into thematic groups using spaCy + KMeans.
    Returns a DataFrame with 'Code', 'Group' columns.
    """
    nlp = spacy.load("en_core_web_lg")
    codes = codebook_df["Code"].tolist()

    # Get vector for each code
    vectors = np.array([nlp(code).vector for code in codebook_df["Code"]])

    # Cluster vectors
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(vectors)

    # Assign Group names
    codebook_df["Group"] = [f"Group_{i:02d}" for i in labels]
    return codebook_df[["Code", "Group"]]

# ─────────────────────────────────────────────────────────────
# 5. Optional NLP Enhancement (spaCy or Gemini)
# ─────────────────────────────────────────────────────────────

def suggest_group_labels(codebook_df, save_path=os.path.join(OUTPUT_DIR, "group_labels.csv")):
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

    def clean_label(label):
        words = re.findall(r"\b\w+\b", label)
        return " ".join(words[:2]) if words else "Unlabeled"

    grouped = codebook_df.groupby("Group")["Code"].apply(list).to_dict()
    labels_dict = {}

    fewshot_examples = (
    "Here are some example groups and how they were labeled:\n"
    "- ['Donor Dependence', 'External Funding', 'Reliance on Partners'] → Donor Funding Risk\n"
    "- ['Champion Leader', 'Ministerial Backing', 'Political Buy-in'] → High-Level Political Support\n"
    "- ['Delayed Execution', 'Delivery Lags', 'Slow Turnaround'] → Implementation Bottlenecks\n"
    "- ['Data Gaps', 'Lack of Evidence', 'No M&E'] → Weak Evidence Systems\n"
    "- ['Reporting Confusion', 'Mandate Ambiguity', 'Jurisdiction Clash'] → Mandate and Role Clarity\n"
    "- ['Staff Turnover', 'Vacancies', 'Skill Shortages'] → Capacity Constraints\n"
    "- ['Private Sector Reluctance', 'Low Trust', 'Misaligned Incentives'] → Stakeholder Engagement Barriers\n\n"
    )

    print("Generating descriptive labels for each group using Gemini (few-shot, max 3 words)...")

    for group, codes in grouped.items():
        prompt = (
            fewshot_examples +
            "Now, give a short and specific label for this group of codes.\n"
            "Use a **maximum of 3 words**, but fewer is okay.\n"
            "Avoid vague terms like 'issues', 'challenges', or 'factors'.\n"
            "Focus on clarity and actionability.\n\n"
            + "\n".join(f"- {code}" for code in codes)
        )
        try:
            response = model.generate_content(prompt)
            label_raw = response.text.strip().split("\n")[0]
            label = clean_label(label_raw)  # already truncates to 3 words
        except Exception as e:
            label = "Unlabeled"
            print(f"Failed to label {group}: {e}")
        labels_dict[group] = label
        print(f"{group}: {label}")

    labels_df = pd.DataFrame(list(labels_dict.items()), columns=["Group", "Label"])
    labels_df.to_csv(save_path, index=False)
    print(f"Saved group labels to {save_path}")

    codebook_df = codebook_df.merge(labels_df, on="Group", how="left")
    return codebook_df

def map_codified_to_labels(codified_df, codebook_df_with_labels):
    """
    Extract bracketed code from codified text and map to group label.
    """
    pattern = re.compile(r"\[(.*?)\]")
    codified_df["Code"] = codified_df["Text"].apply(
        lambda x: extract_bracketed_codes(x)[0] if extract_bracketed_codes(x) else None
    )

    # Merge in Label (ignore Group entirely)
    mapping = codebook_df_with_labels[["Code", "Label"]].drop_duplicates()
    codified_df = codified_df.merge(mapping, on="Code", how="left")
    return codified_df

# ─────────────────────────────────────────────────────────────
# 6. Save Codified Data
# ─────────────────────────────────────────────────────────────

def save_codified_matrix(df, path=CODIFIED_CSV):
    df.to_csv(path, index=False)

# ─────────────────────────────────────────────────────────────
# 7. Generate Visualizations
# ─────────────────────────────────────────────────────────────

def plot_code_clusters_scatter(codebook_df, method='pca', filename=None):
    nlp = spacy.load("en_core_web_sm")
    vectors = np.array([nlp(code).vector for code in codebook_df["Code"]])

    if method == 'pca':
        reducer = PCA(n_components=2)
        title = "Code Clusters (PCA)"
        suffix = "pca"
    else:
        reducer = TSNE(n_components=2, random_state=42)
        title = "Code Clusters (t-SNE)"
        suffix = "tsne"

    reduced = reducer.fit_transform(vectors)
    df_plot = codebook_df.copy()
    df_plot["x"] = reduced[:, 0]
    df_plot["y"] = reduced[:, 1]

    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        data=df_plot,
        x="x",
        y="y",
        hue="Label",
        palette="tab20",
        s=70,
        alpha=0.7,
        edgecolor="k",
        legend=False
    )
    for _, row in df_plot.iterrows():
        plt.text(row["x"], row["y"], row["Code"], fontsize=7, alpha=0.6)
    plt.title(title)
    plt.tight_layout()

    if filename is None:
        filename = f"code_clusters_{suffix}.png"
    plt.savefig(filename)
    plt.close()


def plot_group_size_bar(codebook_df, filename=None):
    group_counts = codebook_df["Label"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=group_counts.index, y=group_counts.values, palette="tab20")
    plt.title("Number of Codes per Group (Labeled)")
    plt.xlabel("Group Label")
    plt.ylabel("Count")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()

    if filename is None:
        filename = "group_size_bar_labeled.png"
    plt.savefig(filename)
    plt.close()


def plot_dendrogram(codebook_df, filename=None):
    nlp = spacy.load("en_core_web_sm")
    vectors = np.array([nlp(code).vector for code in codebook_df["Code"]])
    labels = [
        f"{row['Label']} → {row['Code']}" 
        for _, row in codebook_df.iterrows()
    ]

    linked = linkage(vectors, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(linked,
               labels=labels,
               orientation='top',
               leaf_rotation=90,
               leaf_font_size=8)
    plt.title("Dendrogram of Code Clusters")
    plt.tight_layout()

    if filename is None:
        filename = "code_cluster_dendrogram.png"
    plt.savefig(filename)
    plt.close()


def plot_bar_chart(df, filename=None):
    pivot = df.groupby(["Dimension", "SWOT"]).size().unstack().fillna(0)
    pivot.plot(kind="bar", stacked=True)
    plt.title("SWOT Items by Dimension")
    plt.ylabel("Count")
    plt.tight_layout()

    if filename is None:
        filename = "strengths_by_dimension_bar.png"
    plt.savefig(filename)
    plt.close()


def generate_wordcloud(df, swot_type, dimension=None, filename=None):
    filtered = df[df["SWOT"] == swot_type]
    if dimension:
        filtered = filtered[filtered["Dimension"] == dimension]

    if "Label" not in filtered.columns and "Group" in filtered.columns:
        labels_df = pd.read_csv(os.path.join(OUTPUT_DIR, "00_group_labels.csv"))
        filtered = filtered.merge(labels_df, on="Group", how="left")

    text = " ".join(filtered["Label"].dropna())
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)

    if filename is None:
        filename = f"{swot_type.lower()}_{dimension.lower() if dimension else 'all'}_wordcloud.png"
    wc.to_file(os.path.join(OUTPUT_DIR, filename))


def plot_heatmap_swot_dimension(df, filename=None):
    pivot = df.groupby(["Label", "SWOT", "Dimension"]).size().reset_index(name="Count")
    heatmap_data = pivot.pivot_table(index="Label", columns="Dimension", values="Count", aggfunc="sum", fill_value=0)

    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5)
    plt.title("SWOT Group Label Frequency by Dimension")
    plt.ylabel("Group Label")
    plt.xlabel("Dimension")
    plt.tight_layout()

    if filename is None:
        filename = "heatmap_swot_by_dimension.png"
    plt.savefig(filename)
    plt.close()

def plot_swot_group_heatmap(df, dimension, filename):
    """
    Generates a single heatmap for a given Dimension, showing the frequency
    of Group Labels across SWOT categories.
    """
    # Define a consistent order for the SWOT columns
    swot_order = ["Strengths", "Weakness", "Opportunity", "Threat"]
    
    # Filter data for the specified dimension
    dim_df = df[df['Dimension'] == dimension]
    
    # Create the pivot table
    pivot_data = dim_df.groupby(['Label', 'SWOT']).size().unstack(fill_value=0)
    pivot_data = pivot_data.reindex(columns=swot_order, fill_value=0)

    if pivot_data.empty:
        print(f"     Skipping empty dimension: {dimension}")
        return

    fig_height = max(8, len(pivot_data) * 0.4)
    plt.figure(figsize=(12, fig_height))
    
    sns.heatmap(
        pivot_data, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5, cbar=True
    )
    
    plt.title(f"SWOT Hotspots for Dimension: {dimension}", fontsize=16, pad=20)
    plt.xlabel("SWOT Category", fontsize=12)
    plt.ylabel("Group Label", fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    try:
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"✓ Saved heatmap to: {filename}")
    except Exception as e:
        print(f"Error saving heatmap for {dimension}: {e}")
def plot_codebook_faceted_groups(codebook_df, output_dir=OUTPUT_DIR, filename=None):
    import textwrap
    import matplotlib.patches as patches

    grouped = codebook_df.groupby("Label")["Code"].apply(list).reset_index()
    grouped["Code"] = grouped["Code"].apply(lambda codes: "\n".join(f"• {code}" for code in codes))

    grouped["Size"] = grouped["Code"].apply(lambda x: x.count("\n") + 1)
    grouped = grouped.sort_values(by="Size", ascending=False)

    n_cols = 4
    n_rows = int(np.ceil(len(grouped) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5), constrained_layout=True)

    for ax, (_, row) in zip(axes.flat, grouped.iterrows()):
        ax.axis("off")
        ax.add_patch(
            patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", linewidth=0.5,
                                   edgecolor="gray", facecolor="white", transform=ax.transAxes)
        )
        ax.text(0.03, 1.02, row["Label"], fontsize=13, fontweight="bold", va="bottom", ha="left", transform=ax.transAxes)
        ax.text(0.03, 0.92, row["Code"], fontsize=8, va="top", ha="left", transform=ax.transAxes)

    for i in range(len(grouped), len(axes.flat)):
        axes.flat[i].axis("off")

    fig.suptitle("Grouped Codes (Text View)", fontsize=14)

    if filename is None:
        filename = "codes_per_group_textgrid.png"
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_swot_stacked_bar_by_dimension(df, output_dir=OUTPUT_DIR, filename=None):
    swot_order = ["Strengths", "Weakness", "Opportunity", "Threat"]
    pivot = df.groupby(["Dimension", "SWOT"]).size().unstack(fill_value=0)

    pivot = pivot[swot_order] if all(col in pivot.columns for col in swot_order) else pivot

    pivot.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab10")
    plt.title("SWOT Summary by Dimension (Stacked Bar)")
    plt.xlabel("Dimension")
    plt.ylabel("Number of Items")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="SWOT Category")
    plt.tight_layout()

    if filename is None:
        filename = "swot_stacked_by_dimension.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✓ Saved stacked bar chart: {os.path.join(output_dir, filename)}")


def plot_hierarchical_swot_network(codified_df, output_dir=None, filename=None):
    """
    Generates a static, well-spaced hierarchical network visualization for a
    SINGLE dimension (Dimension -> SWOT Category -> Group Label).
    """
    print("Generating hierarchical SWOT network visualization...")

    if output_dir is None:
        output_dir = OUTPUT_DIR # Fallback to global if not provided

    if 'Label' not in codified_df.columns:
        print("Error: 'Label' column not found. Cannot generate network.")
        return

    df_for_viz = codified_df[codified_df['Label'].notna() & codified_df['Dimension'].notna() & codified_df['SWOT'].notna()].copy()

    if df_for_viz.empty:
        print("No data available for visualization after filtering.")
        return

    # Since the input df is pre-filtered, all rows have the same dimension.
    dimension_name = df_for_viz['Dimension'].iloc[0]
    
    G = nx.DiGraph()
    node_colors = {}
    swot_color_map = {"Strengths": "#4CAF50", "Weakness": "#F44336", "Opportunity": "#FFC107", "Threat": "#2196F3"}

    # Add single Dimension node
    G.add_node(dimension_name, label=dimension_name, type="Dimension", color="#90CAF9", size=3000)
    node_colors[dimension_name] = "#90CAF9"

    # Add SWOT and Label nodes
    for _, row in df_for_viz.iterrows():
        swot_cat, label = row["SWOT"], row["Label"]
        
        # Add SWOT node if it doesn't exist
        if not G.has_node(swot_cat):
            color = swot_color_map.get(swot_cat, "#BDBDBD")
            G.add_node(swot_cat, label=swot_cat, type="SWOT", color=color, size=2000)
            G.add_edge(dimension_name, swot_cat)
            node_colors[swot_cat] = color

        # Add Label node if it doesn't exist
        if not G.has_node(label):
            G.add_node(label, label=label, type="Label", color="#E0E0E0", size=1500)
            node_colors[label] = "#E0E0E0"
        
        # Add edge from SWOT to Label if it doesn't exist
        if not G.has_edge(swot_cat, label):
            G.add_edge(swot_cat, label)

    # --- New Plotting & Layout Logic ---
    num_labels = len([n for n, d in G.nodes(data=True) if d['type'] == 'Label'])
    fig_height = max(8, num_labels * 0.5)  # Adjust height based on label count
    plt.figure(figsize=(18, fig_height))

    # --- Hierarchical Layout (Multipartite) ---
    pos = {}
    x_coords = {'Dimension': 0, 'SWOT': 2, 'Label': 4.5} # X-coordinates for each column

    # Layer 0: Dimension
    pos[dimension_name] = (x_coords['Dimension'], 0)

    # Layer 1: SWOT
    swot_order = ["Strengths", "Weakness", "Opportunity", "Threat"]
    swot_nodes = sorted([n for n, d in G.nodes(data=True) if d['type'] == 'SWOT'], key=lambda n: swot_order.index(n))
    swot_y = np.linspace(len(swot_nodes), -len(swot_nodes), len(swot_nodes)) if swot_nodes else [0]
    for node, y in zip(swot_nodes, swot_y):
        pos[node] = (x_coords['SWOT'], y)

    # Layer 2: Labels
    label_nodes = sorted([n for n, d in G.nodes(data=True) if d['type'] == 'Label'])
    label_y_max = max(10, len(label_nodes) / 2) # Spread vertically
    label_y = np.linspace(label_y_max, -label_y_max, len(label_nodes)) if label_nodes else [0]
    for node, y in zip(label_nodes, label_y):
        pos[node] = (x_coords['Label'], y)
    
    # --- Drawing ---
    node_sizes = [G.nodes[n]['size'] for n in G]
    node_colors_list = [G.nodes[n]['color'] for n in G]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, node_size=node_sizes, alpha=0.95)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowstyle="-|>", arrowsize=15, alpha=0.5, connectionstyle='arc3,rad=0.05')

    # Draw labels with better alignment and wrapping
    for node, (x, y) in pos.items():
        label_text = G.nodes[node]['label']
        if G.nodes[node]['type'] == 'Label':
            wrapped_text = "\n".join(textwrap.wrap(label_text, width=30))
            plt.text(x + 0.1, y, wrapped_text, ha='left', va='center', fontsize=10)
        else:
            plt.text(x, y, label_text, ha='center', va='center', fontsize=12, fontweight='bold')

    plt.title(f"Hierarchical SWOT Groups for Dimension: {dimension_name}", fontsize=18, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.xlim(-1, x_coords['Label'] + 4) # Add padding to the right for long labels

    # --- Saving ---
    full_output_path = filename if filename is not None else os.path.join(output_dir, "hierarchical_swot_network.png")
    if output_dir and filename and not os.path.dirname(filename):
        full_output_path = os.path.join(output_dir, filename)

    try:
        plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved hierarchical SWOT network to: {full_output_path}")
    except Exception as e:
        print(f"Error saving hierarchical network visualization: {e}")

def plot_single_swot_code_network(df, dimension, swot_category, filename):
    """
    Generates a simple, clean, left-to-right network for a single Dimension 
    and SWOT category, showing the specific raw codes.
    """
    print(f"Generating network for Dimension='{dimension}' and SWOT='{swot_category}'...")
    
    # 1. Filter data
    df_filtered = df[
        (df['Dimension'] == dimension) & 
        (df['SWOT'] == swot_category)
    ].dropna(subset=['Code'])

    if df_filtered.empty:
        print(f"No data found for '{dimension} -> {swot_category}'. Skipping plot.")
        return

    # 2. Build the graph
    G = nx.DiGraph()
    swot_color_map = {"Strengths": "#4CAF50", "Weakness": "#F44336", "Opportunity": "#FFC107", "Threat": "#2196F3"}
    
    G.add_node(dimension, type='Dimension', color='#ADD8E6', size=5000)
    G.add_node(swot_category, type='SWOT', color=swot_color_map.get(swot_category, "#BDBDBD"), size=4000)
    unique_codes = sorted(list(df_filtered['Code'].unique()))
    for code in unique_codes:
        G.add_node(code, type='Code', color='#E8E8E8', size=3000)

    G.add_edge(dimension, swot_category)
    for code in unique_codes:
        G.add_edge(swot_category, code)

    # 3. Manual Left-to-Right Layout and Plotting
    num_codes = len(unique_codes)
    # Adjust height to give each code label plenty of vertical space
    fig_height = max(6, num_codes * 1.2)
    plt.figure(figsize=(16, fig_height))

    # --- Manually define positions for a clean hierarchy ---
    pos = {}
    # Define the x-coordinate for each column
    x_coords = {'Dimension': 0, 'SWOT': 2, 'Code': 4}

    # Position the single Dimension and SWOT nodes in the center of the y-axis
    pos[dimension] = (x_coords['Dimension'], 0)
    pos[swot_category] = (x_coords['SWOT'], 0)

    # Create evenly spaced y-coordinates to fan out the code nodes vertically
    y_positions = np.linspace(-num_codes, num_codes, num_codes) if num_codes > 1 else [0]
    
    for i, code in enumerate(unique_codes):
        pos[code] = (x_coords['Code'], y_positions[i])
    # --- End of manual layout ---

    # Draw the graph components
    nx.draw_networkx_nodes(G, pos, node_size=[G.nodes[n]['size'] for n in G], 
                           node_color=[G.nodes[n]['color'] for n in G])
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, 
                           alpha=0.6, connectionstyle='arc3,rad=0.1')

    # Draw labels with wrapping and better alignment
    for node, (x, y) in pos.items():
        data = G.nodes[node]
        label_text = node
        ha = 'center'
        fontweight = 'bold'
        fontsize = 12

        if data['type'] == 'Code':
            label_text = node.replace('[', '').replace(']', '')
            wrapped_label = '\n'.join(textwrap.wrap(label_text, width=30))
            ha = 'left'
            fontweight = 'normal'
            fontsize = 9
            # Draw the text to the right of the node
            plt.text(x + 0.1, y, wrapped_label, ha=ha, va='center', fontsize=fontsize, fontweight=fontweight)
        else:
            # Draw text in the center of the node
            plt.text(x, y, label_text, ha=ha, va='center', fontsize=fontsize, fontweight=fontweight)

    plt.title(f"Raw Codes for {dimension}: {swot_category}", fontsize=16, loc='left', pad=20)
    plt.box(False)
    # Adjust plot limits to ensure all labels are visible
    plt.margins(x=0.1, y=0.1)
    plt.tight_layout()

    # 4. Save the figure
    try:
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"✓ Saved single SWOT network to: {filename}")
    except Exception as e:
        print(f"Error saving single SWOT network: {e}")


def plot_swot_quadrant_chart(df_dimension, dimension_name, filename):
    """
    Generates a SWOT quadrant chart for a single dimension, displaying the
    raw codes as a word cloud, treating multi-word codes as single entities.
    """
    print(f"Generating SWOT quadrant word cloud for: {dimension_name}")

    # 1. Prepare frequency dictionaries for each quadrant
    def prepare_frequencies(codes_series):
        """Creates a dictionary of { 'Code Phrase': count }."""
        if codes_series.size == 0:
            return {}
        # Clean codes by removing brackets
        cleaned_codes = [code.replace('[', '').replace(']', '').strip() for code in codes_series]
        # Return a frequency dictionary, e.g., {'Political Risk': 2, 'High-level Support': 1}
        return Counter(cleaned_codes)

    strengths_freqs = prepare_frequencies(df_dimension[df_dimension['SWOT'] == 'Strengths']['Code'].dropna().unique())
    weaknesses_freqs = prepare_frequencies(df_dimension[df_dimension['SWOT'] == 'Weakness']['Code'].dropna().unique())
    opportunities_freqs = prepare_frequencies(df_dimension[df_dimension['SWOT'] == 'Opportunity']['Code'].dropna().unique())
    threats_freqs = prepare_frequencies(df_dimension[df_dimension['SWOT'] == 'Threat']['Code'].dropna().unique())

    # 2. Set up a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"SWOT Analysis for Dimension: {dimension_name}", fontsize=22, y=0.97)

    # 3. Helper function to generate and plot a word cloud from frequencies
    def plot_wc_in_quadrant(ax, frequencies, title, colormap):
        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
        ax.axis('off')

        # Check if the frequency dictionary is empty
        if not frequencies:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, fontsize=14, alpha=0.5)
            return

        # Create the word cloud object
        wc = WordCloud(background_color="white", colormap=colormap,
                       width=400, height=400, max_words=100,
                       contour_width=1, contour_color=colormap,
                       prefer_horizontal=0.95, relative_scaling=0.5)
        
        # *** Generate the word cloud from the frequency dictionary ***
        wc.generate_from_frequencies(frequencies)
        
        ax.imshow(wc, interpolation='bilinear')

    # 4. Plot each word cloud in its designated quadrant
    # Upper-Left: Weaknesses
    plot_wc_in_quadrant(axes[0, 0], weaknesses_freqs, 'Weaknesses', 'Reds')
    # Upper-Right: Strengths
    plot_wc_in_quadrant(axes[0, 1], strengths_freqs, 'Strengths', 'Greens')
    # Lower-Left: Opportunities
    plot_wc_in_quadrant(axes[1, 0], opportunities_freqs, 'Opportunities', 'Blues')
    # Lower-Right: Threats
    plot_wc_in_quadrant(axes[1, 1], threats_freqs, 'Threats', 'Oranges')

    # 5. Final adjustments and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    try:
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"✓ Saved SWOT quadrant word cloud to: {filename}")
    except Exception as e:
        print(f"Error saving quadrant word cloud: {e}")

# End of file