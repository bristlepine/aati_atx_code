from aati_figures import *
import os
import pandas as pd

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Override default output path inside aati_figures
import aati_figures
aati_figures.OUTPUT_DIR = OUTPUT_DIR

# Helpers
def exists(path):
    return os.path.exists(path)

def save_df(df, path):
    df.to_pickle(path)

def load_df(path):
    return pd.read_pickle(path)

# run.py (NEW version)
def get_fig_name(step_num, base, sub_char=''):
    """Generates a formatted figure name, e.g., '10a_my_figure.png'."""
    sub_part = f"{sub_char}" if sub_char else ""
    fname = f"{step_num:02d}{sub_part}_{base}.png"
    return os.path.join(OUTPUT_DIR, fname)

# STEP 1: Load raw matrix
raw_path = os.path.join(OUTPUT_DIR, "raw_df.pkl")
print("▶ STEP 1: Load raw matrix")
if exists(raw_path):
    raw_df = load_df(raw_path)
    print("✓ STEP 1: Raw matrix already loaded.")
else:
    raw_df = load_raw_matrix(RAW_CSV)
    save_df(raw_df, raw_path)
    print(f"Loaded: {raw_df.shape}")

# STEP 2: Extract codebook
codebook_path = os.path.join(OUTPUT_DIR, "codebook_df.pkl")
print("▶ STEP 2: Extract codebook")
if exists(codebook_path):
    codebook_df = load_df(codebook_path)
    print("✓ STEP 2: Codebook already loaded.")
else:
    codebook_df = extract_codebook(raw_df, save_path=CODEBOOK_MAPPING_CSV)
    save_df(codebook_df, codebook_path)

# STEP 3: Auto-cluster codes
grouped_path = os.path.join(OUTPUT_DIR, "grouped_df.pkl")
print("▶ STEP 3: Auto-cluster codes")
if exists(grouped_path):
    grouped_df = load_df(grouped_path)
    print("✓ STEP 3: Grouped codes already loaded.")
else:
    grouped_df = auto_cluster_codes(codebook_df, num_clusters=25)
    save_df(grouped_df, grouped_path)

# STEP 4: Gemini labels
labeled_path = os.path.join(OUTPUT_DIR, "labeled_df.pkl")
labels_csv = os.path.join(OUTPUT_DIR, "group_labels.csv")
print("▶ STEP 4: Generate Gemini labels")
if exists(labeled_path):
    labeled_df = load_df(labeled_path)
    print("✓ STEP 4: Labeled groups already loaded.")
else:
    if exists(labels_csv):
        labels_df = pd.read_csv(labels_csv)
        labeled_df = grouped_df.merge(labels_df, on="Group", how="left")
    else:
        labeled_df = suggest_group_labels(grouped_df, save_path=labels_csv)
    save_df(labeled_df, labeled_path)
print("Sample labeled groups:", labeled_df.groupby("Label").size().head(5))

# STEP 5: Diagnostics
print("▶ STEP 5: Plot diagnostics")
fig_5a = get_fig_name(5, "group_size_bar_labeled", 'a')
if not exists(fig_5a):
    plot_group_size_bar(labeled_df, filename=fig_5a)
    print(f"✓ Saved diagnostics chart: {fig_5a}")
else:
    print(f"✓ Skipping existing chart: {os.path.basename(fig_5a)}")

fig_5b = get_fig_name(5, "code_clusters_pca", 'b')
if not exists(fig_5b):
    plot_code_clusters_scatter(labeled_df, method="pca", filename=fig_5b)
    print(f"✓ Saved diagnostics chart: {fig_5b}")
else:
    print(f"✓ Skipping existing chart: {os.path.basename(fig_5b)}")

# STEP 6: Main Heatmap
codified_path = os.path.join(OUTPUT_DIR, "codified_df.pkl")
print("▶ STEP 6: Generate main heatmap")
if exists(codified_path):
    codified_df = load_df(codified_path)
    print("✓ Codified data already loaded.")
else:
    codified_df = codify_swot(raw_df)
    codified_df = map_codified_to_labels(codified_df, labeled_df)
    codified_df = codified_df[codified_df["Label"].notna()]
    save_df(codified_df, codified_path)
    save_codified_matrix(codified_df, path=CODIFIED_CSV)

fig_6 = get_fig_name(6, "heatmap_swot_by_dimension")
if not exists(fig_6):
    plot_heatmap_swot_dimension(codified_df, filename=fig_6)
    print(f"✓ Saved main heatmap: {fig_6}")
else:
    print(f"✓ Skipping existing chart: {os.path.basename(fig_6)}")

# STEP 7: Codebook visual
print("▶ STEP 7: Grouped codes as text view")
fig_7 = get_fig_name(7, "codes_per_group_textgrid")
if not exists(fig_7):
    plot_codebook_faceted_groups(labeled_df, output_dir=OUTPUT_DIR, filename=fig_7)
    print(f"✓ Saved codebook visual: {fig_7}")
else:
    print(f"✓ Skipping existing chart: {os.path.basename(fig_7)}")

# STEP 9: Generate stacked bar chart by dimension
print("▶ STEP 9: Generate stacked bar chart")
fig_9 = get_fig_name(9, "swot_stacked_by_dimension")
if not exists(fig_9):
    plot_swot_stacked_bar_by_dimension(codified_df, output_dir=OUTPUT_DIR, filename=fig_9)
else:
    print(f"✓ Skipping existing chart: {os.path.basename(fig_9)}")


# STEP 10: Generate Hierarchical SWOT Network Visualizations
print("▶ STEP 10: Generate Hierarchical SWOT Networks")
unique_dimensions = codified_df['Dimension'].unique()
for i, dimension in enumerate(unique_dimensions):
    safe_dim_name = dimension.replace('&', 'and').replace(' ', '_').lower()
    sub_char = chr(97 + i) # 'a', 'b', 'c'...
    
    fig_10 = get_fig_name(10, f"hierarchical_network_{safe_dim_name}", sub_char)

    if not exists(fig_10):
        print(f"  -> Generating network for dimension: {dimension}")
        df_single_dimension = codified_df[codified_df['Dimension'] == dimension]
        plot_hierarchical_swot_network(
            codified_df=df_single_dimension,
            output_dir=OUTPUT_DIR,
            filename=fig_10
        )
    else:
        print(f"  -> Skipping existing network for dimension: {dimension}")


# STEP 11: Generate Dimension-Specific SWOT Hotspot Heatmaps
print("▶ STEP 11: Generate Dimension-Specific SWOT Hotspot Heatmaps")
for i, dimension in enumerate(unique_dimensions):
    safe_dim_name = dimension.replace('&', 'and').replace(' ', '_').lower()
    sub_char = chr(97 + i) # 'a', 'b', 'c'...
    
    fig_11 = get_fig_name(11, f"heatmap_hotspots_{safe_dim_name}", sub_char)
    
    if not exists(fig_11):
        plot_swot_group_heatmap(codified_df, dimension=dimension, filename=fig_11)
    else:
        print(f"  -> Skipping existing heatmap for dimension: {dimension}")

# STEP 12: Generate a single, focused raw code network
print("▶ STEP 12: Generate Specific Raw Code Network Example")

# --- Define the specific slice of data you want to visualize ---
DIMENSION_TO_PLOT = "Governance"
SWOT_TO_PLOT = "Strengths"
# --- You can change the two lines above to explore other slices ---

fig_12 = get_fig_name(12, f"network_{DIMENSION_TO_PLOT.lower()}_{SWOT_TO_PLOT.lower()}")

if not exists(fig_12):
    plot_single_swot_code_network(
        df=codified_df, 
        dimension=DIMENSION_TO_PLOT, 
        swot_category=SWOT_TO_PLOT, 
        filename=fig_12
    )
else:
    print(f"✓ Skipping existing chart: {os.path.basename(fig_12)}")


# STEP 13: Generate SWOT Quadrant Charts
print("▶ STEP 13: Generate SWOT Quadrant Charts")
unique_dimensions = codified_df['Dimension'].unique()

for i, dimension in enumerate(unique_dimensions):
    safe_dim_name = dimension.replace('&', 'and').replace(' ', '_').lower()
    sub_char = chr(97 + i)
    
    fig_13 = get_fig_name(13, f"quadrant_chart_{safe_dim_name}", sub_char)
    
    if not exists(fig_13):
        df_single_dimension = codified_df[codified_df['Dimension'] == dimension]
        plot_swot_quadrant_chart(
            df_dimension=df_single_dimension,
            dimension_name=dimension,
            filename=fig_13
        )
    else:
        print(f"  -> Skipping existing quadrant chart for dimension: {dimension}")


print("\n✅ All tasks complete!")