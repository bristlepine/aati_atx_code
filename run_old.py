# run_aati.py
# .\venv\Scripts\Activate
# pip freeze > requirements.txt
# gcloud auth application-default login
# python aati_run.py

from aati_figures import *
import os
import pandas as pd

STEP = "all"  # Change to "load", "codify", "extract", "cluster", "visualize" as needed

os.makedirs(OUTPUT_DIR, exist_ok=True)

raw_df = None
codified_df = None
codebook_df = None

# STEP 1: Load raw matrix
if STEP in ("load", "all"):
    print("STEP 1: Load raw matrix")
    raw_df = load_raw_matrix(RAW_CSV)
    print("Loaded raw matrix with shape:", raw_df.shape)

# STEP 2: Codify SWOT entries
if STEP in ("codify", "all"):
    if raw_df is None:
        raw_df = load_raw_matrix(RAW_CSV)
    print("STEP 2: Codify SWOT entries")
    codified_df = codify_swot(raw_df)
    codified_df.to_csv(CODIFIED_CSV, index=False)
    print(f"Codified SWOT saved to {CODIFIED_CSV}")

# STEP 3: Extract codebook
if STEP in ("extract", "all"):
    if raw_df is None:
        raw_df = load_raw_matrix(RAW_CSV)
    print("STEP 3: Extract codebook")
    codebook_df = extract_codebook(raw_df, save_path=os.path.join(OUTPUT_DIR, "03_codebook_raw_freq.csv"))
    print(codebook_df.head())

# STEP 4: Auto-cluster code (no change needed here, it correctly adds labels)
if STEP in ("cluster", "all"):
    if codebook_df is None:
        codebook_df = pd.read_csv(os.path.join(OUTPUT_DIR, "03_codebook_raw_freq.csv"))
    print("STEP 4: Auto-cluster codes")
    grouped_df = auto_cluster_codes(codebook_df, num_clusters=40)
    grouped_df.to_csv(CODEBOOK_MAPPING_CSV, index=False)
    
    # Gemini group labels
    grouped_df = suggest_group_labels(grouped_df)
    
    plot_code_clusters_scatter(grouped_df, method="pca")
    plot_code_clusters_scatter(grouped_df, method="tsne")
    plot_group_size_bar(grouped_df)
    plot_dendrogram(grouped_df)
    print(f"Clustered mapping saved to {CODEBOOK_MAPPING_CSV}")
    print(grouped_df.head())

# STEP 5: Generate visualizations
if STEP in ("visualize", "all"):
    if codified_df is None:
        codified_df = pd.read_csv(CODIFIED_CSV)
    
    # --- IMPORTANT CHANGE STARTS HERE ---
    # Always load and merge labels for codebook_df in visualize step
    # This ensures 'Label' is present regardless of previous steps' state
    codebook_df = pd.read_csv(CODEBOOK_MAPPING_CSV) # Load the clustered codes (Code, Group)
    label_df = pd.read_csv(os.path.join(OUTPUT_DIR, "04a4_group_labels.csv")) # Load the Group to Label mapping
    codebook_df = codebook_df.merge(label_df, on="Group", how="left") # Merge to add 'Label'
    
    print("Post-merge columns (for visualization):", list(codebook_df.columns))
    print(codebook_df.head())
    print("Codebook Groups (for visualization):", codebook_df["Group"].unique()[:5])
    print("Label Groups (for visualization):", label_df["Group"].unique()[:5])
    # --- IMPORTANT CHANGE ENDS HERE ---


    # 🧠 Map bracketed codes from Text into codified_df
    codified_df = map_codified_to_labels(codified_df, codebook_df)

    # Drop rows without mapped labels (optional)
    codified_df = codified_df[codified_df["Label"].notna()]

    # Generate charts
    plot_bar_chart(codified_df)
    plot_heatmap_swot_dimension(codified_df)
    plot_grouped_bar_by_dimension(codified_df, top_n=8)

    # Word clouds for each (SWOT × Dimension)
    for swot in ["Strengths", "Weakness", "Opportunity", "Threat"]:
        for dim in codified_df["Dimension"].dropna().unique():
            generate_wordcloud(codified_df, swot_type=swot, dimension=dim)
    print("Visualizations saved to /outputs/")