# run.py (Final Version)
# venv\Scripts\activate

from aati_figures import *
import os
import pandas as pd

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helpers ---
def exists(path): return os.path.exists(path)
def save_df(df, path): df.to_pickle(path)
def load_df(path): return pd.read_pickle(path)
def get_fig_name(step_num, base, sub_char=''):
    sub_part = f"{sub_char}" if sub_char else ""
    return os.path.join(OUTPUT_DIR, f"{step_num:02d}{sub_part}_{base}.png")

# === WORKFLOW ===

# STEPS 1-4: Data Loading and Processing
print("▶ STEP 1: Load and Codify Raw Data")
codified_path = os.path.join(OUTPUT_DIR, "01_codified_df.pkl")
if exists(codified_path):
    codified_df = load_df(codified_path)
    print("✓ Codified data already loaded.")
else:
    raw_df = load_raw_matrix(RAW_CSV)
    codified_df = codify_swot(raw_df)
    codified_df = normalize_dimension_names(codified_df)  # ← ADD THIS LINE
    codified_df['Code'] = codified_df['Text'].apply(lambda x: extract_bracketed_codes(x)[0] if extract_bracketed_codes(x) else None)
    codified_df.dropna(subset=['Code'], inplace=True)
    print("  -> Cleaning and standardizing 'Source' column...")
    codified_df['Source'] = codified_df['Source'].str.strip()
    source_mapping = {
        'Academic Literature': 'Literature Review',
        'AATI + TASC\n\nNotebook': 'AATI-TASC Synthesis'
    }
    codified_df['Source'] = codified_df['Source'].replace(source_mapping)
    save_df(codified_df, codified_path)
    print(f"✓ Raw data loaded, cleaned, and codified: {codified_df.shape}")

print("▶ STEP 2: Extract Initial Codebook")
raw_codebook_df = pd.DataFrame(codified_df['Code'].value_counts()).reset_index()
raw_codebook_df.columns = ['Code', 'Count']

print("▶ STEP 3: Consolidate Similar Codes")
consolidation_map_path = os.path.join(OUTPUT_DIR, "03_consolidation_map.pkl")
consolidation_csv_path = os.path.join(OUTPUT_DIR, "03_consolidation_map.csv")
if exists(consolidation_map_path):
    consolidation_map_df = load_df(consolidation_map_path)
    print("✓ Consolidation map already loaded.")
else:
    consolidation_map_df = consolidate_codes_with_gemini(raw_codebook_df, save_path=consolidation_csv_path)
    save_df(consolidation_map_df, consolidation_map_path)

print("▶ STEP 3a: Reporting Consolidated Changes")
changes_csv_path = os.path.join(OUTPUT_DIR, "03a_consolidated_code_changes.csv")
report_code_changes(consolidation_map_df, save_path=changes_csv_path)

print("▶ STEP 4: Apply Consolidation")
final_df = apply_consolidation_map(codified_df, consolidation_map_df)
final_df.to_csv(os.path.join(OUTPUT_DIR, "04_final_cleaned_matrix.csv"), index=False)
print("✓ Consolidation applied. Final dataset is ready.")

# STEP 5: Generate 2x2 SWOT Summary Grid Chart
print("▶ STEP 5: Generate SWOT Summary Grid by Source")
fig_5 = get_fig_name(5, "swot_summary_grid_by_source")
if not exists(fig_5):
    plot_swot_summary_grid_by_source(final_df, filename=fig_5)
else:
    print(f"  -> Skipping existing SWOT summary grid chart.")

# STEP 6: Generate SWOT Quadrant Charts
print("▶ STEP 6: Generate SWOT Quadrant Charts")
fig_6a = get_fig_name(6, "quadrant_chart_all_dimensions", 'a')
if not exists(fig_6a):
    print("  -> Generating quadrant chart for All Dimensions Combined...")
    plot_swot_quadrant_chart(df_dimension=final_df, dimension_name="All Dimensions Combined", filename=fig_6a)
else:
    print(f"  -> Skipping existing combined quadrant chart.")

unique_dimensions = final_df['Dimension'].unique()
for i, dimension in enumerate(unique_dimensions):
    safe_dim_name = dimension.replace('&', 'and').replace(' ', '_').lower()
    sub_char = chr(98 + i)
    fig_6 = get_fig_name(6, f"quadrant_chart_{safe_dim_name}", sub_char)
    if not exists(fig_6):
        df_single_dimension = final_df[final_df['Dimension'] == dimension]
        plot_swot_quadrant_chart(df_dimension=df_single_dimension, dimension_name=dimension, filename=fig_6)
    else:
        print(f"  -> Skipping existing quadrant chart for dimension: {dimension}")

# ----------------------------------------------------------------------
# STEP 7: Generate SWOT Heatmaps
# ----------------------------------------------------------------------
print("▶ STEP 7: Generate SWOT Heatmaps")

# --- First, generate the combined heatmap for ALL dimensions ---
fig_7a = get_fig_name(7, "heatmap_swot_combined", 'a')
if not exists(fig_7a):
    plot_combined_swot_heatmap(final_df, filename=fig_7a)
else:
    print(f"  -> Skipping existing combined SWOT heatmap.")

# --- Then, loop and generate a heatmap for each individual dimension ---
for i, dimension in enumerate(unique_dimensions):
    safe_dim_name = dimension.replace('&', 'and').replace(' ', '_').lower()
    sub_char = chr(98 + i)
    fig_7 = get_fig_name(7, f"heatmap_hotspots_{safe_dim_name}", sub_char)
    
    if not exists(fig_7):
        df_single_dimension = final_df[final_df['Dimension'] == dimension]
        plot_swot_heatmap_by_dimension(df_dimension=df_single_dimension, dimension_name=dimension, filename=fig_7)
    else:
        print(f"  -> Skipping existing heatmap for dimension: {dimension}")
# ----------------------------------------------------------------------

print("\n✅ All tasks complete!")