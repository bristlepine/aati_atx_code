# preprocessor.py
# venv\Scripts\activate

import csv
import re
import os

# Input text files and their sources
input_files = {
    "NotebookLM_Gray_Matrix.txt": "Gray Lit",
    "NotebookLM_Lit_Matrix.txt": "Academic Lit",
    "NotebookLM_KII_Matrix.txt": "KII"
}

# Helper function to extract SWOT bullets
def extract_bullets(section_text):
    entries = re.findall(r'◦\s*\[(.*?)\]\s*(.*?)(?=\n◦|\n•|\Z)', section_text, re.DOTALL)
    results = []
    for label, content in entries:
        label = label.strip()
        content = content.strip().replace('\n', ' ')
        results.append(f"[{label}] {content}")
    return results

# Helper function to extract SWOT entries from text
def process_text(text, source):
    swot_rows = []

    # Match each dimension block
    dimension_blocks = re.findall(r'(Dimension\s+\d+:\s+.*?)(?=Dimension\s+\d+:|\Z)', text, re.DOTALL)
    print(f"\n📘 {source} — Found {len(dimension_blocks)} dimension block(s).")

    for idx, block in enumerate(dimension_blocks, start=1):
        dimension_header = re.search(r'Dimension\s+\d+:\s+(.*)', block)
        if not dimension_header:
            print(f"[Warning] Skipping block {idx} — couldn't extract dimension name.")
            continue

        dimension_name = dimension_header.group(1).strip()
        print(f"🔹 {source} → Dimension {idx}: {dimension_name}")

        # Extract each SWOT section
        def get_section(section_title):
            pattern = rf'{section_title}:\s*•(.*?)(?=(Strengths:|Weaknesses:|Opportunities:|Threats:|Dimension\s+\d+:|\Z))'
            match = re.search(pattern, block, re.DOTALL)
            return extract_bullets(match.group(1)) if match else []

        strengths = get_section("Strengths")
        weaknesses = get_section("Weaknesses")
        opportunities = get_section("Opportunities")
        threats = get_section("Threats")

        print(f"   - ✅ Strengths: {len(strengths)}")
        print(f"   - ⚠️  Weaknesses: {len(weaknesses)}")
        print(f"   - 🌱 Opportunities: {len(opportunities)}")
        print(f"   - 🔥 Threats: {len(threats)}")

        swot_rows.append({
            'Source': source,
            'Dimension': dimension_name,
            'Strengths': "\n\n".join(strengths),
            'Weakness': "\n\n".join(weaknesses),
            'Opportunity': "\n\n".join(opportunities),
            'Threat': "\n\n".join(threats)
        })

    return swot_rows

# Collect rows from all input files
swot_matrix = []

for file_path, source_name in input_files.items():
    print(f"\n📂 Checking file: {file_path}")
    if not os.path.exists(file_path):
        print(f"⚠️  Skipped missing file: {file_path}")
        continue

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        swot_matrix.extend(process_text(text, source_name))

# Write combined CSV
csv_filename = "NotebookLM_Combined_Matrix.csv"
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Source', 'Dimension', 'Strengths', 'Weakness', 'Opportunity', 'Threat']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in swot_matrix:
        writer.writerow(row)

print(f"\n✅ Done! Combined CSV file '{csv_filename}' created with {len(swot_matrix)} total row(s).")
