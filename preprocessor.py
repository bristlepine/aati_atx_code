# preprocessor.py
# venv\Scripts\activate

import csv
import re
import os
from collections import defaultdict

# Input text files and their sources
input_files = {
    "NotebookLM_Gray_Matrix.txt": "Gray Lit",
    "NotebookLM_Lit_Matrix.txt": "Academic Lit",
    "NotebookLM_KIIFGD_Matrix.txt": "KII FGD"
}

# Extract SWOT bullets from • or ◦ lists, including inline sources
def extract_bullets(section_text):
    # This pattern looks for a bullet, then captures its label and content.
    # It then optionally captures a 'Source:' line immediately following the bullet content.
    entries = re.findall(
        r'[◦•]\s*\[(.*?)\]\s*(.*?)(?:\s*Source:\s*(.+?))?(?=\n[◦•]|\Z)',
        section_text,
        re.DOTALL
    )
    results = []
    for label, content, inline_source in entries:
        label = label.strip()
        content = content.strip().replace('\n', ' ')
        bullet_text = f"[{label}] {content}"
        if inline_source:
            # Append the inline source to the content of the bullet, or handle it as a separate field
            # For now, let's append it to the bullet text for simplicity and then process it later for DocSources
            bullet_text += f" (Source: {inline_source.strip()})"
            results.append((bullet_text, inline_source.strip()))
        else:
            results.append((bullet_text, None)) # No inline source
    return results


# Helper function to extract only Source lines from a text segment
def extract_sources_from_text_body(text_content):
    found_sources = set()
    # Find all lines starting with "Source:" followed by any characters until the end of the line
    source_lines = re.findall(r'^\s*Source:\s*(.+)', text_content, re.MULTILINE)
    for src in source_lines:
        found_sources.add(src.strip())
    return list(found_sources) # Return as a list of unique sources

# Unified processor
def process_text(text, source):
    swot_rows = []
    current_dimension = None

    # Match Dimension or SWOT Source lines
    blocks = re.split(
        r'(Dimension\s+\d+:\s*.*?|Strengths\s+Source:.*?|Weaknesses\s+Source:.*?|Opportunities\s+Source:.*?|Threats\s+Source:.*?|Source:.*?)\n',
        text
    )

    current_swot_type = None

    for i in range(1, len(blocks), 2):
        header = blocks[i].strip()
        body = blocks[i + 1] if i + 1 < len(blocks) else ""

        dim_match = re.match(r'Dimension\s+\d+:\s*(.*?)(Strengths|Weaknesses|Opportunities|Threats)?\s+Source:.*', header)
        swot_match = re.match(r'(Strengths|Weaknesses|Opportunities|Threats)\s+Source:\s*(.+)', header)

        # Initialize swot_source for the current block to the overall file source
        swot_source = source

        if dim_match:
            current_dimension = dim_match.group(1).strip()
            current_swot_type = dim_match.group(2).strip() if dim_match.group(2) else None
            swot_source_match = re.search(r'Source:\s*(.+)', header)
            if swot_source_match:
                swot_source = swot_source_match.group(1).strip()
        elif swot_match:
            current_swot_type = swot_match.group(1).strip()
            swot_source = swot_match.group(2).strip()
        else:
            continue

        if current_dimension and current_swot_type:
            # bullets now returns (bullet_text, inline_source) tuples
            extracted_bullets_and_sources = extract_bullets(body)
            bullets = [item[0] for item in extracted_bullets_and_sources]
            inline_sources_found = [item[1] for item in extracted_bullets_and_sources if item[1]]

            # Merge bullets into existing row if exists
            existing_row = next((r for r in swot_rows if r['Dimension'] == current_dimension and r['Source'] == source), None)
            if not existing_row:
                existing_row = {
                    'Source': source,
                    'Dimension': current_dimension,
                    'Strengths': '',
                    'Weakness': '',
                    'Opportunity': '',
                    'Threat': '',
                    'DocSources': []
                }
                swot_rows.append(existing_row)

            column = {
                'Strengths': 'Strengths',
                'Weaknesses': 'Weakness',
                'Opportunities': 'Opportunity',
                'Threats': 'Threat'
            }[current_swot_type]

            existing_content = existing_row[column]
            new_content = "\n\n".join(bullets)
            if existing_content:
                existing_row[column] += "\n\n" + new_content
            else:
                existing_row[column] = new_content

            # Add the block-level swot_source
            if swot_source and swot_source not in existing_row['DocSources']:
                existing_row['DocSources'].append(swot_source)

            # Add any inline sources found within the bullets
            for inline_src in inline_sources_found:
                if inline_src and inline_src not in existing_row['DocSources']:
                    existing_row['DocSources'].append(inline_src)


    # Deduplicate and format DocSources
    for row in swot_rows:
        docs = row.get('DocSources', [])
        row['DocSources'] = "; ".join(sorted(set(docs)))

    # 🧾 Print summary for inline format
    if swot_rows:
        summary = defaultdict(lambda: {
            'Strengths': 0, 'Weakness': 0, 'Opportunity': 0, 'Threat': 0, 'Docs': set()
        })

        for row in swot_rows:
            dim = row['Dimension']
            summary[dim]['Strengths'] += row['Strengths'].count('[')
            summary[dim]['Weakness'] += row['Weakness'].count('[')
            summary[dim]['Opportunity'] += row['Opportunity'].count('[')
            summary[dim]['Threat'] += row['Threat'].count('[')
            for doc in row.get('DocSources', '').split(';'):
                doc = doc.strip()
                if doc:
                    summary[dim]['Docs'].add(doc)

        all_docs = set()
        for dim, counts in summary.items():
            doc_list = sorted(counts['Docs'])
            all_docs.update(doc_list)
            print(
                f"🔹 {source} → {dim}: ✅ {counts['Strengths']} strengths, "
                f"⚠️ {counts['Weakness']} weaknesses, 🌱 {counts['Opportunity']} opps, 🔥 {counts['Threat']} threats "
                f"({len(doc_list)} sources)"
            )

        print(f"📊 {source} → Total unique document sources used: {len(all_docs)}")

    # Fallback to legacy format (unchanged as the inline source handling is for the primary processing path)
    if not swot_rows:
        print(f"\n📘 {source} — Fallback to LEGACY FORMAT")
        dimension_blocks = re.findall(r'(Dimension\s+\d+:\s+.*?)(?=Dimension\s+\d+:|\Z)', text, re.DOTALL)

        summary = defaultdict(lambda: {
            'Strengths': 0, 'Weakness': 0, 'Opportunity': 0, 'Threat': 0, 'Docs': set()
        })

        for idx, block in enumerate(dimension_blocks, start=1):
            dimension_header = re.search(r'Dimension\s+\d+:\s+(.*)', block)
            if not dimension_header:
                print(f"[Warning] Skipping block {idx} — couldn't extract dimension name.")
                continue

            dimension_name = dimension_header.group(1).strip()
            doc_sources = set()

            # Capture source lines in block
            source_lines = re.findall(r'^\s*Source:\s*(.+)', block, re.MULTILINE)
            for src in source_lines:
                doc_sources.add(src.strip())

            def get_section(section_title):
                # This needs to be adapted for the legacy path if inline sources are possible here too
                pattern = rf'{section_title}:\s*(.*?)(?=Strengths:|Weaknesses:|Opportunities:|Threats:|Dimension\s+\d+:|\Z)'
                match = re.search(pattern, block, re.DOTALL)
                # For legacy, extract_bullets won't handle inline sources, so we just use its original output
                return [item[0] for item in extract_bullets(match.group(1))] if match else []

            strengths = get_section("Strengths")
            weaknesses = get_section("Weaknesses")
            opportunities = get_section("Opportunities")
            threats = get_section("Threats")

            swot_rows.append({
                'Source': source,
                'Dimension': dimension_name,
                'Strengths': "\n\n".join(strengths),
                'Weakness': "\n\n".join(weaknesses),
                'Opportunity': "\n\n".join(opportunities),
                'Threat': "\n\n".join(threats),
                'DocSources': "; ".join(sorted(doc_sources)) if doc_sources else ''
            })

            summary[dimension_name]['Strengths'] += len(strengths)
            summary[dimension_name]['Weakness'] += len(weaknesses)
            summary[dimension_name]['Opportunity'] += len(opportunities)
            summary[dimension_name]['Threat'] += len(threats)
            summary[dimension_name]['Docs'].update(doc_sources)

        all_docs = set()
        for dim, counts in summary.items():
            doc_list = sorted(counts['Docs'])
            all_docs.update(doc_list)
            print(
                f"🔹 {source} → {dim}: ✅ {counts['Strengths']} strengths, "
                f"⚠️ {counts['Weakness']} weaknesses, 🌱 {counts['Opportunity']} opps, 🔥 {counts['Threat']} threats "
                f"({len(doc_list)} sources)"
            )

        print(f"📊 {source} → Total unique document sources used: {len(all_docs)}")
    
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
    fieldnames = ['Source', 'Dimension', 'Strengths', 'Weakness', 'Opportunity', 'Threat', 'DocSources']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in swot_matrix:
        writer.writerow(row)

print(f"\n✅ Done! Combined CSV file '{csv_filename}' created with {len(swot_matrix)} total row(s).")