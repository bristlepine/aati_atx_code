# README.md – AATI Project

# AATI SWOT Visualization & Analysis

This project supports the African Agricultural Transformation Initiative (AATI) by analyzing qualitative SWOT (Strengths, Weaknesses, Opportunities, Threats) data collected across multiple delivery models. It uses Python to codify, process, and visualize the data, helping distill cross-country insights and thematic patterns across eight ATx dimensions.

## 🔍 Project Purpose

To analyze the ATx comparative matrix and produce insightful figures such as:
- Bar charts by dimension and SWOT category
- Word clouds of common terms in each SWOT type
- Radar plots comparing models across dimensions
- Heatmaps of SWOT frequency by country/dimension

## 🗃️ File Structure

```
├── aati_figures.py       # Main script for data processing and visualization
├── Desk Research Matrix - Combined Matrix.csv      # Raw Matrix
├── codified_matrix.csv   # Codified Matrix
├── README.md             # This file
└── outputs/              # Auto-generated figures and tables
```

## ⚙️ Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Dependencies include:
- pandas
- matplotlib
- seaborn
- wordcloud
- nltk  *(for basic tokenization)*
- scikit-learn *(optional, for clustering or topic modeling)*
- spacy *(optional, for advanced NLP)*

If using word clouds, download NLTK stopwords:
```bash
python -m nltk.downloader stopwords
```

- If using Gemini for codification (instead of local NLP), make sure your API key is stored as an environment variable: `GEMINI_API_KEY`


## ▶️ Usage

Run the analysis:
```bash
python aati_figures.py
```

The script will:
- Load the raw matrix from `Desk Research Matrix - Combined Matrix.csv`
- Automatically codify and expand individual SWOT points using NLP (via spaCy or Gemini if available)
- Save the result to `codified_matrix.csv`
- Clean and tokenize SWOT text fields
- Generate visualizations saved to the `outputs/` folder

## 📈 Output Examples

- outputs/strengths_by_dimension_bar.png
- outputs/governance_wordcloud.png
- outputs/swot_heatmap.png
- outputs/radar_governance_vs_finance.png

## 📝 Notes

- The raw matrix should include: `Source`, `Dimension`, `Strengths`, `Weakness`, `Opportunity`, `Threat`.
- The script splits multiline SWOT fields into individual rows, preserving metadata, and stores the codified output in `codified_matrix.csv`.

## 📮 Contact

For questions or contributions, contact the AATI analysis team or open an issue in the repository.
