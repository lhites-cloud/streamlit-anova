cat > README.md << EOF
# 🌾 Field Trial ANOVA Analyzer

A Streamlit app for analyzing agricultural field trials using ANOVA.

**Supports:**
- RCBD (Randomized Complete Block Design)
- Augmented RCBD
- Alpha Lattice designs

**Features:**
- ANOVA table generation
- Coefficient of Variation (CV) calculation
- R² computation
- BLUP estimation via REML
- PDF report generation

## Installation

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Usage

1. Upload your CSV file
2. Map your columns (rep, genotype, yield, etc.)
3. Select experimental design
4. Run analysis and download PDF report
EOF
