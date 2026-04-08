import streamlit as st
import pandas as pd
import numpy as np
import io
from itertools import combinations

# statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# reportlab for PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Field Trial ANOVA",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Field Trial ANOVA Analyzer")
st.markdown(
    "Supports **Augmented RCBD**, **RCBD**, and **Alpha Lattice** designs. "
    "Calculates ANOVA table, CV, R², and BLUPs."
)

# ─────────────────────────────────────────────────────────────
# Sidebar – design selector & column mapper
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    design = st.selectbox(
        "Experimental Design",
        ["RCBD", "Augmented RCBD", "Alpha Lattice"],
    )
    st.markdown("---")
    st.markdown(
        "**Column requirements**\n"
        "- **RCBD**: `rep`, `genotype`, `yield`\n"
        "- **Augmented RCBD**: `rep`, `genotype`, `check`, `yield`\n"
        "- **Alpha Lattice**: `rep`, `block`, `genotype`, `yield`"
    )

# ─────────────────────────────────────────────────────────────
# File uploader
# ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin the analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("📋 Data Preview")
st.dataframe(df.head(20), use_container_width=True)
st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────
# Column mapping
# ─────────────────────────────────────────────────────────────
st.subheader("🗂️ Map Your Columns")
cols = df.columns.tolist()

col1, col2, col3, col4 = st.columns(4)
with col1:
    rep_col = st.selectbox("Replication column", cols, key="rep")
with col2:
    gen_col = st.selectbox("Genotype column", cols, key="gen")
with col3:
    yld_col = st.selectbox("Yield (response) column", cols, key="yld")
with col4:
    if design == "Alpha Lattice":
        blk_col = st.selectbox("Incomplete block column", cols, key="blk")
    elif design == "Augmented RCBD":
        chk_col = st.selectbox("Check indicator column (0/1 or name)", cols, key="chk")

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def cv(residuals, grand_mean):
    """
    Coefficient of Variation (%).
    FIXED #1: Handle zero or near-zero grand_mean to avoid division errors.
    """
    if grand_mean == 0 or abs(grand_mean) < 1e-10:
        st.warning("Grand mean is zero or near-zero. CV cannot be meaningfully calculated.")
        return 0.0
    
    rmse = np.sqrt(np.mean(residuals**2))
    return (rmse / grand_mean) * 100


def compute_blups_mixed(data, formula_fixed, formula_random_col, response):
    """
    Approximate BLUPs using statsmodels MixedLM.
    Random effect = genotype (or block-within-rep for alpha lattice).
    
    FIXED #2: Return a tuple (blup_series, error_msg) for consistent type handling.
    Returns (Series, None) on success or (None, error_msg) on failure.
    """
    try:
        model = smf.mixedlm(
            f"{response} ~ {formula_fixed}",
            data,
            groups=data[formula_random_col],
        )
        result = model.fit(reml=True, method="lbfgs")
        blups = result.random_effects
        blup_series = pd.Series({k: v.iloc[0] for k, v in blups.items()}, name="BLUP")
        return blup_series, None
    except Exception as e:
        return None, str(e)


def anova_table_from_model(model):
    """Extract type-I ANOVA table."""
    return anova_lm(model, typ=1)


def validate_required_columns(df, required_cols, design_name):
    """
    FIXED #3: Validate that all required columns exist in the dataframe.
    Raises an error with clear messaging if any are missing.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(
            f"**{design_name}** requires the following columns: {', '.join(required_cols)}\n\n"
            f"Missing columns: {', '.join(missing)}\n\n"
            f"Available columns: {', '.join(df.columns.tolist())}"
        )
        st.stop()


# ─────────────────────────────────────────────────────────────
# Run Analysis button
# ─────────────────────────────────────────────────────────────
run = st.button("▶️ Run Analysis", type="primary")

if not run:
    st.stop()

# Clean data
try:
    df[yld_col] = pd.to_numeric(df[yld_col], errors="coerce")
    df = df.dropna(subset=[yld_col])
    df[rep_col] = df[rep_col].astype(str)
    df[gen_col] = df[gen_col].astype(str)
    grand_mean = df[yld_col].mean()
except Exception as e:
    st.error(f"Data preparation error: {e}")
    st.stop()

results_store = {}   # will hold everything for PDF

# ─────────────────────────────────────────────────────────────
# ANALYSIS BRANCHES
# ─────────────────────────────────────────────────────────────

# ── RCBD ─��────────────────────────────────────────────────────
if design == "RCBD":
    st.header("📊 RCBD Analysis")
    
    # FIXED #3: Validate required columns
    validate_required_columns(df, [rep_col, gen_col, yld_col], "RCBD")
    
    try:
        formula = f"{yld_col} ~ C({rep_col}) + C({gen_col})"
        ols_model = smf.ols(formula, data=df).fit()
        aov = anova_table_from_model(ols_model)

        residuals = ols_model.resid
        cv_val = cv(residuals, grand_mean)
        r2_val = ols_model.rsquared

        st.subheader("ANOVA Table")
        st.dataframe(aov.style.format("{:.4f}"), use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Grand Mean", f"{grand_mean:.4f}")
        m2.metric("CV (%)", f"{cv_val:.2f}")
        m3.metric("R²", f"{r2_val:.4f}")

        # BLUPs via MixedLM (genotype = random)
        st.subheader("BLUPs – Genotype Random Effects")
        blup_series, blup_error = compute_blups_mixed(df, f"C({rep_col})", gen_col, yld_col)
        
        # FIXED #2: Check if blup_series is not None
        if blup_series is not None:
            blup_df = blup_series.reset_index()
            blup_df.columns = ["Genotype", "BLUP"]
            blup_df = blup_df.sort_values("BLUP", ascending=False).reset_index(drop=True)
            st.dataframe(blup_df, use_container_width=True)
        else:
            blup_df = pd.DataFrame()
            st.warning(f"BLUPs could not be computed: {blup_error}")

        results_store = dict(
            design=design, aov=aov, cv=cv_val, r2=r2_val,
            grand_mean=grand_mean, blup_df=blup_df,
        )

    except Exception as e:
        st.error(f"RCBD analysis failed: {e}")
        st.stop()

# ── Augmented RCBD ────────────────────────────────────────────
elif design == "Augmented RCBD":
    st.header("📊 Augmented RCBD Analysis")
    
    # FIXED #3: Validate required columns
    validate_required_columns(df, [rep_col, gen_col, chk_col, yld_col], "Augmented RCBD")
    
    try:
        df["is_check"] = df[chk_col].astype(str).str.lower().isin(["1", "yes", "true", "check"])
        checks = df[df["is_check"]]
        tests = df[~df["is_check"]]

        # ANOVA on checks only (balanced)
        if checks.empty:
            st.warning("No check entries detected. Verify the check column values.")
        formula_chk = f"{yld_col} ~ C({rep_col}) + C({gen_col})"
        ols_chk = smf.ols(formula_chk, data=checks).fit() if len(checks) > 0 else None

        # Full model treating all genotypes + rep
        formula_full = f"{yld_col} ~ C({rep_col}) + C({gen_col})"
        ols_full = smf.ols(formula_full, data=df).fit()
        aov_full = anova_table_from_model(ols_full)

        residuals = ols_full.resid
        cv_val = cv(residuals, grand_mean)
        r2_val = ols_full.rsquared

        st.subheader("ANOVA Table (Full – Checks + Tests)")
        st.dataframe(aov_full.style.format("{:.4f}"), use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Grand Mean", f"{grand_mean:.4f}")
        m2.metric("CV (%)", f"{cv_val:.2f}")
        m3.metric("R²", f"{r2_val:.4f}")

        # Adjusted means for test genotypes
        st.subheader("Adjusted Means (Test Genotypes)")
        pred = ols_full.get_prediction(df).summary_frame(alpha=0.05)
        df["adj_mean"] = pred["mean"]
        adj_test = (
            df[~df["is_check"]]
            .groupby(gen_col)["adj_mean"]
            .mean()
            .reset_index()
            .rename(columns={"adj_mean": "Adjusted Mean"})
            .sort_values("Adjusted Mean", ascending=False)
        )
        st.dataframe(adj_test, use_container_width=True)

        # BLUPs
        st.subheader("BLUPs – Genotype Random Effects")
        blup_series, blup_error = compute_blups_mixed(df, f"C({rep_col})", gen_col, yld_col)
        
        # FIXED #2: Check if blup_series is not None
        if blup_series is not None:
            blup_df = blup_series.reset_index()
            blup_df.columns = ["Genotype", "BLUP"]
            blup_df = blup_df.sort_values("BLUP", ascending=False).reset_index(drop=True)
            st.dataframe(blup_df, use_container_width=True)
        else:
            blup_df = pd.DataFrame()
            st.warning(f"BLUPs could not be estimated: {blup_error}")

        results_store = dict(
            design=design, aov=aov_full, cv=cv_val, r2=r2_val,
            grand_mean=grand_mean, blup_df=blup_df, adj_means=adj_test,
        )

    except Exception as e:
        st.error(f"Augmented RCBD analysis failed: {e}")
        st.stop()

# ── Alpha Lattice ─────────────────────────────────────────────
elif design == "Alpha Lattice":
    st.header("📊 Alpha Lattice Analysis")
    
    # FIXED #3: Validate required columns
    validate_required_columns(df, [rep_col, blk_col, gen_col, yld_col], "Alpha Lattice")
    
    try:
        df[blk_col] = df[blk_col].astype(str)
        # Nested incomplete block within rep
        df["rep_blk"] = df[rep_col] + ":" + df[blk_col]

        formula = f"{yld_col} ~ C({rep_col}) + C(rep_blk) + C({gen_col})"
        ols_model = smf.ols(formula, data=df).fit()
        aov = anova_table_from_model(ols_model)

        residuals = ols_model.resid
        cv_val = cv(residuals, grand_mean)
        r2_val = ols_model.rsquared

        st.subheader("ANOVA Table")
        st.dataframe(aov.style.format("{:.4f}"), use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Grand Mean", f"{grand_mean:.4f}")
        m2.metric("CV (%)", f"{cv_val:.2f}")
        m3.metric("R²", f"{r2_val:.4f}")

        # Efficiency vs RCBD
        formula_rcbd = f"{yld_col} ~ C({rep_col}) + C({gen_col})"
        ols_rcbd = smf.ols(formula_rcbd, data=df).fit()
        ms_e_alpha = np.mean(residuals**2)
        ms_e_rcbd = np.mean(ols_rcbd.resid**2)
        efficiency = (ms_e_rcbd / ms_e_alpha) * 100
        st.metric("Relative Efficiency vs RCBD (%)", f"{efficiency:.2f}")

        # BLUPs – genotype random, rep+block fixed
        st.subheader("BLUPs – Genotype Random Effects")
        blup_series, blup_error = compute_blups_mixed(
            df, f"C({rep_col}) + C(rep_blk)", gen_col, yld_col
        )
        
        # FIXED #2: Check if blup_series is not None
        if blup_series is not None:
            blup_df = blup_series.reset_index()
            blup_df.columns = ["Genotype", "BLUP"]
            blup_df = blup_df.sort_values("BLUP", ascending=False).reset_index(drop=True)
            st.dataframe(blup_df, use_container_width=True)
        else:
            blup_df = pd.DataFrame()
            st.warning(f"BLUPs could not be estimated: {blup_error}")

        results_store = dict(
            design=design, aov=aov, cv=cv_val, r2=r2_val,
            grand_mean=grand_mean, blup_df=blup_df, efficiency=efficiency,
        )

    except Exception as e:
        st.error(f"Alpha Lattice analysis failed: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────
# PDF Report Generation
# ─────────────────────────────────────────────────────────────

def df_to_reportlab_table(data_df, title=None):
    """Convert a pandas DataFrame to a ReportLab Table."""
    items = []
    if title:
        styles = getSampleStyleSheet()
        items.append(Paragraph(title, styles["Heading3"]))
        items.append(Spacer(1, 6))

    # Header + rows
    header = [str(c) for c in data_df.columns]
    rows = [header]
    for _, row in data_df.iterrows():
        rows.append([f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])

    tbl = Table(rows, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F8E9")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    items.append(tbl)
    return items


def build_pdf(results):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"],
        textColor=colors.HexColor("#1B5E20"), spaceAfter=10,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        textColor=colors.HexColor("#2E7D32"), spaceBefore=14, spaceAfter=6,
    )
    story = []

    # Cover
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Field Trial ANOVA Report", title_style))
    story.append(Paragraph(f"Design: {results['design']}", styles["Heading2"]))
    story.append(Spacer(1, 0.2*inch))

    # Summary stats
    story.append(Paragraph("Summary Statistics", h2_style))
    summary_data = [
        ["Statistic", "Value"],
        ["Grand Mean", f"{results['grand_mean']:.4f}"],
        ["CV (%)", f"{results['cv']:.2f}"],
        ["R-Square", f"{results['r2']:.4f}"],
    ]
    if "efficiency" in results:
        summary_data.append(["Relative Efficiency vs RCBD (%)", f"{results['efficiency']:.2f}"])
    summ_tbl = Table(summary_data)
    summ_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F8E9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(summ_tbl)
    story.append(Spacer(1, 0.2*inch))

    # ANOVA table
    story.append(Paragraph("ANOVA Table", h2_style))
    aov_display = results["aov"].reset_index()
    aov_display.columns = [str(c) for c in aov_display.columns]
    for items in df_to_reportlab_table(aov_display):
        story.append(items)
    story.append(Spacer(1, 0.2*inch))

    # Adjusted means (Augmented RCBD only)
    if "adj_means" in results and not results["adj_means"].empty:
        story.append(Paragraph("Adjusted Means – Test Genotypes", h2_style))
        for item in df_to_reportlab_table(results["adj_means"]):
            story.append(item)
        story.append(Spacer(1, 0.2*inch))

    # BLUPs
    if isinstance(results["blup_df"], pd.DataFrame) and not results["blup_df"].empty:
        story.append(PageBreak())
        story.append(Paragraph("BLUPs – Genotype Random Effects", h2_style))
        for item in df_to_reportlab_table(results["blup_df"]):
            story.append(item)

    # Footer note
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph(
        "BLUPs estimated via Restricted Maximum Likelihood (REML) using statsmodels MixedLM.",
        styles["Italic"],
    ))

    doc.build(story)
    buf.seek(0)
    return buf


st.markdown("---")
st.subheader("📥 Download Results")

pdf_buf = build_pdf(results_store)
st.download_button(
    label="⬇️ Download PDF Report",
    data=pdf_buf,
    file_name=f"ANOVA_{results_store['design'].replace(' ', '_')}_report.pdf",
    mime="application/pdf",
    type="primary",
)

st.success("✅ Analysis complete! Download your PDF report above.")