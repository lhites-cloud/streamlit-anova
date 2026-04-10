import streamlit as st
import pandas as pd
import numpy as np
import io
from itertools import combinations

# statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

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
    "Calculates ANOVA table, CV, R², BLUPs, and **broad-sense heritability (H²)**."
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
    st.markdown("---")
    st.markdown(
        "**Heritability (H²)**\n\n"
        "Broad-sense H² estimated from ANOVA variance components:\n\n"
        "σ²g = (MS_g − MS_e) / r\n\n"
        "H² = σ²g / (σ²g + σ²e / r)\n\n"
        "Negative σ²g is set to 0 (no genetic signal detected)."
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
    """Coefficient of Variation (%)."""
    if grand_mean == 0 or abs(grand_mean) < 1e-10:
        st.warning("Grand mean is zero or near-zero. CV cannot be meaningfully calculated.")
        return 0.0
    rmse = np.sqrt(np.mean(residuals**2))
    return (rmse / grand_mean) * 100


def compute_heritability(ms_genotype, ms_error, n_reps):
    """
    Broad-sense heritability (H²) from ANOVA variance components.

    Components:
        σ²g = (MS_genotype - MS_error) / n_reps
        σ²e = MS_error
        H²  = σ²g / (σ²g + σ²e / n_reps)

    Negative σ²g is clamped to 0 (no detectable genetic variance).

    Returns dict with σ²g, σ²e, H², and a reliability flag.
    """
    sigma2_e = ms_error
    sigma2_g_raw = (ms_genotype - ms_error) / n_reps
    sigma2_g = max(0.0, sigma2_g_raw)

    if (sigma2_g + sigma2_e / n_reps) < 1e-12:
        h2 = 0.0
    else:
        h2 = sigma2_g / (sigma2_g + sigma2_e / n_reps)

    return {
        "sigma2_g": sigma2_g,
        "sigma2_g_raw": sigma2_g_raw,
        "sigma2_e": sigma2_e,
        "H2": h2,
        "negative_var_warning": sigma2_g_raw < 0,
    }


def display_heritability(h2_dict, design_note=""):
    """Render heritability metrics in Streamlit."""
    st.subheader("🧬 Broad-Sense Heritability (H²)")

    if design_note:
        st.caption(design_note)

    if h2_dict["negative_var_warning"]:
        st.warning(
            "⚠️ Raw σ²g was negative (MS_genotype < MS_error), indicating no detectable "
            "genetic variance above noise. σ²g has been set to 0 and H² = 0. "
            "This can occur with few genotypes, large error variance, or a true absence "
            "of genetic differences."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("H²", f"{h2_dict['H2']:.4f}")
    c2.metric("H² (%)", f"{h2_dict['H2'] * 100:.2f}")
    c3.metric("σ²g (genotypic var.)", f"{h2_dict['sigma2_g']:.4f}")
    c4.metric("σ²e (error var.)", f"{h2_dict['sigma2_e']:.4f}")

    # Qualitative interpretation
    h2 = h2_dict["H2"]
    if h2 >= 0.70:
        interp = "🟢 **High** heritability — phenotypic values are reliable proxies for genotypic values; selection is expected to be effective."
    elif h2 >= 0.40:
        interp = "🟡 **Moderate** heritability — some environmental influence; replicated testing recommended before selection decisions."
    else:
        interp = "🔴 **Low** heritability — trait is strongly influenced by environment; caution advised when selecting based on phenotype alone."

    st.info(interp)


def compute_blups_mixed(data, formula_fixed, formula_random_col, response):
    """
    Approximate BLUPs using statsmodels MixedLM.
    Returns (Series, error_msg).
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
    """Validate that all required columns exist."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(
            f"**{design_name}** requires the following columns: {', '.join(required_cols)}\n\n"
            f"Missing columns: {', '.join(missing)}\n\n"
            f"Available columns: {', '.join(df.columns.tolist())}"
        )
        st.stop()


def build_html_report(results):
    """Generate a styled HTML report."""
    h2_section = ""
    if "h2" in results:
        h2 = results["h2"]
        warn = (
            "<p style='color:#b45309;background:#fef3c7;padding:10px;border-radius:6px;'>"
            "⚠️ Raw σ²g was negative (MS_genotype &lt; MS_error). σ²g set to 0; H² = 0.</p>"
            if h2["negative_var_warning"] else ""
        )
        h2_section = f"""
        <div class="section">
            <h2>🧬 Broad-Sense Heritability (H²)</h2>
            {warn}
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">H²</div>
                    <div class="metric-value">{h2['H2']:.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">H² (%)</div>
                    <div class="metric-value">{h2['H2']*100:.2f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">σ²g (genotypic var.)</div>
                    <div class="metric-value">{h2['sigma2_g']:.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">σ²e (error var.)</div>
                    <div class="metric-value">{h2['sigma2_e']:.4f}</div>
                </div>
            </div>
            <p style="font-size:0.95em;color:#555;">
                <strong>Formula:</strong> σ²g = (MS_genotype − MS_error) / r &nbsp;|&nbsp;
                H² = σ²g / (σ²g + σ²e / r)
            </p>
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Field Trial ANOVA Report</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1100px;
                margin: 0 auto;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            header {{
                background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
            header p {{ font-size: 1.1em; opacity: 0.9; }}
            .content {{ padding: 40px; }}
            h2 {{
                color: #1B5E20;
                border-bottom: 3px solid #2E7D32;
                padding-bottom: 10px;
                margin-top: 30px;
                margin-bottom: 20px;
                font-size: 1.8em;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 20px;
                margin: 20px 0 30px 0;
            }}
            .metric {{
                background: linear-gradient(135deg, #F1F8E9 0%, #E8F5E9 100%);
                border-left: 5px solid #2E7D32;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}
            .metric-label {{
                font-size: 0.9em;
                color: #555;
                margin-bottom: 8px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .metric-value {{
                font-size: 1.8em;
                font-weight: bold;
                color: #1B5E20;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                border-radius: 8px;
                overflow: hidden;
            }}
            thead {{ background-color: #2E7D32; color: white; }}
            th {{ padding: 15px; text-align: left; font-weight: 600; letter-spacing: 0.5px; }}
            td {{ padding: 12px 15px; border-bottom: 1px solid #e0e0e0; }}
            tbody tr:nth-child(even) {{ background-color: #F1F8E9; }}
            tbody tr:hover {{ background-color: #E8F5E9; }}
            footer {{
                background-color: #f5f5f5;
                padding: 20px 40px;
                text-align: center;
                color: #666;
                font-size: 0.9em;
                border-top: 1px solid #e0e0e0;
            }}
            .section {{ page-break-inside: avoid; margin-bottom: 30px; }}
            @media print {{ body {{ background: white; }} .container {{ box-shadow: none; }} }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🌾 Field Trial ANOVA Report</h1>
                <p><strong>Design:</strong> {results['design']}</p>
            </header>
            <div class="content">
                <div class="section">
                    <h2>Summary Statistics</h2>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Grand Mean</div>
                            <div class="metric-value">{results['grand_mean']:.4f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">CV (%)</div>
                            <div class="metric-value">{results['cv']:.2f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">R-Square</div>
                            <div class="metric-value">{results['r2']:.4f}</div>
                        </div>
    """

    if "efficiency" in results:
        html_content += f"""
                        <div class="metric">
                            <div class="metric-label">Relative Efficiency vs RCBD (%)</div>
                            <div class="metric-value">{results['efficiency']:.2f}</div>
                        </div>
        """

    html_content += """
                    </div>
                </div>
    """

    # Heritability section
    html_content += h2_section

    # ANOVA table
    html_content += """
                <div class="section">
                    <h2>ANOVA Table</h2>
    """
    aov_html = results["aov"].to_html(float_format=lambda x: f"{x:.6f}", classes="table")
    html_content += aov_html

    # Adjusted means (Augmented RCBD only)
    if "adj_means" in results and not results["adj_means"].empty:
        html_content += """
                </div>
                <div class="section">
                    <h2>Adjusted Means – Test Genotypes</h2>
        """
        adj_html = results["adj_means"].to_html(
            float_format=lambda x: f"{x:.4f}", index=False, classes="table"
        )
        html_content += adj_html

    # BLUPs
    if isinstance(results["blup_df"], pd.DataFrame) and not results["blup_df"].empty:
        html_content += """
                </div>
                <div class="section">
                    <h2>BLUPs – Genotype Random Effects</h2>
        """
        blup_html = results["blup_df"].to_html(
            float_format=lambda x: f"{x:.4f}", index=False, classes="table"
        )
        html_content += blup_html

    html_content += """
                </div>
            </div>
            <footer>
                <p><strong>Heritability note:</strong> H² = σ²g / (σ²g + σ²e/r), estimated via ANOVA variance components (Hallauer &amp; Miranda, 1988). Negative σ²g clamped to 0.</p>
                <p><strong>BLUPs note:</strong> Estimated via Restricted Maximum Likelihood (REML) using statsmodels MixedLM.</p>
                <p>Generated by Field Trial ANOVA Analyzer</p>
            </footer>
        </div>
    </body>
    </html>
    """
    return html_content


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
    n_reps = df[rep_col].nunique()
except Exception as e:
    st.error(f"Data preparation error: {e}")
    st.stop()

results_store = {}

# ─────────────────────────────────────────────────────────────
# ANALYSIS BRANCHES
# ─────────────────────────────────────────────────────────────

# ── RCBD ─────────────────────────────────────────────────────
if design == "RCBD":
    st.header("📊 RCBD Analysis")
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

        # ── Heritability ──────────────────────────────────────
        # Identify genotype and residual rows in the ANOVA table
        gen_row = [r for r in aov.index if gen_col.lower() in r.lower()]
        res_row = [r for r in aov.index if "residual" in r.lower()]

        if gen_row and res_row:
            ms_g = aov.loc[gen_row[0], "mean_sq"]
            ms_e = aov.loc[res_row[0], "mean_sq"]
            h2_dict = compute_heritability(ms_g, ms_e, n_reps)
            display_heritability(
                h2_dict,
                design_note=f"Using MS_genotype = {ms_g:.4f}, MS_error = {ms_e:.4f}, r = {n_reps} reps."
            )
        else:
            h2_dict = None
            st.warning("Could not locate genotype and residual rows in ANOVA table for H² computation.")

        st.subheader("BLUPs – Genotype Random Effects")
        blup_series, blup_error = compute_blups_mixed(df, f"C({rep_col})", gen_col, yld_col)

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
            h2=h2_dict,
        )

    except Exception as e:
        st.error(f"RCBD analysis failed: {e}")
        st.stop()

# ── Augmented RCBD ────────────────────────────────────────────
elif design == "Augmented RCBD":
    st.header("📊 Augmented RCBD Analysis")
    validate_required_columns(df, [rep_col, gen_col, chk_col, yld_col], "Augmented RCBD")

    try:
        df["is_check"] = df[chk_col].astype(str).str.lower().isin(["1", "yes", "true", "check"])
        checks = df[df["is_check"]]
        tests  = df[~df["is_check"]]

        if checks.empty:
            st.warning("No check entries detected. Verify the check column values.")

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

        # ── Heritability (test genotypes only) ────────────────
        # For Augmented RCBD heritability is most meaningful for
        # test genotypes where each appears once per design.
        # We refit on test entries only for an unbiased MS_g estimate.
        h2_dict = None
        if not tests.empty and tests[gen_col].nunique() > 1:
            try:
                formula_test = f"{yld_col} ~ C({rep_col}) + C({gen_col})"
                ols_test = smf.ols(formula_test, data=tests).fit()
                aov_test = anova_table_from_model(ols_test)

                gen_row = [r for r in aov_test.index if gen_col.lower() in r.lower()]
                res_row = [r for r in aov_test.index if "residual" in r.lower()]

                if gen_row and res_row:
                    ms_g = aov_test.loc[gen_row[0], "mean_sq"]
                    ms_e = aov_test.loc[res_row[0], "mean_sq"]
                    n_reps_test = tests[rep_col].nunique()
                    h2_dict = compute_heritability(ms_g, ms_e, n_reps_test)
                    display_heritability(
                        h2_dict,
                        design_note=(
                            f"Estimated from **test genotypes only** (checks excluded). "
                            f"MS_genotype = {ms_g:.4f}, MS_error = {ms_e:.4f}, r = {n_reps_test} reps."
                        )
                    )
            except Exception as he:
                st.warning(f"H² could not be estimated for test genotypes: {he}")
        else:
            st.info("H² not estimated: fewer than 2 test genotypes found.")

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

        st.subheader("BLUPs – Genotype Random Effects")
        blup_series, blup_error = compute_blups_mixed(df, f"C({rep_col})", gen_col, yld_col)

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
            h2=h2_dict,
        )

    except Exception as e:
        st.error(f"Augmented RCBD analysis failed: {e}")
        st.stop()

# ── Alpha Lattice ─────────────────────────────────────────────
elif design == "Alpha Lattice":
    st.header("📊 Alpha Lattice Analysis")
    validate_required_columns(df, [rep_col, blk_col, gen_col, yld_col], "Alpha Lattice")

    try:
        df[blk_col] = df[blk_col].astype(str)
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

        formula_rcbd = f"{yld_col} ~ C({rep_col}) + C({gen_col})"
        ols_rcbd = smf.ols(formula_rcbd, data=df).fit()
        ms_e_alpha = np.mean(residuals**2)
        ms_e_rcbd  = np.mean(ols_rcbd.resid**2)
        efficiency = (ms_e_rcbd / ms_e_alpha) * 100
        st.metric("Relative Efficiency vs RCBD (%)", f"{efficiency:.2f}")

        # ── Heritability (using effective error from Alpha Lattice model) ──
        gen_row = [r for r in aov.index if gen_col.lower() in r.lower()]
        res_row = [r for r in aov.index if "residual" in r.lower()]
        h2_dict = None

        if gen_row and res_row:
            ms_g = aov.loc[gen_row[0], "mean_sq"]
            ms_e = aov.loc[res_row[0], "mean_sq"]   # effective error after block adjustment
            h2_dict = compute_heritability(ms_g, ms_e, n_reps)
            display_heritability(
                h2_dict,
                design_note=(
                    f"MS_error is the **block-adjusted** residual mean square from the Alpha Lattice model. "
                    f"MS_genotype = {ms_g:.4f}, MS_error (adjusted) = {ms_e:.4f}, r = {n_reps} reps."
                )
        )
        else:
            st.warning("Could not locate genotype and residual rows for H² computation.")

        st.subheader("BLUPs – Genotype Random Effects")
        blup_series, blup_error = compute_blups_mixed(
            df, f"C({rep_col}) + C(rep_blk)", gen_col, yld_col
        )

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
            h2=h2_dict,
        )

    except Exception as e:
        st.error(f"Alpha Lattice analysis failed: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────
# HTML Report Download
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📥 Download Results")

html_report = build_html_report(results_store)
st.download_button(
    label="⬇️ Download HTML Report",
    data=html_report,
    file_name=f"ANOVA_{results_store['design'].replace(' ', '_')}_report.html",
    mime="text/html",
    type="primary",
)

st.success("✅ Analysis complete! Download your HTML report above.")
