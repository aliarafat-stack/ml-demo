import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.data_loader import DATA_PATH, CATEGORICAL_COLS, NUMERIC_COLS, load_raw_data, get_encoded_data, get_train_test, get_onehot_train_test

st.set_page_config(page_title="Preprocessing", page_icon="🔧", layout="wide")
st.title("Data Preprocessing Pipeline")
st.markdown(
    "Before feeding data to a machine learning model, we need to **clean**, "
    "**transform**, and **split** it. This page walks through every step we applied, "
    "shows the data before and after, and explains *why*."
)
st.markdown("---")

tab_missing, tab_encoding, tab_scaling, tab_split, tab_reference = st.tabs([
    "Missing Data",
    "Categorical Encoding",
    "Feature Scaling",
    "Train / Test Split",
    "Imputation Reference Guide",
])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Missing Data
# ═══════════════════════════════════════════════════════════════════════════════
with tab_missing:
    st.subheader("Step 1: Handling Missing Data")

    st.markdown("#### Why this matters")
    st.markdown(
        "Most ML algorithms cannot handle missing values. If we leave blanks in, "
        "the model will either crash or silently ignore those rows, losing valuable data. "
        "Proper imputation preserves sample size and avoids bias."
    )

    st.markdown("---")
    st.markdown("#### What we found")

    raw_csv = pd.read_csv(DATA_PATH)

    total_charges_blanks = raw_csv[raw_csv["TotalCharges"].str.strip() == ""]
    n_blanks = len(total_charges_blanks)

    col_info, col_sample = st.columns([1, 2])
    with col_info:
        st.metric("Rows with missing TotalCharges", n_blanks)
        st.metric("Total rows in dataset", f"{len(raw_csv):,}")
        st.metric("% missing", f"{n_blanks / len(raw_csv):.2%}")
    with col_sample:
        st.markdown("**Rows where `TotalCharges` is blank:**")
        display_cols = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
        st.dataframe(total_charges_blanks[display_cols].head(10), use_container_width=True)
        st.caption(
            "Notice: all these customers have tenure = 0 — they just signed up "
            "and haven't been billed yet, so TotalCharges is an empty string."
        )

    st.markdown("---")
    st.markdown("#### What we did")

    st.markdown(
        "We used **Median Imputation** — replacing blanks with the median of `TotalCharges`."
    )

    before_col, arrow_col, after_col = st.columns([5, 1, 5])
    with before_col:
        st.markdown("**Before** (raw CSV)")
        sample_before = total_charges_blanks[display_cols].head(5).copy()
        st.dataframe(sample_before, use_container_width=True)

    with arrow_col:
        st.markdown("")
        st.markdown("")
        st.markdown("### →")

    with after_col:
        st.markdown("**After** (median filled)")
        cleaned = load_raw_data()
        sample_after = cleaned.loc[total_charges_blanks.index[:5], ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
        st.dataframe(sample_after, use_container_width=True)

    median_val = pd.to_numeric(raw_csv["TotalCharges"], errors="coerce").median()
    st.info(f"Median value used for filling: **${median_val:,.2f}**")

    st.markdown("---")
    st.markdown("#### Why Median instead of Mean?")
    st.markdown(
        """
        `TotalCharges` is **right-skewed** — a few long-tenure customers have very high
        totals (> $8,000) while most are much lower. The mean gets pulled up by those outliers.
        The median is robust to skew and gives a more representative "typical" value.

        | Statistic | Value |
        |---|---|
        | Mean | ${:,.2f} |
        | Median | ${:,.2f} |
        | Min | ${:,.2f} |
        | Max | ${:,.2f} |
        """.format(
            pd.to_numeric(raw_csv["TotalCharges"], errors="coerce").mean(),
            median_val,
            pd.to_numeric(raw_csv["TotalCharges"], errors="coerce").min(),
            pd.to_numeric(raw_csv["TotalCharges"], errors="coerce").max(),
        )
    )

    st.markdown("---")
    st.markdown("#### What else could we have done?")
    st.markdown(
        """
        - **Fill with 0:** Since these are brand-new customers (tenure=0), filling with 0
          is arguably the most logical choice for this specific column.
        - **Drop the rows:** Only 11 rows out of 7,043 (0.16%) — dropping them would have
          minimal impact, but we'd lose data unnecessarily.
        - **KNN Imputation:** Use similar customers' TotalCharges to estimate the missing value.
          Overkill for 11 rows but powerful when missingness is higher.
        - **Regression Imputation:** Predict TotalCharges from tenure and MonthlyCharges
          (they're highly correlated). Again, overkill here but useful in general.
        """
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Categorical Encoding
# ═══════════════════════════════════════════════════════════════════════════════
with tab_encoding:
    st.subheader("Step 2: Encoding Categorical Features")

    st.markdown("#### Why this matters")
    st.markdown(
        "ML models work with numbers, not text. A column like `Contract` with values "
        '"Month-to-month", "One year", "Two year" needs to be converted to numbers. '
        "The encoding method matters because it affects how the model interprets relationships."
    )

    st.markdown("---")
    st.markdown("#### What we did — two strategies for two model types")
    st.markdown(
        """
        We use **different encodings for different models**, because each model type
        interprets numbers differently:

        | Model | Encoding Used | Why |
        |---|---|---|
        | **Random Forest, XGBoost** | Label Encoding (integer per category) | Tree models split on thresholds — they don't care about magnitude |
        | **Logistic Regression** | One-Hot Encoding (binary column per category) | Linear models treat numbers as having magnitude — One-Hot avoids false ordinal relationships |
        """
    )

    st.markdown("##### Label Encoding (for tree-based models)")
    st.markdown(
        "Each unique text value gets a unique integer."
    )

    raw_df = load_raw_data()
    enc_df, encoders = get_encoded_data()

    selected_cat = st.selectbox(
        "Pick a categorical column to see before/after",
        CATEGORICAL_COLS,
        index=CATEGORICAL_COLS.index("Contract"),
    )

    before_enc, arrow_enc, after_enc = st.columns([5, 1, 5])
    with before_enc:
        st.markdown(f"**Before** — `{selected_cat}` (text)")
        sample_idx = raw_df.head(10).index
        before_sample = raw_df.loc[sample_idx, ["customerID", selected_cat]].copy()
        st.dataframe(before_sample, use_container_width=True)

    with arrow_enc:
        st.markdown("")
        st.markdown("")
        st.markdown("### →")

    with after_enc:
        st.markdown(f"**After** — `{selected_cat}` (integer)")
        after_sample = enc_df.loc[sample_idx, [selected_cat]].copy()
        after_sample.insert(0, "customerID", raw_df.loc[sample_idx, "customerID"].values)
        st.dataframe(after_sample, use_container_width=True)

    le = encoders[selected_cat]
    mapping_df = pd.DataFrame({
        "Original Value": le.classes_,
        "Encoded Value": range(len(le.classes_)),
    })
    st.markdown(f"**Mapping for `{selected_cat}`:**")
    st.dataframe(mapping_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### All 15 categorical columns encoded")
    summary_data = []
    for col in CATEGORICAL_COLS:
        le_col = encoders[col]
        summary_data.append({
            "Column": col,
            "Unique Values": len(le_col.classes_),
            "Classes": ", ".join(le_col.classes_),
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("##### One-Hot Encoding (for Logistic Regression)")
    st.markdown(
        "For Logistic Regression, we use **One-Hot Encoding** via `pd.get_dummies(drop_first=True)`. "
        "Each category becomes its own binary (0/1) column, avoiding false ordinal relationships. "
        "We drop the first category per feature to prevent multicollinearity (the \"dummy variable trap\")."
    )

    X_train_oh, _, _, _, feature_cols_oh = get_onehot_train_test()

    oh_before, oh_arrow, oh_after = st.columns([5, 1, 5])
    with oh_before:
        st.markdown(f"**Before** — `Contract` (text)")
        oh_sample = raw_df.head(6)[["customerID", "Contract"]].copy()
        st.dataframe(oh_sample, use_container_width=True)

    with oh_arrow:
        st.markdown("")
        st.markdown("")
        st.markdown("### →")

    with oh_after:
        st.markdown("**After** — One-Hot columns")
        oh_cols = [c for c in feature_cols_oh if c.startswith("Contract_")]
        oh_display = X_train_oh.head(6)[oh_cols].copy()
        oh_display.index = range(1, len(oh_display) + 1)
        st.dataframe(oh_display, use_container_width=True)

    st.markdown(
        f"One-Hot Encoding expands our {len(CATEGORICAL_COLS)} categorical columns "
        f"into **{len([c for c in feature_cols_oh if c not in NUMERIC_COLS])} binary columns** "
        f"(total features: **{len(feature_cols_oh)}** vs **{len(CATEGORICAL_COLS) + len(NUMERIC_COLS)}** "
        f"with Label Encoding)."
    )

    st.markdown("---")
    st.markdown("#### Why two encodings?")
    st.markdown(
        """
        We chose **Label Encoding for tree-based models** (Random Forest, XGBoost) because
        they split on thresholds (e.g., "is Contract < 1?") — the integer mapping works
        perfectly and they don't assume ordering between numbers.

        We chose **One-Hot Encoding for Logistic Regression** because it's a linear model
        that interprets feature values as having magnitude. With Label Encoding, the model
        might treat "Month-to-month = 2" as "twice" something compared to "DSL = 0", which
        is meaningless. One-Hot Encoding eliminates this problem entirely.

        **This dual-encoding approach gives each model the representation it works best with.**
        """
    )

    st.markdown("---")
    st.markdown("#### Other encoding alternatives")

    st.markdown(
        """
        | Technique | How it works | Best for |
        |---|---|---|
        | **Ordinal Encoding** | Like Label Encoding but you manually set the order (e.g., Month-to-month=0, One year=1, Two year=2) | Features with a natural order |
        | **Target Encoding** | Replace each category with the mean of the target (churn rate) for that category | High-cardinality features (many unique values) |
        | **Binary Encoding** | Converts integers to binary then splits into columns | Reduces dimensionality vs One-Hot for high-cardinality |
        """
    )

    st.success(
        "**Our approach:** Using the optimal encoding per model type is a best practice "
        "in production ML pipelines. It adds a small amount of complexity but ensures each "
        "model sees data in the format it handles best."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Feature Scaling
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scaling:
    st.subheader("Step 3: Feature Scaling (StandardScaler)")

    st.markdown("#### Why this matters")
    st.markdown(
        "Some algorithms (Logistic Regression, SGDClassifier, KNN, SVM) are sensitive to "
        "the **magnitude** of features. If `tenure` ranges 0–72 but `TotalCharges` ranges "
        "18–8,685, the model will overweight TotalCharges simply because its numbers are bigger. "
        "Scaling puts all features on the same playing field."
    )

    st.markdown("---")
    st.markdown("#### Standardization vs Normalization — they are NOT the same")
    st.markdown(
        """
        These two terms are often confused. Here's the difference:

        | | **Standardization** (what we did) | **Normalization** |
        |---|---|---|
        | **Goal** | Center data around 0 with unit variance | Squeeze data into a fixed range (usually 0–1) |
        | **Formula** | (x - mean) / std | (x - min) / (max - min) |
        | **Result** | Mean = 0, Std = 1. Values can be negative or > 1 | All values between 0 and 1 |
        | **Sensitive to outliers?** | Less sensitive (uses mean/std) | Very sensitive (one outlier stretches the whole scale) |
        | **Best for** | Algorithms that assume normally distributed features (Logistic Regression, SVM) | Neural networks, image pixel values, algorithms that need bounded input |

        We used **Standardization (StandardScaler)** because our models (SGDClassifier, Logistic Regression)
        use gradient descent, which works best when features are centered around zero with comparable spread.
        """
    )

    st.markdown("---")
    st.markdown("#### What we did")
    st.markdown(
        "**StandardScaler** transforms each feature to have **mean = 0** and **standard deviation = 1**."
    )
    st.latex(r"x_{\text{scaled}} = \frac{x - \mu}{\sigma}")

    enc_df_full, _ = get_encoded_data()
    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    sample_rows = enc_df_full[feature_cols].head(8)

    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(enc_df_full[feature_cols])
    scaled_sample = pd.DataFrame(
        all_scaled[:8],
        columns=feature_cols,
        index=sample_rows.index,
    )

    show_cols = st.multiselect(
        "Select columns to compare",
        feature_cols,
        default=["tenure", "MonthlyCharges", "TotalCharges", "Contract"],
    )

    if show_cols:
        before_sc, arrow_sc, after_sc = st.columns([5, 1, 5])
        with before_sc:
            st.markdown("**Before scaling**")
            st.dataframe(sample_rows[show_cols], use_container_width=True)
        with arrow_sc:
            st.markdown("")
            st.markdown("")
            st.markdown("### →")
        with after_sc:
            st.markdown("**After scaling** (mean=0, std=1)")
            st.dataframe(scaled_sample[show_cols].round(3), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Feature statistics before scaling")
    stats = enc_df_full[NUMERIC_COLS].describe().T[["mean", "std", "min", "max"]]
    stats.columns = ["Mean", "Std Dev", "Min", "Max"]
    st.dataframe(stats.round(2), use_container_width=True)

    st.markdown(
        "Notice how `TotalCharges` has a mean of ~2,280 and range of 0–8,685 while `SeniorCitizen` "
        "is just 0 or 1. After scaling, they all center around 0 with comparable spread."
    )

    st.markdown("---")
    st.markdown("#### Before vs After — Distribution Graphs")
    st.markdown(
        "The tables above show numbers, but graphs make the difference dramatic. "
        "Pick a feature to see how its distribution shifts after StandardScaler."
    )

    import plotly.graph_objects as go_fig
    from plotly.subplots import make_subplots as mk_sub

    scale_feature = st.selectbox(
        "Feature to visualize",
        ["tenure", "MonthlyCharges", "TotalCharges"],
        key="scale_viz_feature",
    )

    before_vals = enc_df_full[scale_feature].values
    col_idx = feature_cols.index(scale_feature)
    after_vals = all_scaled[:, col_idx]

    fig_scale = mk_sub(rows=1, cols=2, subplot_titles=["Before Scaling", "After Scaling (StandardScaler)"])

    fig_scale.add_trace(
        go_fig.Histogram(x=before_vals, nbinsx=40, marker_color="#636EFA", name="Before", showlegend=False),
        row=1, col=1,
    )
    fig_scale.add_trace(
        go_fig.Histogram(x=after_vals, nbinsx=40, marker_color="#00CC96", name="After", showlegend=False),
        row=1, col=2,
    )

    before_mean = float(np.mean(before_vals))
    after_mean = float(np.mean(after_vals))

    fig_scale.add_vline(x=before_mean, line_dash="dash", line_color="red", annotation_text=f"mean={before_mean:.1f}", row=1, col=1)
    fig_scale.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="mean=0", row=1, col=2)

    fig_scale.update_layout(
        title_text=f"Effect of StandardScaler on `{scale_feature}`",
        height=350,
    )
    fig_scale.update_xaxes(title_text=scale_feature, row=1, col=1)
    fig_scale.update_xaxes(title_text=f"{scale_feature} (scaled)", row=1, col=2)
    fig_scale.update_yaxes(title_text="Count", row=1, col=1)
    fig_scale.update_yaxes(title_text="Count", row=1, col=2)

    st.plotly_chart(fig_scale, use_container_width=True)

    before_std = float(np.std(before_vals))
    after_std = float(np.std(after_vals))
    sc1, sc2 = st.columns(2)
    sc1.metric("Before", f"mean = {before_mean:,.1f}, std = {before_std:,.1f}")
    sc2.metric("After", f"mean ≈ 0, std ≈ 1  ✓")
    st.caption(
        "After scaling, every feature has mean=0 and std=1 — that's exactly what StandardScaler "
        "is designed to do. This puts all features on equal footing so no single feature "
        "dominates just because its raw numbers are bigger."
    )
    st.markdown(
        "**What to notice:** The **shape** of the distribution stays exactly the same — "
        "StandardScaler doesn't change the data's shape, it just shifts and rescales it "
        "so the mean lands at 0 and the spread (std) becomes 1. "
        "This is why it's a safe transformation — no information is lost."
    )

    st.markdown("---")
    st.markdown("#### Which models need scaling?")
    st.markdown(
        """
        | Model | Needs scaling? | Why |
        |---|---|---|
        | **Logistic Regression** | Yes | Uses gradient descent — large features dominate the gradient |
        | **SGDClassifier** | Yes | Same reason — SGD is very sensitive to scale |
        | **SVM / KNN** | Yes | Distance-based — unscaled features distort distances |
        | **Random Forest** | No | Splits on thresholds — scale doesn't matter |
        | **XGBoost** | No | Same as Random Forest — tree-based |

        In our demo, we apply scaling **only** on the Live Updates page (SGDClassifier).
        The Churn Models page uses tree-based models that don't need it.
        """
    )

    st.markdown("---")
    st.markdown("#### What else could we have done?")
    st.markdown(
        """
        | Scaler | Formula | Best for |
        |---|---|---|
        | **StandardScaler** (ours) | (x - mean) / std | Features that are roughly normally distributed |
        | **MinMaxScaler** | (x - min) / (max - min), scales to [0, 1] | When you need bounded values (e.g., neural networks) |
        | **RobustScaler** | (x - median) / IQR | When you have outliers (uses median instead of mean) |
        | **MaxAbsScaler** | x / max(abs(x)) | Sparse data (doesn't shift the center) |
        """
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Train / Test Split
# ═══════════════════════════════════════════════════════════════════════════════
with tab_split:
    st.subheader("Step 4: Train / Test Split")

    st.markdown("#### Why this matters")
    st.markdown(
        "We need to know how well our model performs on **data it has never seen**. "
        "If we train and evaluate on the same data, the model could simply memorize "
        "the answers (overfitting) and we'd have no idea it would fail on new customers."
    )

    st.markdown("---")
    st.markdown("#### What we did")

    X_train, X_test, y_train, y_test, _, _ = get_train_test()

    split_col1, split_col2 = st.columns(2)
    with split_col1:
        st.metric("Training set", f"{len(X_train):,} rows (80%)")
        train_churn_rate = y_train.mean()
        st.metric("Training churn rate", f"{train_churn_rate:.1%}")
    with split_col2:
        st.metric("Test set", f"{len(X_test):,} rows (20%)")
        test_churn_rate = y_test.mean()
        st.metric("Test churn rate", f"{test_churn_rate:.1%}")

    st.success(
        f"Churn rate is **{train_churn_rate:.1%}** in training and **{test_churn_rate:.1%}** in test — "
        "nearly identical. This is because we used **stratified splitting**, which preserves the "
        "class distribution in both sets."
    )

    st.markdown("---")
    st.markdown("#### Stratified vs Random split")
    st.markdown(
        """
        | Aspect | Random Split | Stratified Split (ours) |
        |---|---|---|
        | **How** | Randomly assigns 80/20 | Ensures each set has the same churn ratio |
        | **Risk** | Training might get 30% churn, test gets 20% — model learns a skewed world | No imbalance risk |
        | **When to use** | Balanced datasets (50/50 classes) | Imbalanced datasets like ours (73.5% / 26.5%) |
        """
    )

    st.markdown("---")
    st.markdown("#### Why 80/20?")
    st.markdown(
        """
        This is the most common split ratio and a good default:
        - **80% training** gives the model enough data to learn patterns
        - **20% test** (~1,400 rows) is large enough for reliable metric estimates

        Other common ratios:
        - **70/30** — more test data, useful when you want higher confidence in metrics
        - **90/10** — when data is scarce and you need every row for training
        - **60/20/20** (train/validation/test) — when you need a separate validation
          set for hyperparameter tuning
        """
    )

    st.markdown("---")
    st.markdown("#### What about cross-validation?")
    st.markdown(
        """
        Instead of a single 80/20 split, **k-fold cross-validation** splits the data into k
        parts, trains on k-1, tests on the remaining 1, and rotates. This gives k different
        accuracy estimates and is more robust.

        We didn't use it here because:
        - The demo focuses on visual clarity (one train set, one test set is easier to explain)
        - With 7,043 rows, a single stratified split is already reliable
        - Cross-validation would multiply training time (important for SHAP computation)
        """
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Imputation Reference Guide
# ═══════════════════════════════════════════════════════════════════════════════
with tab_reference:
    st.subheader("Missing Data Imputation — Reference Guide")
    st.markdown(
        "Choosing the right imputation technique depends on the type of data, "
        "the amount of missingness, and whether the missingness itself carries information."
    )

    st.markdown("---")

    ref_data = pd.DataFrame([
        {
            "Situation": "Skewed continuous data",
            "Best Technique": "Median Imputation",
            "How it works": "Replace missing values with the median of the column",
            "Why": "Median is robust to outliers and skew — not pulled by extreme values",
            "Example": "TotalCharges in our dataset (right-skewed)",
        },
        {
            "Situation": "Normal (bell-curve) continuous data",
            "Best Technique": "Mean Imputation",
            "How it works": "Replace missing values with the average",
            "Why": "When data is symmetric, mean = median, and mean is the best estimate",
            "Example": "Height, temperature, standardized test scores",
        },
        {
            "Situation": "Categorical data",
            "Best Technique": "Mode Imputation",
            "How it works": "Replace missing values with the most frequent category",
            "Why": "You can't average text — the most common value is the safest guess",
            "Example": "Missing gender → fill with 'Male' if that's most common",
        },
        {
            "Situation": "Time-series / sequential data",
            "Best Technique": "Interpolation or ffill/bfill",
            "How it works": "Use neighboring time points to estimate the gap (linear, spline, etc.)",
            "Why": "Adjacent timestamps are highly correlated — interpolation preserves trends",
            "Example": "Missing stock price on a holiday → average of Friday and Monday",
        },
        {
            "Situation": "Features are correlated",
            "Best Technique": "KNN or Regression Imputation",
            "How it works": "Use other features to predict the missing value (KNN finds similar rows; regression fits a model)",
            "Why": "Leverages relationships between features for a smarter estimate",
            "Example": "Missing TotalCharges → predict from tenure × MonthlyCharges",
        },
        {
            "Situation": "Research / statistical rigor",
            "Best Technique": "MICE (Multiple Imputation by Chained Equations)",
            "How it works": "Iteratively imputes each feature using all others, multiple times, averaging results",
            "Why": "Captures uncertainty — produces multiple complete datasets and pools results",
            "Example": "Clinical trial data where imputation accuracy is critical",
        },
        {
            "Situation": "Missingness is informative",
            "Best Technique": "Indicator + Imputation",
            "How it works": "Add a binary column 'was_missing' (0/1) THEN impute the original",
            "Why": "Sometimes the fact that data is missing IS the signal (e.g., customer refused to answer = higher churn risk)",
            "Example": "Missing income field in a loan application → might indicate risk",
        },
        {
            "Situation": "Very little missing data (< 1%)",
            "Best Technique": "Drop rows",
            "How it works": "Simply remove rows with missing values",
            "Why": "Losing a handful of rows has negligible impact and is the simplest approach",
            "Example": "Our dataset: 11 out of 7,043 rows (0.16%) — dropping would be fine too",
        },
    ])

    st.dataframe(
        ref_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Situation": st.column_config.TextColumn(width="medium"),
            "Best Technique": st.column_config.TextColumn(width="medium"),
            "How it works": st.column_config.TextColumn(width="large"),
            "Why": st.column_config.TextColumn(width="large"),
            "Example": st.column_config.TextColumn(width="large"),
        },
    )

    st.markdown("---")
    st.markdown("#### Decision flowchart")
    st.markdown(
        """
        1. **How much data is missing?**
           - Less than 1%? → **Drop rows** (simplest)
           - More than 1%? → Continue below

        2. **What type of data?**
           - Continuous + skewed → **Median**
           - Continuous + normal → **Mean**
           - Categorical → **Mode**
           - Time-series → **Interpolation / ffill**

        3. **Are features correlated?**
           - Yes → Consider **KNN** or **Regression** imputation for better estimates

        4. **Is the missingness itself meaningful?**
           - Yes → Add a **missing indicator column** before imputing

        5. **Do you need statistical rigor?**
           - Yes → Use **MICE** for multiple imputation
        """
    )

    st.markdown("---")
    st.markdown("#### What we chose and why")
    st.info(
        "For our Telco dataset, only `TotalCharges` had 11 missing values (0.16%). "
        "We used **Median Imputation** because the column is right-skewed. "
        "Dropping rows would also have been perfectly valid given the tiny percentage."
    )
