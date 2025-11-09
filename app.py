# -------------------------------
# Streamlit App Configuration
# -------------------------------
import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(page_title="A/B Testing Simulator", page_icon="🎯", layout="centered")

# Custom styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 10px;
    }
    .stApp {
        background-color: #e8efff;
    }
    h1, h2, h3 {
        color: #2b547e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Dashboard Header
# -------------------------------
st.title("🎯 A/B Testing Simulator")
st.markdown(
    "Use this interactive dashboard to **simulate and visualize A/B testing results**. "
    "Adjust sample sizes and conversion rates to explore how statistical significance behaves under different scenarios."
)
st.markdown("---")

# -------------------------------
# Section A: Input Controls
# -------------------------------
st.header("⚙️ Simulation Parameters")
st.markdown("Define your experiment setup below:")

col1, col2 = st.columns(2)

with col1:
    n_A = st.number_input("Sample size for Group A", min_value=0, max_value=10000, value=0, step=100)
    p_A = st.slider("Conversion Rate for Group A (%)", min_value=0, max_value=100, value=0) / 100

with col2:
    n_B = st.number_input("Sample size for Group B", min_value=0, max_value=10000, value=0, step=100)
    p_B = st.slider("Conversion Rate for Group B (%)", min_value=0, max_value=100, value=0) / 100

st.markdown("---")

# -------------------------------
# Section B: Simulation Output
# -------------------------------
st.header("📊 Simulation Results")

# -------------------------------
# Step 2: Run simulation and show results only if valid inputs
# -------------------------------
if n_A > 0 and n_B > 0 and p_A > 0 and p_B > 0:
    # Generate synthetic data
    group_A = np.random.binomial(1, p_A, n_A)
    group_B = np.random.binomial(1, p_B, n_B)
    df = pd.DataFrame({
        'group': ['A'] * n_A + ['B'] * n_B,
        'converted': np.concatenate([group_A, group_B])
    })

    # Run z-test
    conv_counts = df.groupby('group')['converted'].sum().values
    total_counts = df.groupby('group')['converted'].count().values
    stat, pval = proportions_ztest(conv_counts, total_counts)

    # Summarize conversion rates
    rates = df.groupby('group')['converted'].mean().reset_index()

    # Display results
    st.subheader("Results Summary")
    st.write(rates)
    st.write(f"**Z-statistic:** {stat:.3f}")
    st.write(f"**P-value:** {pval:.3f}")

    if pval < 0.05:
        st.success("✅ Statistically Significant Result")
    else:
        st.warning("❌ Not Statistically Significant")

    # Visualization
    fig, ax = plt.subplots()
    sns.barplot(data=rates, x='group', y='converted', ax=ax)
    ax.set_title(f"Conversion Rates (p-value: {pval:.3f})")
    ax.set_ylabel("Conversion Rate")
    st.pyplot(fig)
    
    # Additional Visualization
    st.markdown("---")
    st.subheader("📉 P-Value vs. Sample Size Analysis")
    st.markdown(
        "This chart shows how increasing the sample size affects the p-value "
        "(holding conversion rates constant). It demonstrates why larger tests provide more confidence."
    )

    sample_sizes = np.linspace(100, max(n_A, n_B), 20, dtype=int)
    p_values = []

    for size in sample_sizes:
        grpA = np.random.binomial(1, p_A, size)
        grpB = np.random.binomial(1, p_B, size)
        counts = [grpA.sum(), grpB.sum()]
        totals = [len(grpA), len(grpB)]
        _, pv = proportions_ztest(counts, totals)
        p_values.append(pv)

    fig2, ax2 = plt.subplots()
    sns.lineplot(x=sample_sizes, y=p_values, marker='o', ax=ax2)
    ax2.axhline(0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
    ax2.set_xlabel("Sample Size (per group)")
    ax2.set_ylabel("P-Value")
    ax2.set_title("P-Value Trend with Increasing Sample Size")
    ax2.legend()
    st.pyplot(fig2)


else:
    st.info("👆 Please enter non-zero values in all four fields to run the simulation.")

