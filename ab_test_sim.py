import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns
# ---------- Step 1: Generate Synthetic Data ----------

def generate_data(n_A=1000, n_B=1000, p_A=0.10, p_B=0.12):
    """Simulate binary conversion data for control (A) and test (B) groups."""
    group_A = np.random.binomial(1, p_A, n_A)
    group_B = np.random.binomial(1, p_B, n_B)
    
    df_A = pd.DataFrame({'group': 'A', 'converted': group_A})
    df_B = pd.DataFrame({'group': 'B', 'converted': group_B})
    
    df = pd.concat([df_A, df_B], ignore_index=True)
    return df

# ---------- Step 2: Run Statistical Test ----------

def run_ab_test(df):
    """Run two-sample z-test for proportions."""
    conv_counts = df.groupby('group')['converted'].sum().values
    total_counts = df.groupby('group')['converted'].count().values
    
    stat, pval = proportions_ztest(conv_counts, total_counts)
    return stat, pval

# ---------- Step 3: Calculate Conversion Rates ----------

def conversion_summary(df):
    """Calculate and summarize conversion rates for each group."""
    summary = df.groupby('group')['converted'].agg(['sum', 'count'])
    summary['conversion_rate'] = summary['sum'] / summary['count']
    return summary

def visualize_results(df, pval):
    """Visualize conversion rates for each group."""
    rates = df.groupby('group')['converted'].mean().reset_index()
    
    sns.barplot(data=rates, x='group', y='converted')
    plt.title(f"Conversion Rates (p-value: {pval:.3f})")
    plt.ylabel('Conversion Rate')
    plt.xlabel('Group')
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    # Ask user for inputs manually
    n_A = int(input("Enter sample size for Group A: "))
    n_B = int(input("Enter sample size for Group B: "))
    p_A = float(input("Enter conversion rate for Group A (e.g., 0.10 for 10%): "))
    p_B = float(input("Enter conversion rate for Group B (e.g., 0.12 for 12%): "))

    # Run simulation
    data = generate_data(n_A, n_B, p_A, p_B)
    stat, pval = run_ab_test(data)
    summary = conversion_summary(data)

    # Display results
    print("\n--- Conversion Summary ---")
    print(summary)
    print(f"\nZ-statistic: {stat:.3f}")
    print(f"P-value: {pval:.3f}")

    if pval < 0.05:
        print("\nResult: Statistically Significant ✅")
    else:
        print("\nResult: Not Statistically Significant ❌")

    visualize_results(data, pval)
