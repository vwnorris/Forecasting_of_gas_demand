import pandas as pd

### Function to make latex tables from Granger top features data ###

station = "Tyskland"

def format_p_value(p):
    """Format p-value to LaTeX-friendly scientific notation"""
    try:
        p = float(p)
        if p == 0.0:
            return r"$\approx 0$"
        elif p < 1e-4:
            exponent = int(f"{p:.1e}".split('e')[1])
            base = float(f"{p:.1e}".split('e')[0])
            return rf"${base:.2f} \cdot 10^{{{exponent}}}$"
        else:
            return f"${p:.5f}$"
    except:
        return str(p)

def df_to_latex_table(df, caption, label, max_rows=179):
    df = df.head(max_rows)
    
    latex = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        r"    \begin{tabular}{r l l l l}",
        r"        \hline",
        r"        \textbf{Rank} & \textbf{Feature} & \textbf{p-value} & \textbf{F-statistic} & \textbf{Best Lag} \\",
        r"        \hline"
    ]

    for _, row in df.iterrows():
        rank = int(row["Rank"])
        feature = row["Feature"].replace("_", r"\_").replace("%", r"\%")
        pval = format_p_value(row["p-value"])
        fstat = f"{row['F-statistic']:.2f}"
        lag = int(row["Best Lag"])
        latex.append(f"        {rank} & {feature} & {pval} & {fstat} & {lag} \\\\")

    latex.extend([
        r"        \hline",
        r"    \end{tabular}",
        rf"    \caption{{Top {max_rows} features ranked by F-statistic from Granger causality test for target \textit{{Volumrate\_{station}\_hourly}}.}}",
        rf"    \label{{tab:granger_{station}_fstat}}",
        r"\end{table}"
    ])

    return "\n".join(latex)

df = pd.read_csv(f"results/granger_top_features_{station}_fstat.csv")
df = df.sort_values(by="F-statistic", ascending=False).reset_index(drop=True)
df["Rank"] = df.index + 1

latex_code = df_to_latex_table(df, "Top features by F-statistic for France", "tab:granger_france_fstat")

with open(f"results/latex/granger_top_feature_{station}_fstat_table.tex", "w") as f:
    f.write(latex_code)

print(f"🫑 LaTeX table written to results/latex/granger_top_feature_{station}_fstat_table.tex")
