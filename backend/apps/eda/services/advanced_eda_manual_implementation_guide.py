"""
AutoBI Advanced EDA manual implementation guide.

Purpose:
    Advanced EDA runs BEFORE cleaning, feature engineering, and ML readiness.
    It helps understand the raw dataset.

Important:
    This file intentionally does NOT implement pandas/matplotlib/seaborn logic.
    The project owner should manually implement these functions.

Shared helpers should go in:
    eda_common.py

Markers:
# MANUAL PANDAS/ML CODE REQUIRED
# MANUAL CODING GUIDE ONLY
"""
def build_advanced_eda_summary(dataset_id, dataset_version="raw", target_column=None):
    """
    Build Advanced EDA summary and graph recommendations.

    This function does not re-detect numeric/categorical/id columns. It reuses
    the existing Profile column groups through eda_common.load_profile_column_groups.
    Only the actual graph-rendering functions below remain manual.
    """
    from .eda_summary import build_advanced_eda_summary as build_summary

    return build_summary(dataset_id, dataset_version=dataset_version, target_column=target_column)


def generate_missing_value_analysis(df):
    """
    Purpose:
        Analyze missing values before cleaning.

    Important:
        This is NOT a Python graph rendering function.
        This should reuse normal pandas missing-value logic.
        Python graph generation is only needed in generate_missing_values_chart().
    """

    if df is None or df.empty:
        return {
            "total_missing_cells": 0,
            "total_missing_percentage": 0,
            "columns_with_missing": 0,
            "missing_by_column": [],
            "warnings": ["Dataset is empty."],
        }

    row_count = int(len(df))
    column_count = int(len(df.columns))
    total_cells = row_count * column_count

    missing_counts = df.isna().sum()
    missing_percentages = (missing_counts / row_count) * 100 if row_count > 0 else missing_counts * 0

    missing_by_column = []

    for column in df.columns:
        missing_count = int(missing_counts[column])

        if missing_count <= 0:
            continue

        missing_percentage = round(float(missing_percentages[column]), 2)

        missing_by_column.append({
            "column": str(column),
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
        })

    missing_by_column = sorted(
        missing_by_column,
        key=lambda item: item["missing_count"],
        reverse=True,
    )

    total_missing_cells = int(missing_counts.sum())

    if total_cells > 0:
        total_missing_percentage = round((total_missing_cells / total_cells) * 100, 2)
    else:
        total_missing_percentage = 0

    return {
        "total_missing_cells": total_missing_cells,
        "total_missing_percentage": total_missing_percentage,
        "columns_with_missing": int(len(missing_by_column)),
        "missing_by_column": missing_by_column,
        "warnings": [],
    }


def generate_distribution_kde_chart(df, column, bins="auto", sample_size=None):
    """
    Purpose:
        Generate raw numeric distribution + KDE as Python image.

    Used by:
        Advanced EDA / Validation EDA graph preview.

    Returns:
        Base64 PNG image payload for frontend <img> rendering.
    """

    import base64
    from io import BytesIO

    import numpy as np
    import pandas as pd

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import seaborn as sns

    if df is None or df.empty:
        return {
            "chart_type": "distribution_kde",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": ["Dataset is empty."],
        }

    if not column or column not in df.columns:
        return {
            "chart_type": "distribution_kde",
            "render_type": "python_image",
            "column": column,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": [f"Column '{column}' was not found."],
        }

    series = pd.to_numeric(df[column], errors="coerce")
    original_count = int(len(series))

    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    valid_count_before_sample = int(len(series))
    missing_or_invalid_count = int(original_count - valid_count_before_sample)

    if series.empty:
        return {
            "chart_type": "distribution_kde",
            "render_type": "python_image",
            "column": column,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {
                "original_count": original_count,
                "valid_count": 0,
                "missing_or_invalid_count": missing_or_invalid_count,
            },
            "warnings": [f"Column '{column}' has no valid numeric values."],
        }

    if sample_size is not None:
        try:
            sample_size = int(sample_size)
        except (TypeError, ValueError):
            sample_size = None

    if sample_size and sample_size > 0 and len(series) > sample_size:
        series = series.sample(n=sample_size, random_state=42)

    valid_count = int(len(series))
    unique_count = int(series.nunique(dropna=True))

    summary = {
        "original_count": original_count,
        "valid_count": valid_count_before_sample,
        "sampled_count": valid_count,
        "missing_or_invalid_count": missing_or_invalid_count,
        "mean": round(float(series.mean()), 4),
        "median": round(float(series.median()), 4),
        "std": round(float(series.std()), 4) if valid_count > 1 else 0,
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
        "skewness": round(float(series.skew()), 4) if valid_count > 2 else None,
        "kurtosis": round(float(series.kurtosis()), 4) if valid_count > 3 else None,
    }

    warnings = []

    if missing_or_invalid_count > 0:
        warnings.append(
            f"{missing_or_invalid_count} missing or invalid values were excluded for visualization."
        )

    if unique_count < 2:
        warnings.append("KDE was skipped because all valid values are identical.")

    try:
        sns.set_theme(style="whitegrid")

        fig, ax = plt.subplots(figsize=(10, 5.5))

        if unique_count >= 2 and valid_count >= 2:
            sns.histplot(
                series,
                bins=bins,
                kde=True,
                stat="density",
                color="#2563eb",
                edgecolor="#ffffff",
                linewidth=0.7,
                alpha=0.72,
                ax=ax,
            )
            y_label = "Density"
        else:
            sns.histplot(
                series,
                bins=1,
                kde=False,
                stat="count",
                color="#2563eb",
                edgecolor="#ffffff",
                linewidth=0.7,
                alpha=0.72,
                ax=ax,
            )
            y_label = "Count"

        mean_value = float(series.mean())
        median_value = float(series.median())

        ax.axvline(
            mean_value,
            linestyle="--",
            linewidth=1.5,
            color="#f97316",
            label=f"Mean: {mean_value:.2f}",
        )

        ax.axvline(
            median_value,
            linestyle=":",
            linewidth=1.8,
            color="#16a34a",
            label=f"Median: {median_value:.2f}",
        )

        ax.set_title(
            f"Distribution of {column}",
            fontsize=14,
            fontweight="bold",
            pad=14,
            color="#0f172a",
        )

        ax.set_xlabel(str(column), fontsize=11, color="#334155")
        ax.set_ylabel(y_label, fontsize=11, color="#334155")

        ax.tick_params(axis="both", labelsize=9, colors="#475569")

        ax.legend(frameon=True, fontsize=9)
        ax.grid(True, alpha=0.25)

        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        buffer.close()
        plt.close(fig)

        return {
            "chart_type": "distribution_kde",
            "render_type": "python_image",
            "column": column,
            "image_base64": image_base64,
            "image_mime_type": "image/png",
            "summary": summary,
            "settings": {
                "bins": bins,
                "sample_size": sample_size,
            },
            "warnings": warnings,
        }

    except Exception as exc:
        plt.close("all")

        return {
            "chart_type": "distribution_kde",
            "render_type": "python_image",
            "column": column,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": summary,
            "settings": {
                "bins": bins,
                "sample_size": sample_size,
            },
            "warnings": [f"Distribution KDE chart generation failed: {str(exc)}"],
        }

def generate_histogram_chart(df, column, bins="auto", sample_size=None):
    """
    Purpose:
        Generate raw histogram as Python PNG image.

    Important:
        - This function only generates graph image.
        - It does not detect numeric/categorical/id columns.
        - Call this only after Profile says the column is numeric.
        - Frontend should render image_base64 using <img>.
    """

    import base64
    from io import BytesIO

    import numpy as np
    import pandas as pd

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return {
            "chart_type": "histogram",
            "render_type": "not_implemented",
            "column": column,
            "image_base64": None,
            "image_mime_type": None,
            "summary": {},
            "warnings": [
                "Python graph libraries are not installed. Install matplotlib and seaborn."
            ],
        }

    if df is None or df.empty:
        return {
            "chart_type": "histogram",
            "render_type": "python_image",
            "column": column,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": ["Dataset is empty."],
        }

    if not column or column not in df.columns:
        return {
            "chart_type": "histogram",
            "render_type": "python_image",
            "column": column,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": [f"Column '{column}' was not found."],
        }

    series = pd.to_numeric(df[column], errors="coerce")
    original_count = int(len(series))

    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    valid_count_before_sample = int(len(series))
    missing_or_invalid_count = int(original_count - valid_count_before_sample)

    if series.empty:
        return {
            "chart_type": "histogram",
            "render_type": "python_image",
            "column": column,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {
                "original_count": original_count,
                "valid_count": 0,
                "missing_or_invalid_count": missing_or_invalid_count,
            },
            "warnings": [f"Column '{column}' has no valid numeric values."],
        }

    if sample_size is not None:
        try:
            sample_size = int(sample_size)
        except (TypeError, ValueError):
            sample_size = None

    if sample_size and sample_size > 0 and len(series) > sample_size:
        series = series.sample(n=sample_size, random_state=42)

    valid_count = int(len(series))

    summary = {
        "original_count": original_count,
        "valid_count": valid_count_before_sample,
        "sampled_count": valid_count,
        "missing_or_invalid_count": missing_or_invalid_count,
        "mean": round(float(series.mean()), 4),
        "median": round(float(series.median()), 4),
        "std": round(float(series.std()), 4) if valid_count > 1 else 0,
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
    }

    warnings = []

    if missing_or_invalid_count > 0:
        warnings.append(
            f"{missing_or_invalid_count} missing or invalid values were excluded for visualization."
        )

    try:
        sns.set_theme(style="whitegrid")

        fig, ax = plt.subplots(figsize=(10, 5.5))

        sns.histplot(
            series,
            bins=bins,
            kde=False,
            stat="count",
            color="#2563eb",
            edgecolor="#ffffff",
            linewidth=0.7,
            alpha=0.75,
            ax=ax,
        )

        mean_value = float(series.mean())
        median_value = float(series.median())

        ax.axvline(
            mean_value,
            linestyle="--",
            linewidth=1.5,
            color="#f97316",
            label=f"Mean: {mean_value:.2f}",
        )

        ax.axvline(
            median_value,
            linestyle=":",
            linewidth=1.8,
            color="#16a34a",
            label=f"Median: {median_value:.2f}",
        )

        ax.set_title(
            f"Histogram of {column}",
            fontsize=14,
            fontweight="bold",
            pad=14,
            color="#0f172a",
        )

        ax.set_xlabel(str(column), fontsize=11, color="#334155")
        ax.set_ylabel("Count", fontsize=11, color="#334155")
        ax.tick_params(axis="both", labelsize=9, colors="#475569")
        ax.legend(frameon=True, fontsize=9)
        ax.grid(True, alpha=0.25)

        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        buffer.close()
        plt.close(fig)

        return {
            "chart_type": "histogram",
            "render_type": "python_image",
            "column": column,
            "image_base64": image_base64,
            "image_mime_type": "image/png",
            "summary": summary,
            "settings": {
                "bins": bins,
                "sample_size": sample_size,
            },
            "warnings": warnings,
        }

    except Exception as exc:
        plt.close("all")

        return {
            "chart_type": "histogram",
            "render_type": "python_image",
            "column": column,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": summary,
            "settings": {
                "bins": bins,
                "sample_size": sample_size,
            },
            "warnings": [f"Histogram chart generation failed: {str(exc)}"],
        }


def generate_boxplot_outlier_chart(df, column, group_by=None, method="iqr"):
    """
    Purpose:
        Generate raw boxplot and outlier summary.

    Manual implementation:
        - Calculate Q1, Q3, IQR, whiskers, outlier count.
        - If group_by is supplied, calculate per group.
        - Return chart payload or image payload.

    # MANUAL PANDAS/ML CODE REQUIRED
    """
def generate_boxplot_outlier_chart(df, column, group_by=None, method="iqr"):
    """
    Purpose:
        Generate raw boxplot and outlier summary as Python PNG image.

    Important:
        - This function only generates graph image + outlier summary.
        - It does not detect numeric/categorical/id columns.
        - Call this only after Profile says `column` is numeric.
        - If group_by is supplied, it creates grouped boxplot.
        - Frontend should render image_base64 using <img>.
    """

    import base64
    from io import BytesIO

    import numpy as np
    import pandas as pd

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return {
            "chart_type": "boxplot_outlier",
            "render_type": "not_implemented",
            "column": column,
            "group_by": group_by,
            "image_base64": None,
            "image_mime_type": None,
            "summary": {},
            "warnings": [
                "Python graph libraries are not installed. Install matplotlib and seaborn."
            ],
        }

    if df is None or df.empty:
        return {
            "chart_type": "boxplot_outlier",
            "render_type": "python_image",
            "column": column,
            "group_by": group_by,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": ["Dataset is empty."],
        }

    if not column or column not in df.columns:
        return {
            "chart_type": "boxplot_outlier",
            "render_type": "python_image",
            "column": column,
            "group_by": group_by,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": [f"Column '{column}' was not found."],
        }

    if group_by and group_by not in df.columns:
        return {
            "chart_type": "boxplot_outlier",
            "render_type": "python_image",
            "column": column,
            "group_by": group_by,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": [f"Group column '{group_by}' was not found."],
        }

    working_df = df.copy()
    working_df[column] = pd.to_numeric(working_df[column], errors="coerce")
    working_df[column] = working_df[column].replace([np.inf, -np.inf], np.nan)

    original_count = int(len(working_df))
    working_df = working_df.dropna(subset=[column])
    valid_count = int(len(working_df))
    missing_or_invalid_count = int(original_count - valid_count)

    if working_df.empty:
        return {
            "chart_type": "boxplot_outlier",
            "render_type": "python_image",
            "column": column,
            "group_by": group_by,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {
                "original_count": original_count,
                "valid_count": 0,
                "missing_or_invalid_count": missing_or_invalid_count,
            },
            "warnings": [f"Column '{column}' has no valid numeric values."],
        }

    series = working_df[column]

    q1 = float(series.quantile(0.25))
    q2 = float(series.quantile(0.50))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (series < lower_bound) | (series > upper_bound)
    outlier_count = int(outlier_mask.sum())

    if valid_count > 0:
        outlier_percentage = round((outlier_count / valid_count) * 100, 2)
    else:
        outlier_percentage = 0

    summary = {
        "method": method,
        "original_count": original_count,
        "valid_count": valid_count,
        "missing_or_invalid_count": missing_or_invalid_count,
        "q1": round(q1, 6),
        "median": round(q2, 6),
        "q3": round(q3, 6),
        "iqr": round(float(iqr), 6),
        "lower_bound": round(float(lower_bound), 6),
        "upper_bound": round(float(upper_bound), 6),
        "min": round(float(series.min()), 6),
        "max": round(float(series.max()), 6),
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_percentage,
    }

    warnings = []

    if missing_or_invalid_count > 0:
        warnings.append(
            f"{missing_or_invalid_count} missing or invalid values were excluded for visualization."
        )

    if method != "iqr":
        warnings.append("Only IQR method is currently implemented for outlier summary.")

    try:
        sns.set_theme(style="whitegrid")

        fig_width = 10
        fig_height = 5.8

        if group_by:
            group_unique_count = int(working_df[group_by].nunique(dropna=True))

            if group_unique_count > 30:
                top_groups = (
                    working_df[group_by]
                    .astype(str)
                    .value_counts()
                    .head(30)
                    .index
                    .tolist()
                )

                working_df = working_df[
                    working_df[group_by].astype(str).isin(top_groups)
                ]

                warnings.append(
                    "Group count was high, so only top 30 groups were shown."
                )

            fig_width = max(10, min(18, group_unique_count * 0.65))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        if group_by:
            plot_df = working_df[[group_by, column]].copy()
            plot_df[group_by] = plot_df[group_by].astype(str)

            order = (
                plot_df.groupby(group_by)[column]
                .median()
                .sort_values(ascending=False)
                .index
                .tolist()
            )

            sns.boxplot(
                data=plot_df,
                x=group_by,
                y=column,
                order=order,
                color="#93c5fd",
                linewidth=1.2,
                fliersize=3,
                ax=ax,
            )

            ax.set_title(
                f"Boxplot of {column} by {group_by}",
                fontsize=14,
                fontweight="bold",
                pad=14,
                color="#0f172a",
            )

            ax.set_xlabel(str(group_by), fontsize=11, color="#334155")
            ax.set_ylabel(str(column), fontsize=11, color="#334155")
            ax.tick_params(axis="x", labelrotation=45, labelsize=8, colors="#475569")
            ax.tick_params(axis="y", labelsize=9, colors="#475569")

        else:
            sns.boxplot(
                y=series,
                color="#93c5fd",
                linewidth=1.2,
                fliersize=4,
                ax=ax,
            )

            ax.set_title(
                f"Boxplot of {column}",
                fontsize=14,
                fontweight="bold",
                pad=14,
                color="#0f172a",
            )

            ax.set_xlabel("", fontsize=11)
            ax.set_ylabel(str(column), fontsize=11, color="#334155")
            ax.tick_params(axis="both", labelsize=9, colors="#475569")

            ax.axhline(
                lower_bound,
                linestyle="--",
                linewidth=1.2,
                color="#f97316",
                label=f"Lower bound: {lower_bound:.2f}",
            )

            ax.axhline(
                upper_bound,
                linestyle="--",
                linewidth=1.2,
                color="#ef4444",
                label=f"Upper bound: {upper_bound:.2f}",
            )

            ax.legend(frameon=True, fontsize=9)

        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        buffer.close()
        plt.close(fig)

        return {
            "chart_type": "boxplot_outlier",
            "render_type": "python_image",
            "column": column,
            "group_by": group_by,
            "image_base64": image_base64,
            "image_mime_type": "image/png",
            "summary": summary,
            "warnings": warnings,
        }

    except Exception as exc:
        plt.close("all")

        return {
            "chart_type": "boxplot_outlier",
            "render_type": "python_image",
            "column": column,
            "group_by": group_by,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": summary,
            "warnings": [f"Boxplot chart generation failed: {str(exc)}"],
        }



def generate_violin_plot(df, column, group_by=None, sample_size=None):
    """
    Purpose:
        Generate raw violin plot distribution as Python PNG image.

    Important:
        - This function only generates graph image.
        - It does not detect numeric/categorical/id columns.
        - Call this only after Profile says `column` is numeric.
        - If group_by is supplied, it creates grouped violin plot.
        - Frontend should render image_base64 using <img>.
    """

    import base64
    from io import BytesIO

    import numpy as np
    import pandas as pd

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return {
            "chart_type": "violin_plot",
            "render_type": "not_implemented",
            "column": column,
            "group_by": group_by,
            "image_base64": None,
            "image_mime_type": None,
            "summary": {},
            "warnings": [
                "Python graph libraries are not installed. Install matplotlib and seaborn."
            ],
        }

    if df is None or df.empty:
        return {
            "chart_type": "violin_plot",
            "render_type": "python_image",
            "column": column,
            "group_by": group_by,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": ["Dataset is empty."],
        }

    if not column or column not in df.columns:
        return {
            "chart_type": "violin_plot",
            "render_type": "python_image",
            "column": column,
            "group_by": group_by,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": [f"Column '{column}' was not found."],
        }

    if group_by and group_by not in df.columns:
        return {
            "chart_type": "violin_plot",
            "render_type": "python_image",
            "column": column,
            "group_by": group_by,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": [f"Group column '{group_by}' was not found."],
        }

    working_df = df.copy()
    working_df[column] = pd.to_numeric(working_df[column], errors="coerce")
    working_df[column] = working_df[column].replace([np.inf, -np.inf], np.nan)

    original_count = int(len(working_df))
    working_df = working_df.dropna(subset=[column])
    valid_count_before_sample = int(len(working_df))
    missing_or_invalid_count = int(original_count - valid_count_before_sample)

    if working_df.empty:
        return {
            "chart_type": "violin_plot",
            "render_type": "python_image",
            "column": column,
            "group_by": group_by,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {
                "original_count": original_count,
                "valid_count": 0,
                "missing_or_invalid_count": missing_or_invalid_count,
            },
            "warnings": [f"Column '{column}' has no valid numeric values."],
        }

    if sample_size is not None:
        try:
            sample_size = int(sample_size)
        except (TypeError, ValueError):
            sample_size = None

    if sample_size and sample_size > 0 and len(working_df) > sample_size:
        working_df = working_df.sample(n=sample_size, random_state=42)

    valid_count = int(len(working_df))
    series = working_df[column]

    unique_count = int(series.nunique(dropna=True))

    summary = {
        "original_count": original_count,
        "valid_count": valid_count_before_sample,
        "sampled_count": valid_count,
        "missing_or_invalid_count": missing_or_invalid_count,
        "mean": round(float(series.mean()), 4),
        "median": round(float(series.median()), 4),
        "std": round(float(series.std()), 4) if valid_count > 1 else 0,
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
        "skewness": round(float(series.skew()), 4) if valid_count > 2 else None,
        "kurtosis": round(float(series.kurtosis()), 4) if valid_count > 3 else None,
    }

    warnings = []

    if missing_or_invalid_count > 0:
        warnings.append(
            f"{missing_or_invalid_count} missing or invalid values were excluded for visualization."
        )

    if unique_count < 2:
        warnings.append("Violin plot may not be meaningful because all valid values are identical.")

    try:
        sns.set_theme(style="whitegrid")

        fig_width = 10
        fig_height = 5.8

        if group_by:
            working_df = working_df.dropna(subset=[group_by])
            working_df[group_by] = working_df[group_by].astype(str)

            group_unique_count = int(working_df[group_by].nunique(dropna=True))

            if group_unique_count > 30:
                top_groups = (
                    working_df[group_by]
                    .value_counts()
                    .head(30)
                    .index
                    .tolist()
                )

                working_df = working_df[working_df[group_by].isin(top_groups)]

                warnings.append(
                    "Group count was high, so only top 30 groups were shown."
                )

            fig_width = max(10, min(18, group_unique_count * 0.65))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        if group_by:
            order = (
                working_df.groupby(group_by)[column]
                .median()
                .sort_values(ascending=False)
                .index
                .tolist()
            )

            sns.violinplot(
                data=working_df,
                x=group_by,
                y=column,
                order=order,
                inner="quartile",
                cut=0,
                linewidth=1,
                color="#93c5fd",
                ax=ax,
            )

            ax.set_title(
                f"Violin Plot of {column} by {group_by}",
                fontsize=14,
                fontweight="bold",
                pad=14,
                color="#0f172a",
            )

            ax.set_xlabel(str(group_by), fontsize=11, color="#334155")
            ax.set_ylabel(str(column), fontsize=11, color="#334155")
            ax.tick_params(axis="x", labelrotation=45, labelsize=8, colors="#475569")
            ax.tick_params(axis="y", labelsize=9, colors="#475569")

        else:
            sns.violinplot(
                y=series,
                inner="quartile",
                cut=0,
                linewidth=1,
                color="#93c5fd",
                ax=ax,
            )

            ax.set_title(
                f"Violin Plot of {column}",
                fontsize=14,
                fontweight="bold",
                pad=14,
                color="#0f172a",
            )

            ax.set_xlabel("", fontsize=11)
            ax.set_ylabel(str(column), fontsize=11, color="#334155")
            ax.tick_params(axis="both", labelsize=9, colors="#475569")

        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        buffer.close()
        plt.close(fig)

        return {
            "chart_type": "violin_plot",
            "render_type": "python_image",
            "column": column,
            "group_by": group_by,
            "image_base64": image_base64,
            "image_mime_type": "image/png",
            "summary": summary,
            "settings": {
                "sample_size": sample_size,
            },
            "warnings": warnings,
        }

    except Exception as exc:
        plt.close("all")

        return {
            "chart_type": "violin_plot",
            "render_type": "python_image",
            "column": column,
            "group_by": group_by,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": summary,
            "settings": {
                "sample_size": sample_size,
            },
            "warnings": [f"Violin plot generation failed: {str(exc)}"],
        }


def generate_qq_plot(df, column, distribution="normal"):
    """
    Purpose:
        Generate QQ plot for raw numeric column.

    Manual implementation:
        - Sort numeric values.
        - Compute theoretical quantiles.
        - Return QQ points and reference line or image payload.

    # MANUAL PANDAS/ML CODE REQUIRED
    """

    import base64
    from io import BytesIO

    import numpy as np
    import pandas as pd

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        return {
            "chart_type": "qq_plot",
            "render_type": "not_implemented",
            "column": column,
            "distribution": distribution,
            "image_base64": None,
            "image_mime_type": None,
            "summary": {},
            "warnings": [
                "Python graph libraries are not installed. Install matplotlib and scipy."
            ],
        }

    if df is None or df.empty:
        return {
            "chart_type": "qq_plot",
            "render_type": "python_image",
            "column": column,
            "distribution": distribution,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": ["Dataset is empty."],
        }

    if not column or column not in df.columns:
        return {
            "chart_type": "qq_plot",
            "render_type": "python_image",
            "column": column,
            "distribution": distribution,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": [f"Column '{column}' was not found."],
        }

    series = pd.to_numeric(df[column], errors="coerce")
    original_count = int(len(series))

    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    valid_count = int(len(series))
    missing_or_invalid_count = int(original_count - valid_count)

    if valid_count < 3:
        return {
            "chart_type": "qq_plot",
            "render_type": "python_image",
            "column": column,
            "distribution": distribution,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {
                "original_count": original_count,
                "valid_count": valid_count,
                "missing_or_invalid_count": missing_or_invalid_count,
            },
            "warnings": ["QQ plot requires at least 3 valid numeric values."],
        }

    if series.nunique(dropna=True) < 2:
        return {
            "chart_type": "qq_plot",
            "render_type": "python_image",
            "column": column,
            "distribution": distribution,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {
                "original_count": original_count,
                "valid_count": valid_count,
                "missing_or_invalid_count": missing_or_invalid_count,
                "unique_count": 1,
            },
            "warnings": ["QQ plot is not meaningful because all valid values are identical."],
        }

    warnings = []

    if missing_or_invalid_count > 0:
        warnings.append(
            f"{missing_or_invalid_count} missing or invalid values were excluded for visualization."
        )

    if distribution != "normal":
        warnings.append("Only normal QQ plot is currently implemented.")

    try:
        values = series.to_numpy(dtype=float)

        # QQ calculation against normal distribution
        theoretical_quantiles, ordered_values = stats.probplot(
            values,
            dist="norm",
            fit=False,
        )

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            theoretical_quantiles,
            ordered_values,
        )

        reference_line = slope * np.array(theoretical_quantiles) + intercept

        summary = {
            "distribution": "normal",
            "original_count": original_count,
            "valid_count": valid_count,
            "missing_or_invalid_count": missing_or_invalid_count,
            "mean": round(float(series.mean()), 6),
            "median": round(float(series.median()), 6),
            "std": round(float(series.std()), 6),
            "min": round(float(series.min()), 6),
            "max": round(float(series.max()), 6),
            "skewness": round(float(series.skew()), 6),
            "kurtosis": round(float(series.kurtosis()), 6),
            "qq_r_value": round(float(r_value), 6),
            "qq_r_squared": round(float(r_value ** 2), 6),
        }

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(
            theoretical_quantiles,
            ordered_values,
            s=18,
            alpha=0.75,
            color="#2563eb",
            edgecolors="none",
        )

        ax.plot(
            theoretical_quantiles,
            reference_line,
            color="#f97316",
            linewidth=1.8,
            linestyle="--",
            label="Reference line",
        )

        ax.set_title(
            f"QQ Plot of {column}",
            fontsize=14,
            fontweight="bold",
            pad=14,
            color="#0f172a",
        )

        ax.set_xlabel("Theoretical Quantiles", fontsize=11, color="#334155")
        ax.set_ylabel("Sample Quantiles", fontsize=11, color="#334155")

        ax.tick_params(axis="both", labelsize=9, colors="#475569")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True, fontsize=9)

        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        buffer.close()
        plt.close(fig)

        return {
            "chart_type": "qq_plot",
            "render_type": "python_image",
            "column": column,
            "distribution": "normal",
            "image_base64": image_base64,
            "image_mime_type": "image/png",
            "summary": summary,
            "warnings": warnings,
        }

    except Exception as exc:
        plt.close("all")

        return {
            "chart_type": "qq_plot",
            "render_type": "python_image",
            "column": column,
            "distribution": distribution,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": [f"QQ plot generation failed: {str(exc)}"],
        }


def generate_correlation_heatmap(df, numeric_columns=None, method="pearson"):
    """
    Purpose:
        Generate raw numeric correlation heatmap as Python PNG image.
    """

    import base64
    from io import BytesIO

    import numpy as np
    import pandas as pd

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return {
            "chart_type": "correlation_heatmap",
            "render_type": "not_implemented",
            "image_base64": None,
            "image_mime_type": None,
            "summary": {},
            "warnings": ["Install matplotlib and seaborn to generate correlation heatmap."],
        }

    if df is None or df.empty:
        return {
            "chart_type": "correlation_heatmap",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": ["Dataset is empty."],
        }

    if numeric_columns:
        valid_columns = [col for col in numeric_columns if col in df.columns]
    else:
        valid_columns = df.select_dtypes(include=["number"]).columns.tolist()

    if len(valid_columns) < 2:
        return {
            "chart_type": "correlation_heatmap",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {"numeric_column_count": len(valid_columns)},
            "warnings": ["Correlation heatmap requires at least 2 numeric columns."],
        }

    working_df = df[valid_columns].apply(pd.to_numeric, errors="coerce")
    working_df = working_df.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are completely empty or constant.
    usable_columns = []
    for col in working_df.columns:
        if working_df[col].dropna().nunique() >= 2:
            usable_columns.append(col)

    if len(usable_columns) < 2:
        return {
            "chart_type": "correlation_heatmap",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {"usable_numeric_column_count": len(usable_columns)},
            "warnings": ["Not enough non-constant numeric columns for correlation."],
        }

    # Limit columns for readability/performance.
    if len(usable_columns) > 20:
        usable_columns = usable_columns[:20]
        column_limit_warning = "Only first 20 numeric columns were shown for readability."
    else:
        column_limit_warning = None

    corr = working_df[usable_columns].corr(method=method)

    try:
        size = max(8, min(18, len(usable_columns) * 0.65))
        fig, ax = plt.subplots(figsize=(size, size * 0.8))

        sns.heatmap(
            corr,
            annot=True if len(usable_columns) <= 12 else False,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.75},
            ax=ax,
        )

        ax.set_title(
            f"{method.title()} Correlation Heatmap",
            fontsize=14,
            fontweight="bold",
            pad=14,
            color="#0f172a",
        )

        ax.tick_params(axis="x", labelrotation=45, labelsize=8, colors="#475569")
        ax.tick_params(axis="y", labelrotation=0, labelsize=8, colors="#475569")

        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        buffer.close()
        plt.close(fig)

        warnings = []
        if column_limit_warning:
            warnings.append(column_limit_warning)

        return {
            "chart_type": "correlation_heatmap",
            "render_type": "python_image",
            "image_base64": image_base64,
            "image_mime_type": "image/png",
            "summary": {
                "method": method,
                "numeric_column_count": len(usable_columns),
                "columns": [str(col) for col in usable_columns],
            },
            "warnings": warnings,
        }

    except Exception as exc:
        plt.close("all")
        return {
            "chart_type": "correlation_heatmap",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {"method": method},
            "warnings": [f"Correlation heatmap generation failed: {str(exc)}"],
        }


def generate_countplot_chart(df, column, top_n=30):
    """
    Purpose:
        Generate category count plot as a Python PNG image.

    Important:
        - This function only renders a graph from an already-selected column.
        - It does not detect categorical/id/text columns.
        - Frontend should render image_base64 using <img>.
    """

    import base64
    from io import BytesIO

    import pandas as pd

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return {
            "chart_type": "countplot",
            "render_type": "not_implemented",
            "column": column,
            "image_base64": None,
            "image_mime_type": None,
            "summary": {},
            "warnings": ["Install matplotlib and seaborn to generate countplot chart."],
        }

    if df is None or df.empty:
        return {
            "chart_type": "countplot",
            "render_type": "python_image",
            "column": column,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": ["Dataset is empty."],
        }

    if not column or column not in df.columns:
        return {
            "chart_type": "countplot",
            "render_type": "python_image",
            "column": column,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": [f"Column '{column}' was not found."],
        }

    series = df[column].dropna().astype(str)
    total_count = int(len(series))

    if series.empty:
        return {
            "chart_type": "countplot",
            "render_type": "python_image",
            "column": column,
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {
                "total_count": 0,
                "category_count": 0,
                "shown_category_count": 0,
                "rows": [],
            },
            "warnings": [f"Column '{column}' has no non-missing values."],
        }

    try:
        top_n = int(top_n)
    except (TypeError, ValueError):
        top_n = 30

    top_n = max(1, top_n)
    counts = series.value_counts()
    category_count = int(len(counts))
    shown_counts = counts.head(top_n)
    rows = [
        {
            "category": str(category),
            "count": int(count),
            "percentage": round((int(count) / total_count) * 100, 2) if total_count else 0,
        }
        for category, count in shown_counts.items()
    ]

    warnings = []
    if category_count > len(shown_counts):
        warnings.append(f"Column has {category_count} categories, so only top {len(shown_counts)} are shown.")

    plot_df = pd.DataFrame(rows)

    try:
        sns.set_theme(style="whitegrid")

        fig_width = max(9, min(18, len(plot_df) * 0.7))
        fig, ax = plt.subplots(figsize=(fig_width, 5.5))

        sns.barplot(
            data=plot_df,
            x="category",
            y="count",
            color="#2563eb",
            ax=ax,
        )

        ax.set_title(
            f"Category Count: {column}",
            fontsize=14,
            fontweight="bold",
            pad=14,
            color="#0f172a",
        )
        ax.set_xlabel(str(column), fontsize=11, color="#334155")
        ax.set_ylabel("Count", fontsize=11, color="#334155")
        ax.tick_params(axis="x", labelrotation=45, labelsize=8, colors="#475569")
        ax.tick_params(axis="y", labelsize=9, colors="#475569")
        ax.grid(True, axis="y", alpha=0.25)

        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        buffer.close()
        plt.close(fig)

        return {
            "chart_type": "countplot",
            "render_type": "python_image",
            "column": column,
            "image_base64": image_base64,
            "image_mime_type": "image/png",
            "summary": {
                "total_count": total_count,
                "category_count": category_count,
                "shown_category_count": int(len(rows)),
                "rows": rows,
            },
            "warnings": warnings,
        }

    except Exception as exc:
        plt.close("all")
        return {
            "chart_type": "countplot",
            "render_type": "error",
            "column": column,
            "image_base64": None,
            "image_mime_type": None,
            "summary": {
                "total_count": total_count,
                "category_count": category_count,
                "shown_category_count": int(len(rows)),
                "rows": rows,
            },
            "warnings": [f"Countplot chart generation failed: {str(exc)}"],
        }


def generate_target_balance_chart(df, target_column):
    """
    Purpose:
        Analyze raw target/class balance as Python PNG image.
    """

    import base64
    from io import BytesIO

    import pandas as pd

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return {
            "chart_type": "target_balance",
            "render_type": "not_implemented",
            "image_base64": None,
            "image_mime_type": None,
            "summary": {},
            "warnings": ["Install matplotlib and seaborn to generate target balance chart."],
        }

    if df is None or df.empty:
        return {
            "chart_type": "target_balance",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": ["Dataset is empty."],
        }

    if not target_column or target_column not in df.columns:
        return {
            "chart_type": "target_balance",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": [f"Target column '{target_column}' was not found."],
        }

    series = df[target_column].dropna().astype(str)

    if series.empty:
        return {
            "chart_type": "target_balance",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {"valid_count": 0},
            "warnings": ["Target column has no valid values."],
        }

    counts = series.value_counts()
    total = int(counts.sum())

    if len(counts) > 30:
        counts = counts.head(30)
        warning = "Target has many classes, so only top 30 classes were shown."
    else:
        warning = None

    plot_df = counts.reset_index()
    plot_df.columns = ["target_value", "count"]
    plot_df["percentage"] = (plot_df["count"] / total) * 100

    try:
        sns.set_theme(style="whitegrid")

        fig_width = max(9, min(18, len(plot_df) * 0.7))
        fig, ax = plt.subplots(figsize=(fig_width, 5.5))

        sns.barplot(
            data=plot_df,
            x="target_value",
            y="count",
            color="#2563eb",
            ax=ax,
        )

        ax.set_title(
            f"Target Balance: {target_column}",
            fontsize=14,
            fontweight="bold",
            pad=14,
            color="#0f172a",
        )
        ax.set_xlabel(str(target_column), fontsize=11, color="#334155")
        ax.set_ylabel("Count", fontsize=11, color="#334155")
        ax.tick_params(axis="x", labelrotation=45, labelsize=8, colors="#475569")
        ax.tick_params(axis="y", labelsize=9, colors="#475569")
        ax.grid(True, axis="y", alpha=0.25)

        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        buffer.close()
        plt.close(fig)

        balance_table = [
            {
                "target_value": str(row["target_value"]),
                "count": int(row["count"]),
                "percentage": round(float(row["percentage"]), 2),
            }
            for _, row in plot_df.iterrows()
        ]

        warnings = []
        if warning:
            warnings.append(warning)

        return {
            "chart_type": "target_balance",
            "render_type": "python_image",
            "target_column": target_column,
            "image_base64": image_base64,
            "image_mime_type": "image/png",
            "summary": {
                "total_count": total,
                "class_count": int(series.nunique()),
                "classes": balance_table,
            },
            "warnings": warnings,
        }

    except Exception as exc:
        plt.close("all")
        return {
            "chart_type": "target_balance",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": [f"Target balance chart generation failed: {str(exc)}"],
        }


def generate_target_rate_by_category(df, category_column, target_column):
    """
    Purpose:
        Analyze raw target rate by category as Python PNG image.

    Works best for binary target columns.
    """

    import base64
    from io import BytesIO

    import numpy as np
    import pandas as pd

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return {
            "chart_type": "target_rate_by_category",
            "render_type": "not_implemented",
            "image_base64": None,
            "image_mime_type": None,
            "summary": {},
            "warnings": ["Install matplotlib and seaborn to generate target rate chart."],
        }

    if df is None or df.empty:
        return {
            "chart_type": "target_rate_by_category",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": ["Dataset is empty."],
        }

    if category_column not in df.columns or target_column not in df.columns:
        return {
            "chart_type": "target_rate_by_category",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": ["Category column or target column was not found."],
        }

    working_df = df[[category_column, target_column]].copy()
    working_df = working_df.dropna(subset=[category_column, target_column])
    working_df[category_column] = working_df[category_column].astype(str)

    target_series = working_df[target_column]

    # Convert common binary labels to 0/1.
    if target_series.dtype == "object" or str(target_series.dtype).startswith("category"):
        normalized = target_series.astype(str).str.strip().str.lower()
        mapping = {
            "yes": 1,
            "y": 1,
            "true": 1,
            "1": 1,
            "positive": 1,
            "no": 0,
            "n": 0,
            "false": 0,
            "0": 0,
            "negative": 0,
        }
        working_df["_target_numeric"] = normalized.map(mapping)
    else:
        working_df["_target_numeric"] = pd.to_numeric(target_series, errors="coerce")

    working_df["_target_numeric"] = working_df["_target_numeric"].replace([np.inf, -np.inf], np.nan)
    working_df = working_df.dropna(subset=["_target_numeric"])

    unique_target_values = sorted(working_df["_target_numeric"].dropna().unique().tolist())

    if len(unique_target_values) > 2:
        return {
            "chart_type": "target_rate_by_category",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {"target_unique_values": unique_target_values[:10]},
            "warnings": ["Target rate by category requires a binary target column."],
        }

    if working_df.empty:
        return {
            "chart_type": "target_rate_by_category",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": ["No valid category/target rows available."],
        }

    grouped = (
        working_df.groupby(category_column)["_target_numeric"]
        .agg(["mean", "count"])
        .reset_index()
    )

    grouped["target_rate"] = grouped["mean"] * 100
    grouped = grouped.sort_values("target_rate", ascending=False)

    if len(grouped) > 30:
        grouped = grouped.head(30)
        warning = "Category count was high, so only top 30 categories were shown."
    else:
        warning = None

    try:
        sns.set_theme(style="whitegrid")

        fig_width = max(9, min(18, len(grouped) * 0.7))
        fig, ax = plt.subplots(figsize=(fig_width, 5.5))

        sns.barplot(
            data=grouped,
            x=category_column,
            y="target_rate",
            color="#2563eb",
            ax=ax,
        )

        ax.set_title(
            f"Target Rate by {category_column}",
            fontsize=14,
            fontweight="bold",
            pad=14,
            color="#0f172a",
        )
        ax.set_xlabel(str(category_column), fontsize=11, color="#334155")
        ax.set_ylabel("Target Rate (%)", fontsize=11, color="#334155")
        ax.tick_params(axis="x", labelrotation=45, labelsize=8, colors="#475569")
        ax.tick_params(axis="y", labelsize=9, colors="#475569")
        ax.grid(True, axis="y", alpha=0.25)

        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        buffer.close()
        plt.close(fig)

        rows = [
            {
                "category": str(row[category_column]),
                "target_rate": round(float(row["target_rate"]), 2),
                "count": int(row["count"]),
            }
            for _, row in grouped.iterrows()
        ]

        warnings = []
        if warning:
            warnings.append(warning)

        return {
            "chart_type": "target_rate_by_category",
            "render_type": "python_image",
            "category_column": category_column,
            "target_column": target_column,
            "image_base64": image_base64,
            "image_mime_type": "image/png",
            "summary": {
                "rows": rows,
                "category_count": int(len(grouped)),
            },
            "warnings": warnings,
        }

    except Exception as exc:
        plt.close("all")
        return {
            "chart_type": "target_rate_by_category",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": [f"Target rate chart generation failed: {str(exc)}"],
        }


def generate_pairplot_matrix(df, numeric_columns=None, sample_size=None):
    """
    Purpose:
        Generate pairplot/scatter-matrix for raw numeric features as Python PNG image.
    """

    import base64
    from io import BytesIO

    import numpy as np
    import pandas as pd

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return {
            "chart_type": "pairplot_matrix",
            "render_type": "not_implemented",
            "image_base64": None,
            "image_mime_type": None,
            "summary": {},
            "warnings": ["Install matplotlib and seaborn to generate pairplot matrix."],
        }

    if df is None or df.empty:
        return {
            "chart_type": "pairplot_matrix",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": ["Dataset is empty."],
        }

    if numeric_columns:
        valid_columns = [col for col in numeric_columns if col in df.columns]
    else:
        valid_columns = df.select_dtypes(include=["number"]).columns.tolist()

    if len(valid_columns) < 2:
        return {
            "chart_type": "pairplot_matrix",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {"numeric_column_count": len(valid_columns)},
            "warnings": ["Pairplot requires at least 2 numeric columns."],
        }

    if len(valid_columns) > 6:
        valid_columns = valid_columns[:6]
        column_limit_warning = "Only first 6 numeric columns were shown for pairplot performance."
    else:
        column_limit_warning = None

    working_df = df[valid_columns].apply(pd.to_numeric, errors="coerce")
    working_df = working_df.replace([np.inf, -np.inf], np.nan).dropna()

    if sample_size is not None:
        try:
            sample_size = int(sample_size)
        except (TypeError, ValueError):
            sample_size = None

    if sample_size and sample_size > 0 and len(working_df) > sample_size:
        working_df = working_df.sample(n=sample_size, random_state=42)

    if len(working_df) < 3:
        return {
            "chart_type": "pairplot_matrix",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {"valid_rows": int(len(working_df))},
            "warnings": ["Pairplot requires at least 3 valid rows."],
        }

    try:
        sns.set_theme(style="whitegrid")

        grid = sns.pairplot(
            working_df,
            diag_kind="hist",
            plot_kws={"s": 18, "alpha": 0.65, "color": "#2563eb"},
            diag_kws={"color": "#2563eb"},
        )

        grid.fig.suptitle(
            "Pairplot Matrix",
            fontsize=14,
            fontweight="bold",
            y=1.02,
            color="#0f172a",
        )

        buffer = BytesIO()
        grid.fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        buffer.close()
        plt.close(grid.fig)

        warnings = []
        if column_limit_warning:
            warnings.append(column_limit_warning)

        return {
            "chart_type": "pairplot_matrix",
            "render_type": "python_image",
            "image_base64": image_base64,
            "image_mime_type": "image/png",
            "summary": {
                "columns": [str(col) for col in valid_columns],
                "row_count_used": int(len(working_df)),
                "column_count_used": int(len(valid_columns)),
            },
            "settings": {"sample_size": sample_size},
            "warnings": warnings,
        }

    except Exception as exc:
        plt.close("all")
        return {
            "chart_type": "pairplot_matrix",
            "render_type": "python_image",
            "image_base64": None,
            "image_mime_type": "image/png",
            "summary": {},
            "warnings": [f"Pairplot generation failed: {str(exc)}"],
        }


def generate_skewness_kurtosis_summary(df, numeric_columns=None):
    """
    Purpose:
        Calculate skewness and kurtosis for raw numeric columns.

    Returns:
        JSON-safe table payload.
    """

    import numpy as np
    import pandas as pd

    if df is None or df.empty:
        return {
            "chart_type": "skewness_kurtosis_summary",
            "render_type": "table",
            "rows": [],
            "summary": {},
            "warnings": ["Dataset is empty."],
        }

    if numeric_columns:
        valid_columns = [col for col in numeric_columns if col in df.columns]
    else:
        valid_columns = df.select_dtypes(include=["number"]).columns.tolist()

    if not valid_columns:
        return {
            "chart_type": "skewness_kurtosis_summary",
            "render_type": "table",
            "rows": [],
            "summary": {},
            "warnings": ["No numeric columns available."],
        }

    rows = []

    for column in valid_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        original_count = int(len(series))

        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        valid_count = int(len(series))
        missing_or_invalid_count = int(original_count - valid_count)

        if valid_count < 3:
            rows.append({
                "column": str(column),
                "valid_count": valid_count,
                "missing_or_invalid_count": missing_or_invalid_count,
                "skewness": None,
                "kurtosis": None,
                "shape": "Not enough data",
                "recommendation": "Need at least 3 valid numeric values.",
            })
            continue

        skewness = float(series.skew())
        kurtosis = float(series.kurtosis())

        if abs(skewness) < 0.5:
            shape = "Approximately symmetric"
        elif skewness > 0:
            shape = "Right-skewed"
        else:
            shape = "Left-skewed"

        recommendation = "No strong transformation needed."

        if abs(skewness) >= 1:
            recommendation = "Highly skewed. Consider log/sqrt/Box-Cox transformation if suitable."

        if abs(kurtosis) >= 3:
            recommendation += " Heavy tails detected. Check outliers."

        rows.append({
            "column": str(column),
            "valid_count": valid_count,
            "missing_or_invalid_count": missing_or_invalid_count,
            "mean": round(float(series.mean()), 6),
            "median": round(float(series.median()), 6),
            "std": round(float(series.std()), 6) if valid_count > 1 else 0,
            "skewness": round(skewness, 6),
            "kurtosis": round(kurtosis, 6),
            "shape": shape,
            "recommendation": recommendation,
        })

    return {
        "chart_type": "skewness_kurtosis_summary",
        "render_type": "table",
        "rows": rows,
        "summary": {
            "numeric_column_count": int(len(valid_columns)),
            "highly_skewed_count": int(
                sum(
                    1
                    for row in rows
                    if row.get("skewness") is not None and abs(row["skewness"]) >= 1
                )
            ),
        },
        "warnings": [],
    }
