import os
import io
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pypdf
from groq import Groq
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image as RLImage,
    Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# ── API KEY ────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not set in Streamlit Secrets or environment.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
BIG_MODEL   = "llama-3.3-70b-versatile"   # AI 1 — data cleaning
SMALL_MODEL = "llama-3.1-8b-instant"      # AI 2 — summary


# ── AI FUNCTIONS ───────────────────────────────────────
def ai_clean_data(text: str) -> str | None:
    """Big AI: unstructured text → CSV string"""
    try:
        response = client.chat.completions.create(
            model=BIG_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Data Cleaner. "
                        "Convert the given unstructured data into clean CSV format. "
                        "Return ONLY the CSV text with a header row. "
                        "No explanation, no markdown, no backticks."
                    )
                },
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI Cleaning Error: {e}")
        return None


def ai_summary(data_str: str) -> str:
    """Small AI: 50-word insight from data"""
    try:
        response = client.chat.completions.create(
            model=SMALL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Expert Data Summarizer. "
                        "Summarize key insights from this data in exactly 50 words. "
                        "Use markdown format: bold for important values, bullet points for insights. "
                        "Highlight highest and lowest values clearly."
                    )
                },
                {"role": "user", "content": data_str}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Summary Error: {e}")
        return "Summary failed."


# ── FILE PARSER ────────────────────────────────────────
def parse_file(file):
    """Returns (data, 'structured' | 'unstructured')"""
    try:
        name = file.name
        if name.endswith(".csv"):
            return pd.read_csv(file), "structured"
        elif name.endswith(".xlsx"):
            return pd.read_excel(file), "structured"
        elif name.endswith(".txt"):
            text = file.read().decode("utf-8")
            return text, "unstructured"
        elif name.endswith(".pdf"):
            reader = pypdf.PdfReader(file)
            text = " ".join(
                page.extract_text()
                for page in reader.pages
                if page.extract_text()
            )
            if not text.strip():
                st.warning("PDF text empty — scanned PDF? OCR not supported yet.")
                return None, None
            return text, "unstructured"
        else:
            st.error("Unsupported file type.")
            return None, None
    except Exception as e:
        st.error(f"File Parse Error: {e}")
        return None, None


def csv_str_to_df(csv_str: str) -> pd.DataFrame:
    """AI CSV output → DataFrame"""
    try:
        return pd.read_csv(io.StringIO(csv_str))
    except Exception as e:
        st.warning(f"Could not parse AI output as table: {e}")
        return pd.DataFrame({"AI_Output": [csv_str]})

# ── BAR CHART ──────────────────────────────────────────
def draw_bar_chart(selected_df, numeric_cols):
    """
    Grouped bar chart:
    - X axis = Examples (rows / auto-detected labels)
    - Y axis = Value (numeric columns)
    - Each group = one row, bars = features
    """

    # ---------------- FIX 1: Clean NaN ----------------
    selected_df = selected_df.copy()
    selected_df[numeric_cols] = selected_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    selected_df[numeric_cols] = selected_df[numeric_cols].fillna(0)

    # ---------------- X-axis detect ----------------
    str_cols = selected_df.select_dtypes(include="object").columns.tolist()

    if str_cols:
        x_labels = selected_df[str_cols[0]].astype(str).tolist()
        x_title  = str_cols[0]
    else:
        x_labels = [str(i) for i in range(len(selected_df))]
        x_title  = "Index"

    # ---------------- Setup ----------------
    n_rows = len(selected_df)
    n_cols = len(numeric_cols)

    if n_cols == 0 or n_rows == 0:
        raise ValueError("No valid data for bar chart")

    fig_width = max(14, n_rows * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    x = np.arange(n_rows)
    width = 0.7 / n_cols

    # ---------------- Bars ----------------
    for i, col in enumerate(numeric_cols):
        values = selected_df[col].values

        offsets = x + i * width - (n_cols - 1) * width / 2
        bars = ax.bar(offsets, values, width, label=col)

        for bar, val in zip(bars, values):
            if pd.isna(val) or val == 0:
                continue

            # Feature name inside bar
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val / 2,
                str(col),
                ha="center", va="center",
                rotation=90, fontsize=6,
                color="white", fontweight="bold"
            )

            # Value on top (safe, no int conversion)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (abs(val) * 0.02),
                f"{val:.2f}".rstrip('0').rstrip('.'),
                ha="center", va="bottom",
                fontsize=6.5, color="black"
            )

    # ---------------- Labels ----------------
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(x_title, fontsize=11, labelpad=8)
    ax.set_ylabel("Value / Score", fontsize=11, labelpad=8)
    ax.set_title("Feature Comparison per Example", fontsize=13, fontweight="bold", pad=12)

    ax.legend(title="Features", loc="upper right", fontsize=8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator())

    plt.tight_layout()
    return fig

# ── PIE CHART ──────────────────────────────────────────
def draw_pie_chart(selected_df, numeric_cols):
    """
    Dynamic Pie chart:
    - Multi numeric cols → feature comparison (sum based)
    - Single numeric col → distribution (categorical safe)
    - Otherwise → user-friendly markdown
    """

    # ---------------- FIX 1: Clean NaN ----------------
    selected_df = selected_df.copy()
    selected_df[numeric_cols] = selected_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    selected_df[numeric_cols] = selected_df[numeric_cols].fillna(0)

    # ---------------- VALIDATION ----------------
    if not numeric_cols or len(numeric_cols) == 0:
        st.markdown("""
> **ℹ️ Pie Chart nahi ban sakta**
>
> Koi numeric column detect nahi hua.
> Numeric data select karo (marks, sales, etc.)
        """)
        return

    # ---------------- MULTI COLUMN ----------------
    if len(numeric_cols) >= 2:
        try:
            values = selected_df[numeric_cols].sum()

            # remove zero-only columns
            values = values[values > 0]

            if len(values) == 0:
                raise ValueError("All values are zero")

            fig, ax = plt.subplots(figsize=(7, 5))
            wedges, texts, autotexts = ax.pie(
                values,
                labels=values.index,
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.80
            )

            for at in autotexts:
                at.set_fontsize(8)

            ax.set_title(
                f"Feature Distribution (Base: {', '.join(values.index)})",
                fontsize=11, fontweight="bold"
            )
            ax.set_ylabel("Percentage (%)", fontsize=9)

            
            return fig

        except Exception as e:
            st.markdown(f"""
> **⚠️ Pie Chart Error**
>
> {str(e)}
            """)
            return

    # ---------------- SINGLE COLUMN ----------------
    col = numeric_cols[0]

    try:
        vc = selected_df[col].value_counts()
        vc = vc[vc > 0]

        unique_count = len(vc)

        if unique_count == 0:
            raise ValueError("No valid values")

        if unique_count <= 20:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.pie(
                vc,
                labels=vc.index,
                autopct="%1.1f%%",
                startangle=90
            )

            ax.set_title(
                f"Distribution of '{col}'",
                fontsize=11, fontweight="bold"
            )
            ax.set_ylabel("Percentage (%)", fontsize=9)

            return fig
        else:
            st.markdown(f"""
> **ℹ️ Pie Chart nahi ban sakta — '{col}' column se**
>
> {unique_count} unique values hain (max 20 allowed for clarity)
>
> **Solution:**
> - Data filter karo
> - Ya categorical column select karo
> - Ya multiple features select karo
            """)

    except Exception as e:
        st.markdown(f"""
> **⚠️ Pie Chart Error**
>
> {str(e)}
        """)

return None 

# ── MAIN UI ────────────────────────────────────────────
st.title("🤖 AI Data Analyzer")
st.caption("Upload CSV / Excel / TXT / PDF → AI cleans → Charts → PDF Report")

file = st.file_uploader("📁 Upload File", type=["csv", "xlsx", "txt", "pdf"])

if file:
    data, data_type = parse_file(file)
    if data is None:
        st.stop()

    # Structured vs Unstructured
    if data_type == "unstructured":
        st.info("📄 Unstructured data detected → AI structuring it...")
        with st.spinner("AI 1 (Big Model) cleaning data..."):
            ai_csv = ai_clean_data(data)
        if not ai_csv:
            st.stop()
        with st.expander("🔍 AI Raw CSV Output"):
            st.text(ai_csv)
        df = csv_str_to_df(ai_csv)
    else:
        df = data

    if df is None or df.empty:
        st.error("DataFrame empty. Kuch parse nahi hua.")
        st.stop()

    st.write("### 📊 Data Preview")
    st.dataframe(df.head(10))

    # Optional Column Selection
    all_cols = df.columns.tolist()
    chosen_cols = st.multiselect(
        "🔹 Select Columns (optional — blank = all)",
        options=all_cols
    )
    if not chosen_cols:
        chosen_cols = all_cols

    # Optional Row Selection
    total_rows = len(df)
    use_rows = st.checkbox(f"🔹 Filter Rows? (Total rows: {total_rows})")
    if use_rows and total_rows > 1:
        r_start, r_end = st.slider(
            "Row Range",
            min_value=0,
            max_value=total_rows - 1,
            value=(0, min(49, total_rows - 1))
        )
        selected_df = df[chosen_cols].iloc[r_start: r_end + 1].reset_index(drop=True)
    else:
        selected_df = df[chosen_cols].reset_index(drop=True)

    st.write("### ✅ Selected Data")
    st.dataframe(selected_df)

    numeric_cols = selected_df.select_dtypes(include="number").columns.tolist()
    chart_img_path = None

    if numeric_cols:
        # ── BAR CHART ──
        st.write("### 📊 Bar Chart")
        try:
            fig_bar = draw_bar_chart(selected_df, numeric_cols)
            st.pyplot(fig_bar)
            # Save for PDF
            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig_bar.savefig(tmp_img.name, bbox_inches="tight", dpi=120)
            plt.close(fig_bar)
            chart_img_path = tmp_img.name
        except Exception as e:
            st.error(f"Bar Chart Error: {e}")
        # ── PIE CHART ──
        st.write("### 🥧 Pie Chart")
        pie_chart_img_path = None  # ✅ define पहले

        try:
            fig_pie = draw_pie_chart(selected_df, numeric_cols)

            if fig_pie:  # ✅ agar chart bana hai tabhi save karo
                st.pyplot(fig_pie)

                # Save for PDF
                tmp_img_pie = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig_pie.savefig(tmp_img_pie.name, bbox_inches="tight", dpi=120)
                plt.close(fig_pie)

                pie_chart_img_path = tmp_img_pie.name

        except Exception as e:
            st.error(f"Pie Chart Error: {e}")

    else:
        st.warning("⚠️ No numeric columns found for charts.")
   
    # ── AI SUMMARY (Markdown) ──
    st.write("### 💡 AI Insight")
    with st.spinner("AI 2 (Small Model) summarizing..."):
        summary = ai_summary(selected_df.describe(include="all").to_string())
    st.markdown(summary)

# ── PDF EXPORT ──
if st.button("📄 Generate PDF Report"):
    try:
        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(
            tmp_pdf.name,
            leftMargin=0.5 * inch, rightMargin=0.5 * inch,
            topMargin=0.5 * inch, bottomMargin=0.5 * inch
        )
        styles = getSampleStyleSheet()
        content = []

        # ---------------- TITLE ----------------
        content.append(Paragraph("🤖 AI Data Analysis Report", styles["Title"]))
        content.append(Spacer(1, 0.2 * inch))

        # ---------------- SUMMARY ----------------
        content.append(Paragraph("1. Key Insights", styles["Heading2"]))

        clean_summary = (
            summary
            .replace("**", "")
            .replace("##", "")
            .replace("•", "-")
        )

        for line in clean_summary.split("\n"):
            line = line.strip()
            if line:
                content.append(Paragraph(line, styles["Normal"]))

        content.append(Spacer(1, 0.2 * inch))

        # ---------------- TABLE ----------------
        content.append(Paragraph("2. Data Table", styles["Heading2"]))

        # ❌ OLD: fixed 50 rows
        # ✅ NEW: dynamic chunk (multi-page safe)
        rows_per_page = 40
        total_rows = len(selected_df)

        for start in range(0, total_rows, rows_per_page):
            chunk_df = selected_df.iloc[start:start + rows_per_page]

            table_data = [chunk_df.columns.tolist()] + chunk_df.astype(str).values.tolist()

            col_count = len(chunk_df.columns)
            available_width = 7.0 * inch
            col_width = available_width / col_count

            pdf_table = Table(table_data, colWidths=[col_width] * col_count, repeatRows=1)

            pdf_table.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#4472C4")),
                ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
                ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, -1), 7),
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#DCE6F1"), colors.white]),
                ("GRID",          (0, 0), (-1, -1), 0.3, colors.grey),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))

            content.append(pdf_table)
            content.append(Spacer(1, 0.2 * inch))

        # ---------------- BAR CHART ----------------
        if chart_img_path:
            content.append(Paragraph("3. Bar Chart", styles["Heading2"]))
            content.append(RLImage(chart_img_path, width=6.5 * inch, height=3.2 * inch))
            content.append(Spacer(1, 0.2 * inch))

        # ---------------- PIE CHART ----------------
        if pie_chart_img_path:
            content.append(Paragraph("4. Pie Chart", styles["Heading2"]))
            content.append(RLImage(pie_chart_img_path, width=5.5 * inch, height=3.5 * inch))
            content.append(Spacer(1, 0.2 * inch))

        # ---------------- BUILD ----------------
        doc.build(content)

        with open(tmp_pdf.name, "rb") as f:
            st.download_button(
                "⬇️ Download PDF",
                f,
                file_name="ai_report.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"PDF Error: {e}")
