import os
import io
import tempfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pypdf
from groq import Groq
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ── API KEY (env se aayegi, hardcode nahi) ──
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not set in environment/secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

BIG_MODEL   = "llama-3.3-70b-versatile"  # AI 1: cleaning
SMALL_MODEL = "llama-3.1-8b-instant"     # AI 2: summary

# ── AI FUNCTIONS ──────────────────────────────────────
def ai_clean_data(text: str) -> str | None:
    """Big AI: unstructured text → CSV format string"""
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
                        "You are an Expert Summerizer. "
                        "Summarize key insights from this data in exactly 50 words. Highlight high/low values. Be concise."
                    )
                },
                {"role": "user", "content": data_str}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Summary Error: {e}")
        return "Summary failed."


# ── FILE PARSER ───────────────────────────────────────
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
    """AI ke CSV output ko DataFrame mein convert karo"""
    try:
        return pd.read_csv(io.StringIO(csv_str))
    except Exception as e:
        st.warning(f"Could not parse AI output as table: {e}")
        # fallback — raw text dikhao
        return pd.DataFrame({"AI_Output": [csv_str]})


# ── MAIN UI ───────────────────────────────────────────
st.title("🤖 AI Data Analyzer")
st.caption("Upload CSV / Excel / TXT / PDF → AI cleans → Charts → PDF Report")

file = st.file_uploader("📁 Upload File", type=["csv", "xlsx", "txt", "pdf"])

if file:
    data, data_type = parse_file(file)

    if data is None:
        st.stop()

    # ── STRUCTURED vs UNSTRUCTURED ──
    if data_type == "unstructured":
        st.info("📄 Unstructured data detected → AI structuring it...")
        with st.spinner("AI 1 (Big Model) cleaning data..."):
            ai_csv = ai_clean_data(data)
        if not ai_csv:
            st.stop()
        with st.expander("AI Raw CSV Output"):
            st.text(ai_csv)
        df = csv_str_to_df(ai_csv)
    else:
        df = data  # already a DataFrame

    if df is None or df.empty:
        st.error("DataFrame empty. Kuch parse nahi hua.")
        st.stop()

    st.write("### 📊 Data Preview")
    st.dataframe(df.head(10))

    # ── OPTIONAL COLUMN SELECTION ──
    all_cols = df.columns.tolist()
    chosen_cols = st.multiselect(
        "🔹 Select Columns (optional — blank = all)",
        options=all_cols
    )
    if not chosen_cols:
        chosen_cols = all_cols  # default: all

    # ── OPTIONAL ROW SELECTION ──
    total_rows = len(df)
    use_rows = st.checkbox(f"🔹 Filter Rows? (Total rows: {total_rows})")
    if use_rows and total_rows > 1:
        r_start, r_end = st.slider(
            "Row Range",
            min_value=0,
            max_value=total_rows - 1,
            value=(0, min(49, total_rows - 1))
        )
        selected_df = df[chosen_cols].iloc[r_start : r_end + 1]
    else:
        selected_df = df[chosen_cols]

    st.write("### ✅ Selected Data")
    st.dataframe(selected_df)

    # ── CHARTS (numeric only) ──
    numeric_cols = selected_df.select_dtypes(include="number").columns.tolist()

    chart_img_path = None

    if numeric_cols:
        st.write("### 📈 Line Chart")
        try:
            st.line_chart(selected_df[numeric_cols])
        except Exception as e:
            st.error(f"Line Chart Error: {e}")

        if len(numeric_cols) == 1:
            st.write("### 🥧 Pie Chart")
            try:
                fig, ax = plt.subplots()
                selected_df[numeric_cols[0]].value_counts().plot.pie(
                    ax=ax, autopct="%1.1f%%"
                )
                ax.set_ylabel("")
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Pie Chart Error: {e}")

        # Save chart image for PDF
        try:
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            selected_df[numeric_cols].plot(ax=ax2)
            ax2.set_title("Data Trend")
            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig2.savefig(tmp_img.name, bbox_inches="tight")
            plt.close(fig2)
            chart_img_path = tmp_img.name
        except Exception:
            chart_img_path = None
    else:
        st.warning("⚠️ No numeric columns found for charts.")

    # ── AI SUMMARY ──
    st.write("### 💡 AI Insight")
    with st.spinner("AI 2 (Small Model) summarizing..."):
        summary = ai_summary(selected_df.describe(include="all").to_string())
    st.info(summary)

    # ── PDF EXPORT ──
    if st.button("📄 Generate PDF Report"):
        try:
            tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            doc = SimpleDocTemplate(tmp_pdf.name)
            styles = getSampleStyleSheet()
            content = []

            content.append(Paragraph("AI Data Analysis Report", styles["Title"]))
            content.append(Spacer(1, 0.2 * inch))
            content.append(Paragraph("Key Insights", styles["Heading2"]))
            content.append(Paragraph(summary, styles["Normal"]))
            content.append(Spacer(1, 0.2 * inch))

            if chart_img_path:
                content.append(Paragraph("Data Chart", styles["Heading2"]))
                content.append(RLImage(chart_img_path, width=5 * inch, height=2.5 * inch))

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
