# 🤖 AI Data Analyzer

> Upload any data file → AI cleans & structures it → Visualize with Charts → Export PDF Report

A multi-model AI-powered data analysis web app built with **Streamlit** and **Groq LLMs**.  
No manual data cleaning needed — just upload and get instant insights.

---

## 🚀 Live Demo

Click here : 👇

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-data-analyzer-rp.streamlit.app/)

st.info("⚡ First load may take few seconds (free hosting)")

---

## 📌 Features

- ✅ Supports **CSV, Excel (.xlsx), TXT, PDF** file formats
- ✅ **AI-powered data cleaning** — unstructured text → structured table automatically
- ✅ **Optional** column & row selection (filter your view)
- ✅ **Grouped Bar Chart** — per-example feature comparison
- ✅ **Pie Chart** — feature distribution with smart fallback messages
- ✅ **AI-generated insights** in markdown format
- ✅ **PDF Export** — Table + Bar Chart + Pie Chart + AI Summary in one file

---

## 🧠 AI Pipeline — How It Works

This app uses a **2-model agentic pipeline**, where each AI agent has a specific role — similar to how Agentic AI systems assign specialized tasks to different models.

```
User Upload
     │
     ▼
┌─────────────────────────────┐
│      File Parser            │  pandas / pypdf / openpyxl
│  CSV / XLSX → DataFrame     │  (structured → skip AI)
│  TXT / PDF  → Raw Text      │  (unstructured → send to AI 1)
└────────────┬────────────────┘
             │ unstructured only
             ▼
┌─────────────────────────────┐
│  🤖 AI Agent 1 — Cleaner    │  llama-3.3-70b-versatile
│  Raw Text → CSV Format      │  (Big Model — complex reasoning)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   User Interaction          │  Streamlit UI
│   Select Columns (optional) │
│   Filter Rows   (optional)  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Chart Engine              │  matplotlib / numpy
│   → Grouped Bar Chart       │
│   → Pie Chart               │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  🤖 AI Agent 2 — Summarizer │  llama-3.1-8b-instant
│  DataFrame Stats → Insight  │  (Small Model — fast & efficient)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   PDF Export                │  reportlab
│   Table + Charts + Summary  │
└─────────────────────────────┘
```

---

## 🤖 Model Details

| Agent | Role | Model | Why This Model |
|---|---|---|---|
| **AI Agent 1** | Data Cleaner | `llama-3.3-70b-versatile` | Complex task — needs to understand messy text, detect columns, and convert to structured CSV. Big model = better reasoning. |
| **AI Agent 2** | Summarizer | `llama-3.1-8b-instant` | Simple task — 50-word insight from stats. Small fast model = lower latency, same quality. |

> **Why 2 different models?**  
> Using a big model for every task wastes API quota and increases response time.  
> Assigning the right model to the right task is a core principle of **Agentic AI design**.

---

## 🗂️ Project Structure

```
ai-data-analyzer/
│
├── app.py               # Main Streamlit app (single file)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🛠️ Tech Stack

| Layer | Tool | Version |
|---|---|---|
| UI Framework | Streamlit | latest |
| AI / LLM | Groq API | latest |
| Data Handling | Pandas, NumPy | latest |
| Charts | Matplotlib | latest |
| PDF Parsing | pypdf | latest |
| Excel Parsing | openpyxl | latest |
| PDF Export | ReportLab | latest |

---

## ⚙️ Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-data-analyzer.git
cd ai-data-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key

Get a free API key from [console.groq.com](https://console.groq.com)

**Option A — Environment variable (local):**
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

**Option B — `.env` file:**
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → select `app.py`
4. Go to **Settings → Secrets** and add:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
5. Click **Deploy** ✅

---

## 📊 Supported File Formats

| Format | Data Type | AI Processing |
|---|---|---|
| `.csv` | Structured | ❌ Direct load |
| `.xlsx` | Structured | ❌ Direct load |
| `.txt` | Unstructured | ✅ AI Agent 1 cleans |
| `.pdf` | Unstructured | ✅ AI Agent 1 cleans |

> **Note:** Scanned PDFs (image-based) are not supported — only text-layer PDFs work.

---

## 📄 PDF Export — What's Included

The downloaded PDF report contains:

1. **AI Key Insights** — markdown-formatted 50-word summary
2. **Data Table** — full selected data (paginated, 40 rows per page)
3. **Bar Chart** — grouped feature comparison
4. **Pie Chart** — feature distribution (only if applicable)

---

## ⚠️ Limitations

- TXT/PDF → Table conversion accuracy depends on how structured the raw text is
- Pie Chart requires either **multiple numeric columns** or a **categorical column with ≤ 20 unique values**
- Scanned PDFs (image-only) are not supported — use text-based PDFs
- Free Groq API has rate limits — large files may hit token limits

---

## 🔮 Planned Upgrades

- [ ] OCR support for scanned PDFs (Tesseract)
- [ ] Auto chart-type recommendation using AI
- [ ] Multi-page PDF with cover page
- [ ] Better Plotly-based interactive charts
- [ ] Support for JSON and XML files

---

## 👨‍💻 Author

**Rahul Prasad** — AI/ML Learner | Building real-world AI projects  
📍 Jamshedpur, India  
🔗 [GitHub](https://github.com/Rahul543-ux)

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

