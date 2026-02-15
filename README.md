# 🧾 ReceiptIQ — AI-Powered Receipt Analyzer with LLM Insights

## 📌 Project Overview

ReceiptIQ is an end-to-end AI-powered system that extracts information from receipt images using OCR, categorizes expenses, analyzes spending patterns, and provides personalized budgeting advice via a Large Language Model (LLM).

**Live Demo:** Run `streamlit run app.py` and open `http://localhost:8501`

---

## 🏗️ Architecture & Pipeline

```
Receipt Image → Image Preprocessing → OCR Text Extraction → Data Parsing
     → Expense Categorization → Spending Analysis → LLM Financial Advice
```

| Stage | Module | Technology |
|-------|--------|-----------|
| 1. Image Processing | `image_processor.py` | OpenCV, Pillow |
| 2. OCR Extraction | `ocr_engine.py` | EasyOCR |
| 3. Data Parsing | `data_parser.py` | Regex, Custom Logic |
| 4. Categorization | `categorizer.py` | Keyword Matching (100+ terms) |
| 5. Spending Analysis | `analyzer.py` | NumPy, Statistical Methods |
| 6. LLM Advice | `llm_advisor.py` | Google Gemini 2.0 Flash |
| 7. Web Interface | `app.py` | Streamlit, Plotly |

---

## ❓ Interview / Hackathon Q&A Preparation

### Q1: Which OCR did you use and why?

**Answer:** We used **EasyOCR** (Python library built on PyTorch with CRAFT text detection + CRNN recognition).

**Why EasyOCR over Tesseract:**
- **Better out-of-the-box accuracy** on real-world photos (camera angles, shadows, blur)
- **Deep learning based** — uses CRAFT for text detection and CRNN for recognition, unlike Tesseract's classical approach
- **No system-level installation** needed (Tesseract requires separate binary install)
- **Supports 80+ languages** natively
- **Returns bounding boxes + confidence scores**, which we use for visualization and quality filtering

---

### Q2: What image preprocessing techniques did you apply and why?

**Answer:** We apply a 7-step preprocessing pipeline in `image_processor.py`:

| Step | Technique | Purpose |
|------|-----------|---------|
| 1 | **Resizing** | Scale small images up to 2000px height for better OCR |
| 2 | **Grayscale conversion** | Reduce complexity from 3 channels to 1 |
| 3 | **Noise reduction** | `cv2.fastNlMeansDenoising` removes camera noise while preserving edges |
| 4 | **CLAHE contrast enhancement** | Adaptive histogram equalization makes faded text readable |
| 5 | **Deskewing** | Corrects rotation up to ±15° using `cv2.minAreaRect` |
| 6 | **Sharpening** | Laplacian kernel makes text edges crisper |
| 7 | **Adaptive thresholding** | Gaussian thresholding creates clean binary image for OCR |

We run OCR on **both** the sharpened image and thresholded image, then pick whichever gives higher confidence scores.

---

### Q3: How do you handle OCR errors and noisy text?

**Answer:** Multiple layers of error handling:

1. **Dual OCR pass** — We OCR both the enhanced and thresholded versions of the image and pick the best result based on average confidence scores
2. **Line grouping** — OCR blocks are grouped into logical lines using adaptive y-position thresholds (auto-calculated from text height)
3. **Skip patterns** — We filter out 20+ non-item patterns (totals, dates, store headers, tax lines, separator lines)
4. **Weight line detection** — Lines like `1.14 lb @ 2.49/lb` are recognized as weight descriptions, not items
5. **Multi-line merging** — If an item name is on one line and its price on the next, they get merged
6. **Name validation** — Items must have at least 2 letters to be valid (rejects pure numbers)

---

### Q4: How does the data parsing work?

**Answer:** The `DataParser` class in `data_parser.py`:

1. **Extracts the rightmost price** from each line (receipts always put prices on the right)
2. **Removes tax flags** (`F`, `T`, `N`, `B`) common on US receipts
3. **Skips weight/quantity lines** like `3 @ 0.58` or `1.14 lb @ 2.49/lb` (these are descriptions, not items)
4. **Merges multi-line items** — detects when an item name has no price and the next line has a price
5. **Extracts store info** from header and **totals** from footer using regex
6. **Validates** that every item has a name (2+ letters) and a reasonable price ($0.01–$9999.99)

---

### Q5: How do you categorize items?

**Answer:** We use **keyword-based classification** with 100+ keywords across 10 categories in `categorizer.py`:

**Categories:** Dairy, Meat & Seafood, Fruits & Vegetables, Bakery, Beverages, Snacks, Canned & Packaged, Frozen Foods, Household, Health & Personal Care, Other

**Key features:**
- Handles **receipt abbreviations**: `BNLS` = boneless, `CHICK` = chicken, `BURG` = burger, `TOM/PASTE` = tomato paste, `W/G WHEAT` = whole grain wheat
- Uses **longest-match-first** strategy for specificity (e.g., "green bell" matches Fruits & Vegetables, not just "green")
- Falls back to **partial word matching** if exact substring match fails
- Each category has an assigned **color** for consistent chart visualization

---

### Q6: How does the spending analysis work?

**Answer:** The `SpendingAnalyzer` class performs:

1. **Basic statistics** — mean, median, min, max, standard deviation of prices using NumPy
2. **Overspending detection** — compares spending % per category against built-in benchmarks (e.g., Meat should be ~20%). Flags categories that exceed benchmark by 30%+
3. **Price anomaly detection** — uses **z-score analysis** (items with |z| > 1.5 are flagged as unusually expensive or cheap)
4. **Health score calculation** — 0-100 composite score based on:
   - Category diversity (+5 for 4+ categories)
   - Healthy food percentage (fruits, vegetables, dairy)
   - Overspending penalties (-4 to -8 per alert)
   - Snack/beverage spending penalty
5. **Savings opportunities** — identifies categories where 15% reduction could save money

---

### Q7: Which LLM did you use and how?

**Answer:** We use **Google Gemini 2.0 Flash** via the `google-generativeai` Python SDK.

**How it works:**
1. We build a **structured prompt** containing: total spent, items count, category breakdown with percentages, top expensive items, overspending alerts, and health score
2. The prompt asks Gemini for: overall assessment, savings tips, nutrition insights, monthly projection, and top 3 action items
3. The API key is stored in a **`.env` file** (not hardcoded) and loaded via `python-dotenv`
4. If the API key is missing or the call fails, the system **falls back to built-in rule-based advice** — the app never crashes

**Why Gemini:**
- Free tier available (sufficient for demo)
- Fast response times
- Good at structured financial analysis
- Simple Python SDK integration

---

### Q8: How does the fallback system work when LLM is unavailable?

**Answer:** The `_generate_fallback_advice()` method generates comprehensive advice without any API call:
- Overall assessment based on health score
- Generic savings tips (store brands, bulk buying, shopping lists)
- Monthly/yearly budget projection (weekly total × 4.3 and × 52)
- Category-specific overspending alerts
- Action items for improvement

This ensures the app is **fully functional even without an API key**.

---

### Q9: What visualization techniques did you use?

**Answer:** We use **Plotly** for interactive charts and **Streamlit** for the UI:

| Chart | Purpose |
|-------|---------|
| **Donut chart** | Spending breakdown by category with pull-out effect |
| **Horizontal bar chart** | Category comparison with amounts and percentages |
| **Treemap** | Hierarchical view of items within categories |
| **Gauge chart** | Spending health score (0-100) with color zones |
| **Bar chart** | Top 5 most expensive items |
| **Image pipeline** | Side-by-side view of each preprocessing step |
| **OCR overlay** | Bounding boxes color-coded by confidence (green/yellow/red) |

All charts use a **dark theme** with consistent color palette.

---

### Q10: What was the biggest challenge and how did you solve it?

**Answer:** Three main challenges:

1. **Real receipt parsing** — Receipts have multi-line items, weight-based pricing (`1.14 lb @ 2.49/lb`), tax flags, and abbreviations. We solved this with multi-line merging, weight line detection, and 100+ keyword mappings.

2. **Unicode encoding on Windows** — EasyOCR's progress bars use Unicode characters that crash on Windows. We solved this with a context manager that temporarily redirects stdout/stderr to devnull during OCR execution.

3. **OCR line grouping** — Text blocks from the same receipt line had slightly different y-positions in photos. We solved this with adaptive y-threshold calculation based on actual text bounding box heights.

---

### Q11: What is your tech stack?

| Layer | Technology | Version |
|-------|-----------|---------|
| **Frontend** | Streamlit | 1.31+ |
| **Image Processing** | OpenCV, Pillow | 4.9+, 10.2+ |
| **OCR** | EasyOCR (CRAFT + CRNN) | 1.7+ |
| **Data Analysis** | Pandas, NumPy | 2.2+, 1.26+ |
| **Visualization** | Plotly | 5.18+ |
| **LLM** | Google Gemini 2.0 Flash | via google-generativeai SDK |
| **Config** | python-dotenv | 1.0+ |
| **Language** | Python | 3.9+ |

---

### Q12: How is the project structured?

```
receipt-analyzer/
├── app.py                  # Main Streamlit application (UI + orchestration)
├── image_processor.py      # Image preprocessing pipeline (OpenCV)
├── ocr_engine.py           # OCR extraction engine (EasyOCR)
├── data_parser.py          # Receipt text parsing & structuring
├── categorizer.py          # Expense categorization (100+ keywords)
├── analyzer.py             # Spending analysis & anomaly detection
├── llm_advisor.py          # LLM integration (Google Gemini)
├── requirements.txt        # Python dependencies
├── .env                    # API key (GOOGLE_API_KEY=your_key)
└── .streamlit/
    └── config.toml         # Streamlit theme configuration
```

Each module is **independent and testable** — they communicate through dictionaries, following a clean pipeline pattern.

---

### Q13: How do you ensure the app doesn't crash?

**Answer:** Error handling at every stage:
- Each pipeline step is wrapped in `try/except` with user-friendly error messages
- LLM has automatic **fallback to built-in analysis**
- Image loading validates format before processing
- OCR stdout is safely redirected to prevent encoding crashes
- Session state management prevents stale data issues
- `st.stop()` prevents cascading errors if an early step fails

---

### Q14: What would you improve with more time?

1. **Multi-receipt comparison** — Upload multiple receipts to track spending over time
2. **Receipt database** — SQLite/PostgreSQL to store historical receipt data
3. **Smart OCR** — Use Gemini Vision API for direct receipt understanding (skip traditional OCR)
4. **Budget alerts** — Set monthly category budgets and get notifications
5. **Export to Excel** — Detailed spending reports with charts
6. **Mobile camera** — Direct camera capture in the web app
7. **Multi-language support** — EasyOCR already supports 80+ languages

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Gemini API key in .env file
# GOOGLE_API_KEY=your_actual_api_key_here

# 3. Run the app
streamlit run app.py
```

---

## 📦 Dependencies

- **streamlit** — Web application framework
- **opencv-python** — Image preprocessing
- **Pillow** — Image loading and format handling
- **easyocr** — Optical Character Recognition
- **numpy** — Numerical computations
- **pandas** — Data structuring and tables
- **plotly** — Interactive visualizations
- **google-generativeai** — Google Gemini LLM SDK
- **python-dotenv** — Environment variable management

