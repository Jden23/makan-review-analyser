# 🍜 Makan Review Analyser

**Which hawker foods do people love or hate, and where are the "bad food" hot spots in Singapore?**

A mini project that combines two NVIDIA Deep Learning Institute courses:

| Course | What this project uses from it |
|---|---|
| 📗 **Fundamentals of Accelerated Data Science** (RAPIDS) | `cudf.pandas`, small data types, ETL + Parquet, `groupby`, binning, pivot tables, charts, **DBSCAN** clustering |
| 📘 **Building Transformer-Based NLP Applications** | **BERT**, WordPiece tokenizer, **fine-tuning** for text classification, train/dev/test, precision / recall / **F1**, mixed precision (FP16) |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jden23/makan-review-analyser/blob/main/makan_review_analyser.ipynb)

---

## 🔄 How it works

```
CSV: 20,200 messy food reviews
  │
  ├─ 1. Load fast on the GPU (cudf.pandas + category / float32 types)     📗 Lessons 1-2
  ├─ 2. Clean the data (ETL) and save as Parquet                          📗 Lesson 4
  ├─ 3. groupby + charts: best/worst dishes and areas                     📗 Lesson 3
  ├─ 4. Fine-tune BERT: read a review → positive 😊 / negative 😞         📘 Lessons 2, 4
  ├─ 5. Test BERT (F1 score) + try new reviews                            📘 Lesson 2, 📗 Lesson 7
  └─ 6. DBSCAN on the reviews BERT says are negative → hot spots map 🗺️   📗 Lessons 2, 6
```

**The key idea:** BERT reads the **text** and decides if a review is bad.
Then the GPU data tools find **where** the bad reviews are packed together.

## 📁 Files

| File | What it is |
|---|---|
| `makan_review_analyser.ipynb` | the main notebook (run this in Google Colab) |
| `data/makan_reviews.csv` | the dataset (20,200 reviews) |
| `data/generate_data.py` | the script that made the dataset |

## ▶️ How to run

1. Click the **Open in Colab** button above.
2. In Colab: **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**. It takes about 10 minutes (BERT training takes most of the time).

## 📊 The data

> ⚠️ **The data is simulated** (fake but realistic). It was made with `data/generate_data.py`.
> Stall names are made up, and the locations are rough neighbourhood centres, not real businesses.

| Column | Example |
|---|---|
| `area` | Toa Payoh |
| `stall` | Ah Huat Chicken Rice |
| `dish` | chicken rice |
| `price` | $4.50 |
| `stars` | 5 |
| `review` | "So good! The chicken rice was great, the rice was fragrant." |
| `lat`, `long` | 1.3343, 103.8563 |

The data is **messy on purpose**, to practise cleaning: duplicate rows, missing values, dish names in different spellings (`"  Chicken Rice "`, `"CHICKEN RICE"`), and prices as text (`"$4.50"`).
About 4% of the star ratings don't match the review words (like real people!), so BERT can't get 100%.

## 🧠 Main concepts (in simple words)

- **cudf.pandas**: one line (`%load_ext cudf.pandas`) makes normal pandas code run on the GPU.
- **ETL**: **E**xtract (load) → **T**ransform (clean) → **L**oad (save as Parquet, which is smaller and faster than CSV).
- **groupby**: split the table into groups → do maths on each → join the answers ("average stars per dish").
- **BERT**: a pretrained Transformer encoder that already knows English. We **fine-tune** it on our reviews so it learns positive vs negative.
- **WordPiece tokenizer**: cuts unknown words into pieces with `##` (e.g. Singlish words like *shiok*).
- **F1 score**: a mix of precision (when it says "negative", is it right?) and recall (did it find all the negatives?).
- **DBSCAN**: finds places where many points are packed close together. `eps` = max distance, `min_samples` = how many are needed, `-1` = noise.

## 💡 Ideas to make it better

- **KNN**: find the nearest "good" stall to each hot spot (📗 Lesson 7)
- **XGBoost**: predict the stars from price + area + dish (📗 Lesson 8)
- **Triton**: put the BERT model on a server, so an app can use it (📘 Lesson 6)
