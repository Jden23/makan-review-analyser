"""Builds makan_review_analyser.ipynb (run: python build_notebook.py)"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

REPO_RAW = "https://raw.githubusercontent.com/Jden23/makan-review-analyser/main/data/makan_reviews.csv"

md(r"""
# 🍜 Makan Review Analyser
**Which hawker foods do people love or hate, and where are the "bad food" hot spots in Singapore?**

This mini project combines **2 NVIDIA DLI courses**:

| Course | What I use from it |
|---|---|
| 📗 **Fundamentals of Accelerated Data Science** (RAPIDS) | `cudf.pandas`, data types, ETL + Parquet, `groupby`, charts, **DBSCAN** clustering |
| 📘 **Building Transformer-Based NLP Applications** | **BERT**, WordPiece tokenizer, **fine-tuning** for text classification, precision / recall / **F1**, mixed precision |

**Pipeline:**
```
CSV (20,200 messy reviews)
  → 1. load fast on GPU (cudf.pandas + small dtypes)
  → 2. clean (ETL) + save as Parquet
  → 3. groupby + charts (which dishes / areas are best and worst?)
  → 4. fine-tune BERT to read a review → positive 😊 / negative 😞
  → 5. test BERT (F1 score) + try my own reviews
  → 6. DBSCAN on the reviews BERT says are negative → "bad food hot spots" on the map 🗺️
```

> ⚠️ **The data is SIMULATED** (fake but realistic), made with `data/generate_data.py`.
> Stall names are made up, and the locations are rough neighbourhood centres, not real businesses.

### ▶️ How to run (Google Colab, free GPU)
1. **Runtime → Change runtime type → T4 GPU**
2. **Runtime → Run all** (about 10 minutes in total)
""")

# ---------------- Step 0 ----------------
md(r"""
## Step 0: Setup
Check that we have a GPU, and turn on **`cudf.pandas`**, so normal pandas code runs on the GPU.
📗 *GPU course, Lesson 1:* `%load_ext cudf.pandas` must come **before** `import pandas`.
""")
code(r"""
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")
code(r"""
# Turn on cuDF pandas (GPU). If there is no GPU, we just use normal pandas (CPU).
try:
    %load_ext cudf.pandas
    print("✅ cudf.pandas is ON: pandas code will run on the GPU")
except Exception as e:
    print("⚠️ cudf.pandas not available, using normal pandas on CPU:", e)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
""")
code(r"""
# cuML = machine learning on the GPU (GPU course, Lesson 6). Colab usually has it already.
try:
    import cuml
except ImportError:
    !pip install -q cuml-cu12 --extra-index-url=https://pypi.nvidia.com

try:
    from cuml.cluster import DBSCAN
    print("✅ using cuML DBSCAN (GPU)")
except Exception:
    from sklearn.cluster import DBSCAN
    print("⚠️ using scikit-learn DBSCAN (CPU)")
""")

# ---------------- Step 1 ----------------
md(r"""
## Step 1: Load the data (with small data types)
📗 *GPU course, Lessons 1 + 2.*
GPU memory is small, so we set **small data types** when loading:
- `area` has only 12 different values → **`category`** (stores each word once + a small code per row)
- `lat` / `long` → **`float32`** (half the size of `float64`)

`stars` and `price` are messy (some are missing, and prices are text like `"$4.50"`), so we load them as text first and clean them in Step 2.
""")
code(rf"""
DATA_URL = "{REPO_RAW}"

dtype_dict = {{
    'review_id': 'int32',
    'area': 'category',
    'lat': 'float32',
    'long': 'float32',
    'stars': 'str',
    'price': 'str',
}}

start = time.time()
df = pd.read_csv(DATA_URL, dtype=dtype_dict)
print(f"Loaded {{len(df):,}} rows in {{time.time() - start:.2f}} s")
df.head()
""")
code(r"""
# How much memory does each column use? (GPU course, Lesson 2)
mem = df.memory_usage(deep=True)
print(mem)
print(f"\nTotal: {mem.sum() / 1024**2:.2f} MB")
""")

# ---------------- Step 2 ----------------
md(r"""
## Step 2: Clean the data (ETL) + save as Parquet
📗 *GPU course, Lesson 4: **E**xtract → **T**ransform → **L**oad.*

The data is messy on purpose. Problems to fix:
1. **Duplicate rows** (copy-paste mistakes)
2. **Missing** review text or stars
3. Dish names with **extra spaces / CAPITALS** (`"  Chicken Rice "`, `"CHICKEN RICE"`)
4. **Price is text** (`"$4.50"`) → change it into a number
""")
code(r"""
print("Before cleaning:", len(df), "rows")
print("Different dish spellings:", df['dish'].nunique())

# 1. remove duplicate rows
df = df.drop_duplicates()

# 2. remove rows with a missing review or missing stars
df = df[df['review'].notna() & (df['review'].str.strip() != '')]
df = df[df['stars'].notna() & (df['stars'].str.strip() != '')]

# 3. fix dish names: remove extra spaces + make lowercase
df['dish'] = df['dish'].str.strip().str.lower()

# 4. "$4.50" -> 4.50 (number), and stars -> small int
df['price'] = df['price'].str.replace('$', '', regex=False).astype('float32')
df['stars'] = df['stars'].astype('float32').astype('int8')

df = df.reset_index(drop=True)
print("After cleaning: ", len(df), "rows")
print("Different dish spellings:", df['dish'].nunique(), "->", sorted(df['dish'].unique().tolist()))
""")
md(r"""
### Make the label for BERT
- ⭐ 4–5 stars → **positive (1)**
- ⭐ 1–2 stars → **negative (0)**
- ⭐ 3 stars = mixed, so we **don't use** them for training BERT

This is like the BERT course labels (cancer = 0, neurological = 1, other = 2), but with 2 groups.
""")
code(r"""
df['dish'] = df['dish'].astype('category')       # only 10 dishes -> category (Lesson 2)
df['date'] = pd.to_datetime(df['date'])
df['label'] = (df['stars'] >= 4).astype('int8')  # 1 = positive, 0 = negative
df['is_mixed'] = df['stars'] == 3

print(df['label'][~df['is_mixed']].value_counts().rename({1: 'positive', 0: 'negative'}))
""")
code(r"""
# LOAD step: save the clean data as Parquet (smaller + faster than CSV, Lesson 4)
df.to_parquet('makan_clean.parquet', index=False)
!ls -lh makan_clean.parquet

# Parquet can read only the columns we need:
small = pd.read_parquet('makan_clean.parquet', columns=['dish', 'stars'])
small.head(3)
""")

# ---------------- Step 3 ----------------
md(r"""
## Step 3: Explore with groupby + charts
📗 *GPU course, Lesson 3: Split → Apply → Combine.*
""")
code(r"""
# Average stars for each dish (best first)
dish_stars = df.groupby('dish', observed=True)['stars'].mean().sort_values(ascending=False)
print(dish_stars.round(2))

dish_stars.plot(kind='bar', figsize=(9, 4), color='orange', title='Average stars per dish')
plt.ylabel('average stars'); plt.ylim(1, 5); plt.show()
""")
code(r"""
# Bad review rate for each area = (number of 1-2 star reviews) / (all reviews in the area)
df['is_bad'] = (df['stars'] <= 2).astype('float32')
area_bad = df.groupby('area', observed=True)['is_bad'].mean().sort_values(ascending=False)
print((area_bad * 100).round(1).astype(str) + ' %')

(area_bad * 100).plot(kind='bar', figsize=(9, 4), color='tomato', title='% of bad reviews per area')
plt.ylabel('% bad (1-2 stars)'); plt.show()
""")
code(r"""
# Binning (Lesson 3): put prices into buckets, then compare stars
df['price_bucket'] = pd.cut(df['price'], bins=[0, 3, 5, 7, 10],
                            labels=['<= $3', '$3-5', '$5-7', '$7-10'])
df.groupby('price_bucket', observed=True)['stars'].agg(['count', 'mean']).round(2)
""")
code(r"""
# Pivot table (Lesson 3): average stars per area (rows) x dish (columns)
pivot = df.pivot_table(index='area', columns='dish', values='stars', aggfunc='mean', observed=True).round(1)
pivot
""")

# ---------------- Step 4 ----------------
md(r"""
## Step 4: Fine-tune BERT to read reviews 😊 / 😞
📘 *BERT course, Lessons 2 + 4.*

- **BERT** = the **encoder** part of the Transformer. It was already **pretrained** on Wikipedia + books, so it knows English.
- We **fine-tune** it: add a small layer for **2 classes** (negative 0 / positive 1) and train a little more on our reviews.
- In the course we used **NeMo**. Here we use **Hugging Face Transformers**, because it's easier to set up in Colab. The ideas are the same:

| NeMo setting (course) | Here |
|---|---|
| `model.language_model.pretrained_model_name=bert-base-cased` | `MODEL_NAME = "bert-base-cased"` |
| `model.dataset.num_classes=2` | `num_labels=2` |
| `model.dataset.max_seq_length=128` | `MAX_SEQ_LENGTH = 64` (reviews are short) |
| `model.train_ds.batch_size=16` | `BATCH_SIZE = 32` |
| `trainer.max_epochs=10` | `MAX_EPOCHS = 2` |
| `trainer.amp_level=O1`, `trainer.precision=16` | `torch.autocast(..., dtype=torch.float16)` (mixed precision) |
""")
code(r"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Training on:", device)

MODEL_NAME = "bert-base-cased"
MAX_SEQ_LENGTH = 64
BATCH_SIZE = 32
MAX_EPOCHS = 2
LR = 2e-5          # learning rate: small steps, so we don't "forget" what BERT already knows

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Vocabulary size:", tokenizer.vocab_size)
""")
md(r"""
### The WordPiece tokenizer
📘 *BERT course, Lesson 4.* Words that BERT doesn't know get cut into **subwords** with `##`.
Singlish words like *shiok* and *jialat* are not in BERT's vocabulary, so they get cut up!
""")
code(r"""
for text in ["The chicken rice was shiok!", "Jialat, the laksa was watery.", "Hello hello"]:
    print(f"{text!r:35} -> {tokenizer.tokenize(text)}")

# Words -> numbers (IDs). "cased" keeps capitals, so Hello and hello have different IDs:
print("\nHello:", tokenizer.convert_tokens_to_ids("Hello"), " hello:", tokenizer.convert_tokens_to_ids("hello"))
""")
md(r"""
### Train / dev / test split
📘 Like `train.tsv` / `dev.tsv` / `test.tsv` in the course (and 📗 Lesson 7, train/test split):
- **train (80%)** = the textbook the model studies 📖
- **dev (10%)** = practice tests during training ✏️
- **test (10%)** = the real exam, used only at the end 📝
""")
code(r"""
labelled = df[~df['is_mixed']]                  # no 3-star reviews
texts = labelled['review'].tolist()
labels = labelled['label'].astype('int64').tolist()

X_train, X_tmp, y_train, y_tmp = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
X_dev, X_test, y_dev, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print(f"train: {len(X_train):,}   dev: {len(X_dev):,}   test: {len(X_test):,}")
""")
code(r"""
def make_batches(texts, labels=None, batch_size=BATCH_SIZE, shuffle=False):
    # turns text into BERT input (token IDs + attention mask), in batches
    order = np.random.permutation(len(texts)) if shuffle else np.arange(len(texts))
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        enc = tokenizer([texts[j] for j in idx], padding=True, truncation=True,
                        max_length=MAX_SEQ_LENGTH, return_tensors='pt').to(device)
        if labels is not None:
            enc['labels'] = torch.tensor([labels[j] for j in idx], device=device)
        yield enc

@torch.no_grad()
def predict(texts, batch_size=256):
    # inference: returns 0/1 predictions and the chance of "positive"
    model.eval()
    probs = []
    for enc in make_batches(texts, batch_size=batch_size):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
            logits = model(**enc).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())   # softmax -> % (BERT course, Lesson 1)
    probs = torch.cat(probs).numpy()
    return (probs >= 0.5).astype(int), probs
""")
code(r"""
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))
print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.0f} million parameters (BERT Base is about 110M)")

for epoch in range(MAX_EPOCHS):
    model.train()
    start, total_loss, steps = time.time(), 0.0, 0
    for enc in make_batches(X_train, y_train, shuffle=True):
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
            loss = model(**enc).loss          # how wrong the model is
        scaler.scale(loss).backward()         # work out how to fix the weights
        scaler.step(optimizer)                # update the weights a little
        scaler.update()
        total_loss += loss.item(); steps += 1

    dev_pred, _ = predict(X_dev)
    print(f"Epoch {epoch + 1}/{MAX_EPOCHS} | loss {total_loss / steps:.4f} | "
          f"dev F1 {f1_score(y_dev, dev_pred, average='macro') * 100:.2f} | {time.time() - start:.0f} s")
""")

# ---------------- Step 5 ----------------
md(r"""
## Step 5: Test BERT on the "real exam" + try my own reviews
📘 *BERT course, Lesson 2:*
- **Precision** = when BERT says "negative", how often is it right?
- **Recall** = out of all the real negative reviews, how many did BERT find?
- **F1** = a mix of both (higher is better, max 1.00)

The score won't be 100%, because about 4% of the stars **don't match the words** on purpose (like real people!).
""")
code(r"""
test_pred, _ = predict(X_test)
print(classification_report(y_test, test_pred, target_names=['negative 😞', 'positive 😊'], digits=3))
""")
code(r"""
# Inference on NEW reviews (BERT course: model.classifytext(queries=[...]))
my_reviews = [
    "Wah the char kway teow got so much wok hei, damn shiok!",
    "Waited one hour for cold soggy prata. Jialat.",
    "The laksa gravy was super lemak, will come back again",
    "Overpriced and the auntie was rude",
    "The satay was half raw and the sauce was watery",   # try your own here!
]
pred, prob = predict(my_reviews)
for text, p, pr in zip(my_reviews, pred, prob):
    print(f"{'😊 positive' if p == 1 else '😞 negative'} ({pr * 100:5.1f}% positive) | {text}")
""")
md(r"""
### Use BERT on ALL the reviews (including the 3-star mixed ones)
Now BERT **reads every review**, and we use **its** answer (not the stars) to find the bad food spots.
""")
code(r"""
start = time.time()
all_pred, all_prob = predict(df['review'].tolist())
df['bert_positive_chance'] = all_prob.astype('float32')
df['bert_says_negative'] = all_pred == 0
print(f"BERT read {len(df):,} reviews in {time.time() - start:.1f} s")
print(f"BERT says {df['bert_says_negative'].mean() * 100:.1f}% are negative")

# What does BERT think of the mixed 3-star reviews?
df[df['is_mixed']][['dish', 'review', 'bert_positive_chance']].head(5)
""")

# ---------------- Step 6 ----------------
md(r"""
## Step 6: Find the "bad food hot spots" with DBSCAN 🗺️
📗 *GPU course, Lesson 6 (and the virus project, Week 1!).*

1. **lat / long → metres** (like the grid converter in Lesson 2, but a simple version for small Singapore)
2. Keep only the reviews **BERT says are negative**
3. **DBSCAN** finds places where **lots** of negative reviews are **packed close together**
   - `eps` = max distance (metres) to count as a neighbour
   - `min_samples` = how many negative reviews are needed nearby to make a hot spot
   - cluster **-1** = noise (not a hot spot)
""")
code(r"""
# lat/long -> metres from a reference point (1 degree of latitude is about 110,574 m)
LAT0, LONG0 = 1.35, 103.82
df['north_m'] = ((df['lat'] - LAT0) * 110574).astype('float32')
df['east_m'] = ((df['long'] - LONG0) * 111320 * np.cos(np.radians(LAT0))).astype('float32')

neg = df[df['bert_says_negative']].reset_index(drop=True)
print(f"{len(neg):,} negative reviews to cluster")

dbscan = DBSCAN(eps=100, min_samples=130)
X = neg[['east_m', 'north_m']].to_numpy(dtype='float32')
neg['cluster'] = np.asarray(dbscan.fit_predict(X))

print("\nReviews in each cluster (-1 = noise):")
print(neg['cluster'].value_counts().sort_index())
""")
code(r"""
# Describe each hot spot: where it is, how big it is, and which dish is the worst there (groupby, Lesson 3)
spots = neg[neg['cluster'] != -1]
hot = spots.groupby('cluster').agg(
    area=('area', lambda s: s.mode().iloc[0]),
    negative_reviews=('review', 'count'),
    lat=('lat', 'mean'),
    long=('long', 'mean'),
)
hot['worst_dish'] = spots.groupby('cluster')['dish'].agg(lambda s: s.mode().iloc[0])
hot.sort_values('negative_reviews', ascending=False)
""")
code(r"""
# Map: grey = all reviews, coloured = bad food hot spots, X = centre of each hot spot
fig, ax = plt.subplots(figsize=(10, 6))
sample = df.sample(min(5000, len(df)), random_state=1)
ax.scatter(sample['long'], sample['lat'], s=2, c='lightgrey', label='all reviews (sample)')
for c, g in spots.groupby('cluster'):
    ax.scatter(g['long'], g['lat'], s=4, label=f"hot spot {c}: {g['area'].mode().iloc[0]}")
ax.scatter(hot['long'], hot['lat'], marker='X', s=200, c='black')
ax.set_title('🍜 Bad food hot spots in Singapore (simulated data)')
ax.set_xlabel('longitude'); ax.set_ylabel('latitude'); ax.legend(markerscale=4, fontsize=8)
plt.show()
""")
code(r"""
# Try a different eps / min_samples (like eps 5000 vs 10000 in the course)
for eps, ms in [(100, 130), (100, 60), (300, 130)]:
    labels_ = np.asarray(DBSCAN(eps=eps, min_samples=ms).fit_predict(X))
    n = len(set(labels_.tolist())) - (1 if -1 in labels_ else 0)
    print(f"eps={eps:>3} m, min_samples={ms:>3} -> {n:>2} hot spots, {np.mean(labels_ == -1) * 100:4.1f}% noise")
""")

# ---------------- Step 7 ----------------
md(r"""
## Step 7 (optional): Send my results to the Streamlit app 🚀
**7a. BERT predictions → the app's Hot Spots map**
Run the next cell. It downloads `bert_predictions.csv`. Upload it to the GitHub repo, into the **`data/`** folder
(on GitHub: open `data/` → **Add file → Upload files**). The app will then let you choose "BERT predictions" on the map.
""")
code(r"""
df[['review_id', 'bert_positive_chance']].to_csv('bert_predictions.csv', index=False)
try:
    from google.colab import files
    files.download('bert_predictions.csv')
except ImportError:
    print("Saved bert_predictions.csv (not in Colab, so no download popup)")
""")
md(r"""
**7b. My fine-tuned BERT → the app's Review Checker**
1. Make a free account on [huggingface.co](https://huggingface.co), then go to **Settings → Access Tokens** and make a token with **write** access.
2. Run the next cell and paste the token when it asks.
3. In the Streamlit app settings, set the secret `MODEL_ID = "<your-hf-username>/makan-bert"` (see the README).
""")
code(r"""
from huggingface_hub import login
login()   # paste your Hugging Face WRITE token

model.push_to_hub("makan-bert")
tokenizer.push_to_hub("makan-bert")
print("✅ Uploaded! Use MODEL_ID = '<your-hf-username>/makan-bert' in the app")
""")

# ---------------- Summary ----------------
md(r"""
## ✅ What I found + what I learned

**Results (simulated data):**
- `groupby` shows which **dishes** get the highest / lowest average stars, and which **areas** have the most bad reviews.
- **BERT**, fine-tuned for only 2 epochs, reads a review and says positive / negative with a high **F1 score**, even for Singlish words it never saw (*shiok*, *jialat*), because the WordPiece tokenizer cuts them into subwords.
- **DBSCAN** on the reviews BERT marked negative finds the **"bad food hot spots"**. They match the areas with the highest bad-review rate in the `groupby` chart, so the two methods agree ✅

**Which course parts I used:**

| Step | Skill | Course |
|---|---|---|
| 0 | `%load_ext cudf.pandas`, run pandas on the GPU | 📗 L1 |
| 1 | `category`, `float32`, `memory_usage` | 📗 L2 |
| 2 | ETL: clean → Parquet | 📗 L4 |
| 3 | `groupby`, `pd.cut` binning, `pivot_table`, bar charts | 📗 L3 |
| 4 | BERT, WordPiece tokenizer, fine-tuning, mixed precision (FP16) | 📘 L1, L2, L4 |
| 5 | train / dev / test split, precision, recall, F1, inference | 📘 L2 + 📗 L7 |
| 6 | lat/long → metres, DBSCAN (`eps`, `min_samples`, -1 = noise) | 📗 L2, L6 |

**Live app:** the Streamlit app (`app.py`) shows the same steps as an interactive website: a review checker, a dashboard, and the hot spots map with `eps` / `min_samples` sliders.

**Ideas to make it better:** KNN to find the nearest "good" stall to each hot spot (📗 L7); XGBoost to predict stars from price + area + dish (📗 L8); put the BERT model on a Triton server (📘 L6).
""")

nb['cells'] = cells
nb['metadata'] = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"display_name": "Python 3", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, 'makan_review_analyser.ipynb')
print("written")
