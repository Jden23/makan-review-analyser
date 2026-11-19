"""
🍜 Makan Review Analyser: Streamlit app
Run locally:  streamlit run app.py

Uses ideas from 2 NVIDIA DLI courses:
  📘 BERT course: a BERT-type model reads a review -> positive / negative
  📗 GPU data science course: ETL cleaning, groupby, charts, DBSCAN hot spots
(On Streamlit Cloud there is no GPU, so this uses normal pandas + scikit-learn.
 The same code runs on the GPU with `%load_ext cudf.pandas` and cuML, see the notebook.)
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN

st.set_page_config(page_title="Makan Review Analyser", page_icon="🍜", layout="wide")

DATA_PATH = "data/makan_reviews.csv"
BERT_PRED_PATH = "data/bert_predictions.csv"   # optional: made by the notebook (Step 7)


def get_model_id():
    # 1) Streamlit secret  2) environment variable  3) a public sentiment model
    try:
        if "MODEL_ID" in st.secrets:
            return st.secrets["MODEL_ID"]
    except Exception:
        pass
    return os.environ.get("MODEL_ID", "distilbert-base-uncased-finetuned-sst-2-english")


MODEL_ID = get_model_id()


# ---------------------------------------------------------------- data (ETL)
@st.cache_data
def load_raw():
    return pd.read_csv(DATA_PATH, dtype={"stars": "str", "price": "str"})


@st.cache_data
def clean(raw):
    """Same ETL steps as the notebook (GPU course, Lesson 4)."""
    steps = {"Raw rows": len(raw)}
    df = raw.drop_duplicates()
    steps["After removing duplicates"] = len(df)
    df = df[df["review"].notna() & (df["review"].str.strip() != "")]
    df = df[df["stars"].notna() & (df["stars"].str.strip() != "")]
    steps["After removing missing review / stars"] = len(df)
    df = df.copy()
    df["dish"] = df["dish"].str.strip().str.lower().astype("category")
    df["area"] = df["area"].astype("category")
    df["price"] = df["price"].str.replace("$", "", regex=False).astype("float32")
    df["stars"] = df["stars"].astype("float32").astype("int8")
    df["lat"] = df["lat"].astype("float32")
    df["long"] = df["long"].astype("float32")
    df["is_bad"] = (df["stars"] <= 2).astype("float32")
    # lat/long -> metres (simple version of the grid converter, GPU course Lesson 2)
    lat0, long0 = 1.35, 103.82
    df["north_m"] = ((df["lat"] - lat0) * 110574).astype("float32")
    df["east_m"] = ((df["long"] - long0) * 111320 * np.cos(np.radians(lat0))).astype("float32")
    return df.reset_index(drop=True), steps


@st.cache_data
def load_bert_predictions():
    if os.path.exists(BERT_PRED_PATH):
        return pd.read_csv(BERT_PRED_PATH)
    return None


# ---------------------------------------------------------------- model (BERT)
@st.cache_resource(show_spinner="Loading the BERT model (first time takes ~30 s)...")
def load_model(model_id):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    # which output number means "positive"?
    labels = {i: str(l).upper() for i, l in model.config.id2label.items()}
    pos_idx = next((i for i, l in labels.items() if "POS" in l), 1)
    return tok, model, pos_idx, torch


def classify(texts, tok, model, pos_idx, torch):
    with torch.no_grad():
        enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        probs = torch.softmax(model(**enc).logits, dim=-1)[:, pos_idx]   # softmax -> % positive
    return probs.numpy()


raw = load_raw()
df, etl_steps = clean(raw)
bert_pred = load_bert_predictions()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🍜 Makan Review Analyser")
st.sidebar.caption("BERT + GPU data science on Singapore hawker reviews")
areas = st.sidebar.multiselect("Areas", sorted(df["area"].cat.categories), placeholder="All areas")
dishes = st.sidebar.multiselect("Dishes", sorted(df["dish"].cat.categories), placeholder="All dishes")
view = df
if areas:
    view = view[view["area"].isin(areas)]
if dishes:
    view = view[view["dish"].isin(dishes)]
st.sidebar.divider()
st.sidebar.warning("⚠️ The data is **simulated** (fake but realistic). Stall names are made up.")
st.sidebar.caption(f"Model: `{MODEL_ID}`")

st.title("🍜 Makan Review Analyser")
st.write("**Which hawker foods do people love or hate, and where are the *bad food hot spots* in Singapore?**")

tab_ai, tab_dash, tab_map, tab_etl, tab_how = st.tabs(
    ["🤖 Review Checker", "📊 Dashboard", "🗺️ Hot Spots", "🧹 Data Cleaning", "ℹ️ How it works"])

# ---------------------------------------------------------------- tab 1: BERT
with tab_ai:
    st.subheader("🤖 Ask BERT: is this review positive or negative?")
    st.caption("📘 BERT course: a Transformer **encoder** reads the whole sentence, then a small classifier gives the % chance.")
    examples = {
        "😋 Shiok CKT": "Wah the char kway teow got so much wok hei, damn shiok!",
        "😡 Cold prata": "Waited one hour for cold soggy prata. Jialat.",
        "🍜 Lemak laksa": "The laksa gravy was super lemak, will come back again",
        "😐 Mixed": "The chicken was tender, but the rice was too oily.",
    }
    if "review_text" not in st.session_state:
        st.session_state.review_text = examples["😋 Shiok CKT"]
    cols = st.columns(len(examples))
    for col, (name, text) in zip(cols, examples.items()):
        if col.button(name, use_container_width=True):
            st.session_state.review_text = text
    text = st.text_area("Type a food review:", key="review_text", height=90)

    if st.button("Check review 🔍", type="primary") and text.strip():
        try:
            tok, model, pos_idx, torch = load_model(MODEL_ID)
            p = float(classify([text], tok, model, pos_idx, torch)[0])
            c1, c2 = st.columns([1, 2])
            if p >= 0.5:
                c1.success(f"### 😊 Positive\n{p * 100:.1f}% positive")
            else:
                c1.error(f"### 😞 Negative\n{(1 - p) * 100:.1f}% negative")
            c2.write("**Chance of positive:**")
            c2.progress(p)
            with st.expander("🔎 How BERT sees your text (WordPiece tokens)"):
                tokens = tok.tokenize(text)
                st.write(" ".join(f"`{t}`" for t in tokens))
                st.caption("Words BERT doesn't know are cut into pieces. `##` means the piece joins the one before it "
                           "(📘 BERT course, Lesson 4). Each token is turned into an ID number:")
                st.code(str(tok.convert_tokens_to_ids(tokens)))
        except Exception as e:
            st.error(f"Could not load the model `{MODEL_ID}`: {e}")

# ---------------------------------------------------------------- tab 2: dashboard
with tab_dash:
    st.subheader("📊 Dashboard")
    st.caption("📗 GPU course, Lesson 3: `groupby` = split → apply → combine.")
    if len(view) == 0:
        st.info("No reviews match the filters.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Reviews", f"{len(view):,}")
        m2.metric("Average stars", f"{view['stars'].mean():.2f} ⭐")
        m3.metric("Bad reviews (1-2 ⭐)", f"{view['is_bad'].mean() * 100:.1f}%")
        m4.metric("Average price", f"${view['price'].mean():.2f}")

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Average stars per dish**")
            dish_stars = view.groupby("dish", observed=True)["stars"].mean().sort_values(ascending=False)
            st.bar_chart(dish_stars, color="#f4a300", horizontal=True)
        with c2:
            st.write("**% bad reviews per area**")
            area_bad = (view.groupby("area", observed=True)["is_bad"].mean() * 100).sort_values(ascending=False)
            st.bar_chart(area_bad, color="#e4572e", horizontal=True)

        c3, c4 = st.columns(2)
        with c3:
            st.write("**Stars by price range** (binning with `pd.cut`)")
            buckets = pd.cut(view["price"], bins=[0, 3, 5, 7, 10], labels=["<= $3", "$3-5", "$5-7", "$7-10"])
            st.dataframe(view.groupby(buckets, observed=True)["stars"].agg(["count", "mean"]).round(2)
                         .rename(columns={"count": "reviews", "mean": "avg stars"}), use_container_width=True)
        with c4:
            st.write("**Number of reviews per star rating**")
            st.bar_chart(view["stars"].value_counts().sort_index(), color="#4c9be8")

        st.write("**Pivot table: average stars per area × dish**")
        pivot = view.pivot_table(index="area", columns="dish", values="stars", aggfunc="mean", observed=True).round(1)
        st.dataframe(pivot, use_container_width=True)

# ---------------------------------------------------------------- tab 3: DBSCAN
with tab_map:
    st.subheader("🗺️ Bad food hot spots (DBSCAN)")
    st.caption("📗 GPU course, Lesson 6 (like the virus project, Week 1): DBSCAN finds places where "
               "**many negative reviews are packed close together**.")
    sources = ["Stars (1-2 ⭐ = negative)"]
    if bert_pred is not None:
        sources.insert(0, "BERT predictions (from the notebook)")
    source = st.radio("Which reviews count as negative?", sources, horizontal=True)
    c1, c2 = st.columns(2)
    eps = c1.slider("eps: max distance to a neighbour (metres)", 50, 500, 100, step=25)
    min_samples = c2.slider("min_samples: negative reviews needed nearby", 10, 300, 110, step=10)

    if source.startswith("BERT"):
        merged = view.merge(bert_pred, on="review_id", how="inner")
        neg = merged[merged["bert_positive_chance"] < 0.5]
    else:
        neg = view[view["stars"] <= 2]
    neg = neg.reset_index(drop=True)

    if len(neg) < min_samples:
        st.info("Not enough negative reviews for these settings. Try a smaller min_samples or remove filters.")
    else:
        X = neg[["east_m", "north_m"]].to_numpy(dtype="float32")
        neg["cluster"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        n_spots = len(set(neg["cluster"])) - (1 if -1 in set(neg["cluster"]) else 0)
        m1, m2, m3 = st.columns(3)
        m1.metric("Negative reviews", f"{len(neg):,}")
        m2.metric("Hot spots found", n_spots)
        m3.metric("Noise (cluster -1)", f"{(neg['cluster'] == -1).mean() * 100:.0f}%")

        palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
                   "#f032e6", "#bfef45", "#469990", "#9a6324", "#800000", "#000075"]
        plot = neg[["lat", "long", "cluster"]].copy()
        plot[["lat", "long"]] = plot[["lat", "long"]].astype("float64")   # the map needs normal floats
        plot["color"] = plot["cluster"].map(lambda c: "#bbbbbb55" if c == -1 else palette[c % len(palette)])
        plot["size"] = np.where(plot["cluster"] == -1, 20, 40)
        st.map(plot, latitude="lat", longitude="long", color="color", size="size", zoom=10)
        st.caption("Coloured dots = hot spots. Grey dots = noise (negative reviews that are not packed together).")

        if n_spots:
            spots = neg[neg["cluster"] != -1]
            hot = spots.groupby("cluster").agg(
                area=("area", lambda s: s.mode().iloc[0]),
                negative_reviews=("review", "count"),
                worst_dish=("dish", lambda s: s.mode().iloc[0]),
                example=("review", "first"),
            ).sort_values("negative_reviews", ascending=False)
            st.write("**Hot spots** (biggest first)")
            st.dataframe(hot, use_container_width=True)

# ---------------------------------------------------------------- tab 4: ETL
with tab_etl:
    st.subheader("🧹 Data cleaning (ETL)")
    st.caption("📗 GPU course, Lesson 4: **E**xtract → **T**ransform → **L**oad. The raw data is messy on purpose.")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Rows after each cleaning step**")
        st.dataframe(pd.DataFrame({"rows": etl_steps}), use_container_width=True)
        st.write(f"**Dish spellings:** {raw['dish'].nunique()} messy → {df['dish'].nunique()} clean")
    with c2:
        st.write("**Examples of messy dish names**")
        messy = raw[raw["dish"] != raw["dish"].str.strip().str.lower()]["dish"].drop_duplicates().head(8)
        st.dataframe(pd.DataFrame({"raw": messy.map(repr), "clean": messy.str.strip().str.lower()}),
                     hide_index=True, use_container_width=True)
    st.write("**Raw data (first rows)**")
    st.dataframe(raw.head(10), hide_index=True, use_container_width=True)
    st.write("**Memory per column after cleaning** (small types like `category`, `int8`, `float32`, GPU course Lesson 2)")
    mem = (df.memory_usage(deep=True) / 1024).round(1).rename("KB")
    st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str), "KB": mem.drop("Index")}), use_container_width=True)

# ---------------------------------------------------------------- tab 5: how it works
with tab_how:
    st.subheader("ℹ️ How it works")
    st.markdown("""
```
CSV (20,200 messy reviews)
  → clean (ETL)                 📗 GPU course, Lesson 4
  → groupby + charts            📗 GPU course, Lesson 3
  → BERT reads each review      📘 BERT course, Lessons 2 + 4
  → DBSCAN finds hot spots      📗 GPU course, Lesson 6
```

| Part of the app | What it shows | Course |
|---|---|---|
| 🤖 Review Checker | a BERT-type model + WordPiece tokens + softmax % | 📘 L1, L2, L4 |
| 📊 Dashboard | `groupby`, binning with `pd.cut`, pivot table | 📗 L3 |
| 🗺️ Hot Spots | lat/long → metres, DBSCAN (`eps`, `min_samples`, -1 = noise) | 📗 L2, L6 |
| 🧹 Data Cleaning | ETL, small data types | 📗 L2, L4 |

**Training BERT** happens in the Colab notebook on a free GPU (`makan_review_analyser.ipynb`).
This app runs on a normal CPU server, so it uses normal pandas + scikit-learn. On a GPU, the same code
runs faster with `%load_ext cudf.pandas` and cuML.
""")
    if MODEL_ID == "distilbert-base-uncased-finetuned-sst-2-english":
        st.info("The Review Checker is using a **public** BERT-type sentiment model (DistilBERT, fine-tuned on movie "
                "reviews). To use **my own fine-tuned model** from the notebook, upload it to Hugging Face and set "
                "`MODEL_ID` in the app's secrets (see the README).")
