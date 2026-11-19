# 🎬 Video demo script (about 3 minutes)

**Tip:** have the Streamlit app open in one browser tab and the Colab notebook in another.
Read slowly. You can change the words to your own.

---

## 1. Intro (20 s): show the app's title
> "Hi, I'm Jayden. This is **Makan Review Analyser**. It answers 2 questions:
> **which hawker foods do people love or hate**, and **where are the bad food hot spots in Singapore?**
> It combines 2 NVIDIA courses: **BERT** for understanding text, and **GPU data science** for working with big data fast.
> The data is simulated: 20,000 fake but realistic reviews."

## 2. Data cleaning (25 s): 🧹 Data Cleaning tab
> "The raw data is messy on purpose: duplicates, missing values, and dish names in different spellings.
> I clean it with an **ETL** process: Extract, Transform, Load.
> Look: there were **30 different dish spellings**, and after cleaning there are only **10**.
> I also use small data types like **category** and **float32** to save memory. That's important on a GPU."

## 3. Dashboard (30 s): 📊 Dashboard tab
> "Here I use **groupby**: split the data into groups, do the maths, then combine.
> Chicken rice has the highest average stars, and satay has the lowest.
> On the right, three areas have many more bad reviews than the others."
*(Pick one area in the sidebar filter to show that the charts update.)*

## 4. BERT Review Checker (45 s): 🤖 Review Checker tab
> "Now the AI part. **BERT** is a Transformer **encoder**. It reads the whole sentence and understands the context.
> I fine-tuned it to say if a review is positive or negative."
*(Click "😋 Shiok CKT" → Check review. Then type your own review.)*
> "It says positive, with this % chance. The % comes from **softmax**.
> If I open the tokens, you can see the **WordPiece tokenizer**: words BERT doesn't know, like *shiok*, are cut into small pieces with `##`."

## 5. Hot spots (40 s): 🗺️ Hot Spots tab
> "Finally, **DBSCAN** finds places where many negative reviews are packed close together, like finding infected clusters in the NVIDIA virus project.
> `eps` is the maximum distance, and `min_samples` is how many reviews are needed to make a hot spot."
*(Move the min_samples slider down, then back up.)*
> "With these settings, it finds 3 hot spots: Jurong West, Punggol and Woodlands. They match the dashboard chart, so the two methods agree."

## 6. Notebook + ending (20 s): show the Colab notebook
> "The BERT training happens in this Colab notebook on a free GPU. I use **cudf.pandas**, so pandas runs on the GPU, and I check BERT with a train / dev / test split and the **F1 score**.
> Thanks for watching!"

---

### 🎥 Recording tips
- **Windows:** press `Win + Alt + R` (Xbox Game Bar). **Mac:** press `Cmd + Shift + 5`.
- Zoom the browser to 110–125% so the text is easy to read.
- Do one practice run first. Open the app **before** recording, because the first BERT load takes about 30 s.
