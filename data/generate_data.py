"""
Makes the SIMULATED (fake but realistic) hawker food review dataset for this project.

- About 20,000 reviews of hawker dishes in 12 Singapore neighbourhoods.
- Stall names are made up. Locations are rough neighbourhood centres, not real businesses.
- The data is a bit "messy" on purpose (extra spaces, mixed CAPITALS, missing values,
  duplicates, prices as text like "$4.50"), so we can practise cleaning it (ETL).

Run:  python generate_data.py   ->  makes makan_reviews.csv
"""
import random
import csv
import math
from datetime import date, timedelta

random.seed(42)
N = 20000

# area: (lat, long, chance that a review is bad)
# A few areas get more bad reviews on purpose, so DBSCAN has "hot spots" to find.
AREAS = {
    "Toa Payoh":   (1.3343, 103.8563, 0.22),
    "Ang Mo Kio":  (1.3691, 103.8454, 0.25),
    "Bedok":       (1.3236, 103.9273, 0.24),
    "Tampines":    (1.3496, 103.9568, 0.23),
    "Jurong West": (1.3404, 103.7090, 0.55),   # hot spot
    "Woodlands":   (1.4382, 103.7890, 0.50),   # hot spot
    "Yishun":      (1.4304, 103.8354, 0.27),
    "Clementi":    (1.3162, 103.7649, 0.21),
    "Bukit Merah": (1.2819, 103.8239, 0.24),
    "Serangoon":   (1.3554, 103.8679, 0.20),
    "Punggol":     (1.3984, 103.9072, 0.48),   # hot spot
    "Chinatown":   (1.2836, 103.8443, 0.26),
}
# how busy each area is (more reviews)
AREA_WEIGHTS = [10, 9, 9, 9, 10, 8, 8, 7, 8, 7, 8, 9]

# dish: (normal price, words for good taste, words for bad taste)
DISHES = {
    "chicken rice":     (4.50, ["the chicken was so tender", "the rice was fragrant", "chilli sauce was shiok"],
                               ["the chicken was dry", "the rice was too oily", "chilli sauce had no kick"]),
    "laksa":            (5.50, ["the gravy was rich and lemak", "full of cockles", "spicy in a good way"],
                               ["the gravy was watery", "way too salty", "the noodles were soggy"]),
    "char kway teow":   (5.00, ["so much wok hei", "the noodles were smoky", "nice crispy lard bits"],
                               ["no wok hei at all", "too oily and wet", "the noodles were burnt"]),
    "nasi lemak":       (4.00, ["the coconut rice was fragrant", "the sambal was sedap", "the chicken wing was crispy"],
                               ["the rice was hard", "the sambal was too sweet", "the fried chicken was cold"]),
    "bak chor mee":     (5.00, ["the vinegar was perfect", "the noodles were springy", "lots of minced pork"],
                               ["too much vinegar", "the noodles were mushy", "very little pork"]),
    "hokkien mee":      (6.00, ["the prawn stock was so rich", "big fresh prawns", "perfect with the sambal"],
                               ["the prawns were not fresh", "the stock was bland", "it was too dry"]),
    "satay":            (8.00, ["the meat was smoky and juicy", "the peanut sauce was amazing", "nicely charred"],
                               ["the meat was tough", "the peanut sauce was watery", "it was half raw"]),
    "roti prata":       (2.50, ["so crispy and fluffy", "the curry was flavourful", "freshly made"],
                               ["it was soggy", "the curry was cold", "it tasted like old oil"]),
    "fishball noodles": (4.50, ["the fishballs were bouncy", "the soup was clear and tasty", "the chilli was good"],
                               ["the fishballs were rubbery", "the soup was too salty", "the noodles were clumpy"]),
    "carrot cake":      (4.00, ["crispy on the outside", "the black version was so good", "nice egg flavour"],
                               ["too greasy", "it was mushy", "it tasted burnt"]),
}
# some dishes are a bit more liked / disliked overall (adds a pattern for groupby)
DISH_BAD_SHIFT = {"satay": +0.08, "hokkien mee": -0.05, "chicken rice": -0.06, "roti prata": +0.05,
                  "carrot cake": +0.03, "laksa": -0.03}

GOOD_OPEN = ["Wow!", "So good!", "Shiok!", "Must try!", "Love it.", "Really nice.", "Steady lah.", ""]
BAD_OPEN = ["Jialat.", "So disappointed.", "Not worth it.", "Never again.", "Quite bad.", "Sian.", ""]
GOOD_EXTRA = ["The uncle was very friendly.", "Worth the queue.", "Good value for money.",
              "Will come back again.", "Big portion.", "Service was fast."]
BAD_EXTRA = ["Waited 40 minutes.", "Too expensive for this.", "The auntie was rude.",
             "Small portion.", "The table was dirty.", "Won't come back."]
MID = ["It was okay.", "Not bad but not great.", "Average only.", "Can eat lah.", "So-so."]

STALL_FIRST = ["Ah Huat", "Ah Seng", "Uncle Lim", "Auntie Mei", "Hup Lee", "Golden", "Lucky",
               "Ming Kee", "Ah Boy", "Heng Heng", "Kim's", "Ah Ma", "Lao Ban", "Hong Kee"]


def make_review(dish, kind):
    """kind: 'good', 'bad' or 'mid' -> returns review text"""
    _, good, bad = DISHES[dish]
    if kind == "good":
        parts = [random.choice(GOOD_OPEN), f"The {dish} was great,", random.choice(good) + "."]
        if random.random() < 0.6:
            parts.append(random.choice(GOOD_EXTRA))
    elif kind == "bad":
        parts = [random.choice(BAD_OPEN), f"The {dish} was bad,", random.choice(bad) + "."]
        if random.random() < 0.6:
            parts.append(random.choice(BAD_EXTRA))
    else:  # mixed review
        parts = [random.choice(MID), random.choice(good).capitalize() + ",",
                 "but " + random.choice(bad) + "."]
    text = " ".join(p for p in parts if p)
    # make some reviews shorter / less obvious, so BERT has to really read
    if random.random() < 0.25:
        text = text.replace(f"The {dish} was great,", "").replace(f"The {dish} was bad,", "")
    text = " ".join(text.split())
    return text[:1].upper() + text[1:]


def stars_for(kind):
    if kind == "good":
        return random.choice([4, 5, 5])
    if kind == "bad":
        return random.choice([1, 1, 2])
    return 3


def to_metres_offset(lat, long, east_m, north_m):
    """move a point by some metres (to spread reviews around the area centre)"""
    new_lat = lat + north_m / 110574
    new_long = long + east_m / (111320 * math.cos(math.radians(lat)))
    return new_lat, new_long


rows = []
area_names = list(AREAS.keys())
stalls = {a: {d: f"{random.choice(STALL_FIRST)} {d.title()}" for d in DISHES} for a in area_names}
start = date(2025, 1, 1)

for i in range(N):
    area = random.choices(area_names, weights=AREA_WEIGHTS)[0]
    lat0, long0, bad_rate = AREAS[area]
    dish = random.choice(list(DISHES))
    p_bad = min(max(bad_rate + DISH_BAD_SHIFT.get(dish, 0), 0.05), 0.9)

    r = random.random()
    if r < 0.12:
        kind = "mid"
    elif r < 0.12 + p_bad * 0.88:
        kind = "bad"
    else:
        kind = "good"

    review = make_review(dish, kind)
    stars = stars_for(kind)
    # about 4% "noisy" labels (people who give stars that don't match their words)
    if kind != "mid" and random.random() < 0.04:
        stars = 6 - stars

    # spread around the centre (most within about 300 m)
    lat, long = to_metres_offset(lat0, long0, random.gauss(0, 150), random.gauss(0, 150))

    price = DISHES[dish][0] + random.choice([-0.5, 0, 0, 0.5, 1.0])
    price_text = f"${price:.2f}"

    # --- make the data a bit messy on purpose ---
    dish_text = dish
    rr = random.random()
    if rr < 0.05:
        dish_text = dish.upper()
    elif rr < 0.10:
        dish_text = "  " + dish.title() + " "
    if random.random() < 0.015:
        review = ""            # missing review text
    stars_text = str(stars)
    if random.random() < 0.01:
        stars_text = ""        # missing stars

    d = start + timedelta(days=random.randint(0, 364))
    rows.append([i + 1, d.isoformat(), area, stalls[area][dish], dish_text,
                 price_text, stars_text, review, round(lat, 6), round(long, 6)])

# about 1% duplicated rows (copy-paste mistakes)
for _ in range(int(N * 0.01)):
    rows.append(list(random.choice(rows)))
random.shuffle(rows)

with open("makan_reviews.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["review_id", "date", "area", "stall", "dish", "price", "stars", "review", "lat", "long"])
    w.writerows(rows)

print(f"Saved {len(rows)} rows to makan_reviews.csv")
