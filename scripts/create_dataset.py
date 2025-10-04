import pandas as pd, os

os.makedirs("data", exist_ok=True)

data = [
    ("I love flying with this airline!", "positive"),
    ("Amazing service and friendly staff!", "positive"),
    ("Super smooth check-in process!", "positive"),
    ("The new update works perfectly, great job!", "positive"),
    ("Had an excellent experience today!", "positive"),
    ("Fast delivery and awesome packaging!", "positive"),
    ("I’m so happy with my new phone!", "positive"),
    ("Totally loved the new design!", "positive"),
    ("Everything was handled professionally.", "positive"),
    ("This app is getting better every update!", "positive"),

    ("The service was okay, nothing special.", "neutral"),
    ("I used the app today, it was fine.", "neutral"),
    ("The product arrived on time.", "neutral"),
    ("It works as expected.", "neutral"),
    ("No complaints, just a normal experience.", "neutral"),
    ("The new feature is alright.", "neutral"),
    ("It’s an average performance overall.", "neutral"),
    ("Neither good nor bad, just fine.", "neutral"),
    ("Nothing remarkable, just standard.", "neutral"),
    ("It’s okay, could be improved.", "neutral"),

    ("This is the worst service ever.", "negative"),
    ("The staff were rude and unhelpful.", "negative"),
    ("The app keeps crashing, very frustrating.", "negative"),
    ("My package arrived late and damaged.", "negative"),
    ("Customer support didn’t respond at all.", "negative"),
    ("Horrible experience, will never come back.", "negative"),
    ("This company doesn’t care about customers.", "negative"),
    ("Terrible product quality.", "negative"),
    ("The flight was delayed and chaotic.", "negative"),
    ("Completely disappointed with the outcome.", "negative"),
] * 5  # repeat to reach 150 rows

df = pd.DataFrame(data, columns=["text", "sentiment"])
df.to_csv("data/twitter.csv", index=False)
print(f"✅ New dataset created with {len(df)} samples at data/twitter.csv")
