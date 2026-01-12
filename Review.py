import pandas as pd

# ================================
# LOAD SOURCE CSV
# ================================
input_file = r"C:\Users\Eshwar\OneDrive\Desktop\stream\google_maps_reviews_with_dates.csv"
output_file = "google_maps_reviews_final.csv"

df = pd.read_csv(input_file)

# ================================
# NORMALIZE COLUMN NAMES
# ================================
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

print("Detected columns:", df.columns.tolist())

# ================================
# USE PARSED_DATE ONLY
# ================================
df["parsed_date"] = pd.to_datetime(df["parsed_date"], errors="coerce")

# ================================
# CREATE FINAL STRUCTURE
# ================================
final_df = pd.DataFrame({
    "date": df["parsed_date"],
    "review_text": df["review_text"],
    "rating": df["rating"],
    "sentiment": df["sentiment"],
    "topics": df["topics"],
    "user": df["user"]
})

# ================================
# CLEAN DATA
# ================================
final_df = final_df.dropna(subset=["date", "review_text"])

# ================================
# SAVE FINAL CSV
# ================================
final_df.to_csv(output_file, index=False)

print("✅ google_maps_reviews_final.csv created successfully")
print("Rows:", len(final_df))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Google Reviews Analytics",
    layout="wide"
)

st.title("📊 Google Reviews Analytics Dashboard")

# ================================
# LOAD FINAL DATA
# ================================
df = pd.read_csv("google_maps_reviews_final.csv")

# Normalize column names (extra safety)
df.columns = df.columns.str.strip().str.lower()

# Ensure date column
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# ================================
# SIDEBAR FILTER
# ================================
st.sidebar.header("🔍 Filters")
date_selection = st.sidebar.date_input(
    "Select date range",
    value=(df["date"].min(), df["date"].max()),
    key="date_range"
)

# Handle single date vs range safely
if isinstance(date_selection, tuple):
    start_date, end_date = date_selection
else:
    start_date = end_date = date_selection

df = df[
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
]


# ================================
# SENTIMENT TREND CHART
# ================================
st.subheader("📈 Sentiment Trend (Weekly)")

trend_df = (
    df.groupby([pd.Grouper(key="date", freq="W"), "sentiment"])
      .size()
      .reset_index(name="count")
)

fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(
    data=trend_df,
    x="date",
    y="count",
    hue="sentiment",
    marker="o",
    ax=ax1
)
ax1.set_xlabel("Week")
ax1.set_ylabel("Number of Reviews")
st.pyplot(fig1)

# ================================
# TOP THEMES
# ================================
st.subheader("🔥 Top Recurring Themes")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ Positive Themes")
    pos_topics = (
        df[df["sentiment"] == "positive"]["topics"]
        .str.split(",")
        .explode()
        .value_counts()
        .head(5)
    )
    st.dataframe(pos_topics)

with col2:
    st.markdown("### ❌ Negative / Neutral Themes")
    neg_topics = (
        df[df["sentiment"].isin(["negative", "neutral"])]["topics"]
        .str.split(",")
        .explode()
        .value_counts()
        .head(5)
    )
    st.dataframe(neg_topics)

# ================================
# AUTO-GENERATED BUSINESS TIPS
# ================================
st.subheader("🤖 AI-Generated Business Improvement Tips")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

analysis_text = ""
for topic in neg_topics.index:
    sample_reviews = (
        df[df["topics"].str.contains(topic, na=False)]
        ["review_text"]
        .head(3)
        .tolist()
    )

    analysis_text += f"""
Topic: {topic}
Customer complaints:
- {'; '.join(sample_reviews)}
"""

prompt = f"""
You are a business consultant.

Based on the customer complaints below,
suggest clear, practical business improvement actions.

{analysis_text}

Respond in concise bullet points.
"""

if st.button("Generate Improvement Suggestions"):
    with st.spinner("Analyzing reviews..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        st.success("Insights Generated")
        st.write(response.choices[0].message.content)
