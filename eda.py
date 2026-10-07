import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/logistics_cleaned.csv")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
monthly = df.groupby(df["Order_Date"].dt.to_period("M"))["Late_Flag"].mean()*100

plt.figure(figsize=(8,5))
plt.plot(monthly.index.astype(str), monthly.values, marker="o")
plt.xticks(rotation=60)
plt.ylabel("Late delivery rate (%)")
plt.title("Monthly Late Delivery Rate")
plt.tight_layout()
plt.savefig("charts/monthly_late_rate.png", dpi=160)
