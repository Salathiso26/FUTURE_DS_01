"""
Task 1: Business Sales Performance Analytics
Future Interns - Data Science & Analytics Track (FUTURE_DS_01)

Dataset: Sample Superstore (Kaggle) - retail order-line transactions, 2014-2017
File used: Sample - Superstore.csv (9,994 records)

Analyzes business sales data to identify revenue trends, top-selling products,
high-value categories/sub-categories, and regional performance, plus what is
driving margin loss (discounting).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

NAVY, GOLD, TEAL, RED = "#1f2a44", "#c9a227", "#2b7a78", "#a34e3c"
money = mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K")


# 1. LOAD & CLEAN

df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")
df.columns = [c.strip() for c in df.columns]

df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")
df["Year"] = df["Order Date"].dt.year
df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)
df = df.drop_duplicates()

print("Rows:", len(df))
print("Missing values by column:")
print(df.isnull().sum()[df.isnull().sum() > 0])


# 2. OVERALL KPIs

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
overall_margin = total_profit / total_sales * 100
n_orders = df["Order ID"].nunique()
n_customers = df["Customer ID"].nunique()
avg_order_value = df.groupby("Order ID")["Sales"].sum().mean()

print("\n=== OVERALL KPIs ===")
print(f"Total Sales:     ${total_sales:,.2f}")
print(f"Total Profit:    ${total_profit:,.2f}")
print(f"Overall Margin:  {overall_margin:.2f}%")
print(f"Orders:          {n_orders:,}")
print(f"Customers:       {n_customers:,}")
print(f"Avg Order Value: ${avg_order_value:,.2f}")


# 3. REVENUE TRENDS (monthly + yearly)


monthly = df.groupby("YearMonth")["Sales"].sum().reset_index()
yearly = df.groupby("Year").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()

print("\n=== YEARLY REVENUE & PROFIT ===")
print(yearly)

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(monthly["YearMonth"], monthly["Sales"], color=NAVY, linewidth=2)
ax.fill_between(range(len(monthly)), monthly["Sales"], color=NAVY, alpha=0.08)
ax.set_title("Monthly Revenue Trend (2014-2017)")
ax.yaxis.set_major_formatter(money)
xt = list(range(0, len(monthly), 4))
ax.set_xticks(xt)
ax.set_xticklabels(monthly["YearMonth"].iloc[xt], rotation=45, ha="right", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(8, 4.2))
x = yearly["Year"].astype(str)
w = 0.35
ax.bar([i - w / 2 for i in range(len(x))], yearly["Sales"], width=w, label="Sales", color=NAVY)
ax.bar([i + w / 2 for i in range(len(x))], yearly["Profit"], width=w, label="Profit", color=GOLD)
ax.set_xticks(range(len(x)))
ax.set_xticklabels(x)
ax.set_title("Yearly Sales vs Profit")
ax.yaxis.set_major_formatter(money)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()


# 4. TOP-SELLING PRODUCTS

top_products = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10)
print("\n=== TOP 10 PRODUCTS BY REVENUE ===")
print(top_products)

labels = [l if len(l) <= 42 else l[:39] + "..." for l in top_products.index[::-1]]
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.barh(labels, top_products.values[::-1], color=NAVY)
ax.set_title("Top 10 Products by Revenue")
ax.xaxis.set_major_formatter(money)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()


# 5. CATEGORY & SUB-CATEGORY PERFORMANCE

cat = df.groupby("Category").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
cat["Margin%"] = cat["Profit"] / cat["Sales"] * 100
cat = cat.sort_values("Sales", ascending=False)
print("\n=== CATEGORY PERFORMANCE ===")
print(cat)

subcat = df.groupby("Sub-Category").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
subcat["Margin%"] = subcat["Profit"] / subcat["Sales"] * 100
subcat = subcat.sort_values("Profit")
print("\n=== SUB-CATEGORY PROFIT (worst first) ===")
print(subcat)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].bar(cat.index, cat["Sales"], color=[NAVY, GOLD, TEAL])
axes[0].set_title("Sales by Category")
axes[0].yaxis.set_major_formatter(money)
axes[0].spines[["top", "right"]].set_visible(False)
axes[0].grid(axis="y", linestyle="--", alpha=0.3)
axes[1].bar(cat.index, cat["Profit"], color=[GOLD if v > 0 else RED for v in cat["Profit"]])
axes[1].set_title("Profit by Category")
axes[1].yaxis.set_major_formatter(money)
axes[1].spines[["top", "right"]].set_visible(False)
axes[1].grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

colors_ = [RED if v < 0 else TEAL for v in subcat["Profit"]]
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.barh(subcat.index, subcat["Profit"], color=colors_)
ax.set_title("Profit by Sub-Category (red = loss-making)")
ax.xaxis.set_major_formatter(money)
ax.axvline(0, color="#333333", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()


# 6. REGIONAL PERFORMANCE

region = df.groupby("Region").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
region["Margin%"] = region["Profit"] / region["Sales"] * 100
region = region.sort_values("Sales", ascending=False)
print("\n=== REGIONAL PERFORMANCE ===")
print(region)

state = df.groupby("State").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).sort_values("Sales", ascending=False)
print("\n=== TOP 10 STATES BY REVENUE ===")
print(state.head(10))
print("\n=== BOTTOM 5 STATES BY PROFIT ===")
print(state.sort_values("Profit").head(5))

fig, ax = plt.subplots(figsize=(8, 4.2))
x = range(len(region))
w = 0.35
ax.bar([i - w / 2 for i in x], region["Sales"], width=w, label="Sales", color=NAVY)
ax.bar([i + w / 2 for i in x], region["Profit"], width=w, label="Profit", color=GOLD)
ax.set_xticks(list(x))
ax.set_xticklabels(region.index)
ax.set_title("Regional Sales vs Profit")
ax.yaxis.set_major_formatter(money)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

top_states = state.head(10)["Sales"]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.barh(top_states.index[::-1], top_states.values[::-1], color=TEAL)
ax.set_title("Top 10 States by Revenue")
ax.xaxis.set_major_formatter(money)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()


# 7. DISCOUNT IMPACT ON PROFIT

bins = pd.cut(df["Discount"], bins=[-0.01, 0, 0.2, 0.4, 1])
discount_bands = df.groupby(bins, observed=True).agg(
    Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "count")
)
print("\n=== DISCOUNT BAND IMPACT ===")
print(discount_bands)

fig, ax = plt.subplots(figsize=(8, 4.4))
sample = df.sample(min(2000, len(df)), random_state=1)
ax.scatter(sample["Discount"], sample["Profit"], alpha=0.35, s=14, color=NAVY)
ax.axhline(0, color=RED, linewidth=1, linestyle="--")
ax.set_title("Discount vs Profit (order-line level)")
ax.set_xlabel("Discount")
ax.set_ylabel("Profit (USD)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()


# 8. CUSTOMER SEGMENT PERFORMANCE

segment = df.groupby("Segment").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
segment["Margin%"] = segment["Profit"] / segment["Sales"] * 100
segment = segment.sort_values("Sales", ascending=False)
print("\n=== SEGMENT PERFORMANCE ===")
print(segment)


# 9. SUMMARY FOR REPORT

summary = {
    "total_sales": round(total_sales, 2),
    "total_profit": round(total_profit, 2),
    "overall_margin_pct": round(overall_margin, 2),
    "orders": int(n_orders),
    "customers": int(n_customers),
    "avg_order_value": round(avg_order_value, 2),
    "best_category": cat["Sales"].idxmax(),
    "worst_margin_category": cat["Margin%"].idxmin(),
    "worst_subcategory": subcat.index[0],
    "worst_subcategory_profit": round(subcat["Profit"].iloc[0], 2),
    "best_region": region["Sales"].idxmax(),
    "worst_margin_region": region["Margin%"].idxmin(),
    "top_state": state.index[0],
    "high_discount_profit_loss": round(discount_bands["Profit"].iloc[2:].sum(), 2),
}

print("\n=== SUMMARY FOR REPORT ===")
for key, value in summary.items():
    print(f"{key}: {value}")
