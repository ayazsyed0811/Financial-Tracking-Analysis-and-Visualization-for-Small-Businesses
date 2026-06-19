import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# -------- MAIN FUNCTION -------- #

def process_file(file):

    # Load file
    df = pd.read_excel(file.name)

    # --- CLEANING ---
    df.rename(columns={"Customer ID": "CustomerID", "Price": "UnitPrice"}, inplace=True)
    df = df.dropna(subset=["CustomerID", "Description"])
    df = df[df["Quantity"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df = df.drop_duplicates()

    # --- CATEGORY ---
    def categorize(desc):
        desc = str(desc).lower()
        if "food" in desc or "cake" in desc:
            return "Food"
        elif "light" in desc or "lamp" in desc:
            return "Electronics"
        elif "plate" in desc or "bowl" in desc:
            return "Utensils"
        elif "bag" in desc or "box" in desc:
            return "Accessories"
        else:
            return "Others"

    df["Category"] = df["Description"].apply(categorize)

    # --- RFM ---
    ref_date = df["InvoiceDate"].max()

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (ref_date - x.max()).days,
        "Invoice": "nunique",
        "TotalPrice": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"].replace(0,1)
    rfm = rfm.reset_index()

    # --- MODEL ---
    X = rfm[["Recency", "Frequency", "AvgOrderValue"]]
    y = rfm["Monetary"]

    model = GradientBoostingRegressor()
    model.fit(X, y)

    rfm["Predicted_CLV"] = model.predict(X)

    # --- TOP CUSTOMERS ---
    top_customers = rfm.sort_values(by="Predicted_CLV", ascending=False).head(5)

    # --- MONTHLY REVENUE GRAPH ---
    df["Month"] = df["InvoiceDate"].dt.to_period("M")
    monthly_rev = df.groupby("Month")["TotalPrice"].sum()

    plt.figure()
    monthly_rev.plot(kind="bar")
    plt.title("Monthly Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    monthly_path = "monthly.png"
    plt.savefig(monthly_path)
    plt.close()

    # --- CATEGORY GRAPH ---
    cat_rev = df.groupby("Category")["TotalPrice"].sum()

    plt.figure()
    cat_rev.plot(kind="bar")
    plt.title("Category Revenue")
    plt.tight_layout()
    cat_path = "category.png"
    plt.savefig(cat_path)
    plt.close()

    # --- FORECAST (Food) ---
    monthly_qty = df.groupby(["Month", "Category"])["Quantity"].sum().reset_index()
    food = monthly_qty[monthly_qty["Category"] == "Food"].copy()

    if len(food) > 2:
        food["idx"] = range(len(food))
        Xf = food[["idx"]]
        yf = food["Quantity"]

        f_model = LinearRegression()
        f_model.fit(Xf, yf)

        pred = f_model.predict([[len(food)]])[0]
    else:
        pred = 0

    # --- SUMMARY TEXT ---
    summary = f"""
    📊 Dataset processed successfully!

    👥 Total Customers: {rfm.shape[0]}
    💰 Average CLV: {rfm["Predicted_CLV"].mean():.2f}
    🔝 Top Customer CLV: {rfm["Predicted_CLV"].max():.2f}

    📦 Next Month Food Demand: {int(pred)}
    """

    return summary, monthly_path, cat_path, top_customers


# -------- UI -------- #

interface = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="Upload Retail Dataset"),
    outputs=[
        gr.Textbox(label="Summary"),
        gr.Image(label="Monthly Revenue"),
        gr.Image(label="Category Revenue"),
        gr.Dataframe(label="Top Customers")
    ],
    title="Financial Analysis & CLV Prediction System",
    description="Upload your dataset to get financial insights, CLV prediction, and forecasting"
)

interface.launch(share=False)