import datetime as dt
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

"""This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 
for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many 
customers of the company are wholesalers. 
https://www.kaggle.com/datasets/carrie1/ecommerce-data
"""

df_ = pd.read_csv("C:\\Users\erend\PycharmProjects\pythonProject\Exercise\data.csv", encoding="ISO-8859-1")
df = df_.copy()

# DATA PREPARATION
"""
Variable Description

InvoiceNo: Invoice number that consists 6 digits. If this code starts with letter 'c', it indicates a cancellation.
StockCode: Product code that consists 5 digits.
Description: Product name.
Quantity: The quantities of each product per transaction.
InvoiceDate: Represents the day and time when each transaction was generated.
UnitPrice: Product price per unit.
CustomerID: Customer number that consists 5 digits. Each customer has a unique customer ID.
Country: Name of the country where each customer resides."""

# top 10 observations
print(df.head(10))

# 8 variables, 541909 observations
print(df.shape)

# columns = 'InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'
print(df.columns)

# information about variables
print(df.info())

# descriptive statistics
print(df.describe().T)

# empty observations in variables
print(df.isnull().sum())

# Since customer-based work will be done, blank observations belonging to the customer are deleted.
df.dropna(subset=['CustomerID'], inplace=True)
print(df.isnull().sum())

# Purchases made by 4372 unique customers
print(df.shape)
print(df["CustomerID"].nunique())

# CUSTOMER ANALYSIS WITH RFM

df = df[(df['UnitPrice'] > 0)]
df = df[df['Quantity'] > 0]

# delete return orders
df = df[~df["InvoiceNo"].str.contains("C", na=False)]

# Converting the type of variables that express a date to date
date_columns = df.columns[df.columns.str.contains("Date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
print(df.info())

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
print(df.head())
print(df.describe().T)

# last shopping date
print(df["InvoiceDate"].max())

analysis_date = dt.datetime(2011, 12, 10)

df2 = df.groupby('CustomerID').agg(pd.Series.tolist)
df2 = df2.reset_index()

rfm = pd.DataFrame()
rfm["CustomerID"] = df2["CustomerID"]

rfm["recency"] = np.NaN
for i in rfm.index:
    rfm["recency"][i] = (analysis_date - max(df2["InvoiceDate"][i])).days

rfm["frequency"] = np.NaN
for i in rfm.index:
    rfm["frequency"][i] = len(df2["InvoiceNo"][i])

rfm["monetary"] = np.NaN
for i in rfm.index:
    rfm["monetary"][i] = sum(df2["TotalPrice"][i])

# rfm scoring
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

# Customer segmentation with rfm
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])


# PARETO ANALYSIS

def pareto_analysis(dataframe, id_, price_col, percentile=0.8):
    dataframe = dataframe.sort_values(price_col, ascending=False)
    dataframe.reset_index(inplace=True)
    dataframe['CumSum'] = dataframe[price_col].cumsum()
    # Threshold setting, 80% of total revenue
    threshold = dataframe[price_col].sum() * percentile
    target_df = dataframe[dataframe['CumSum'] <= threshold]
    print("> Total Revenue :", dataframe[price_col].sum())
    print(f"> %{100 * percentile} of total revenue", " from ", target_df.shape[0])
    print(f"> They, which make up %{100 * percentile} of the total revenue, constitute % "
          f"{round((target_df.shape[0] * 100 / dataframe.shape[0]), 2)} of all.")


# Pareto Anaysis for Customers
pareto_analysis(rfm, "CustomerID", 'monetary', percentile=0.75)
pareto_analysis(rfm, "CustomerID", 'monetary')


# CLTV

# Define outlier_thresholds and replace_with_thresholds functions needed to suppress outliers

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# if there are outliers, suppression
df.describe().T
columns = ["Quantity", "TotalPrice"]

for col in columns:
    replace_with_thresholds(df, col)

df.describe().T

# last shopping date
print(df["InvoiceDate"].max())
analysis_date = dt.datetime(2011, 12, 10)

# Create a new cltv dataframe with CustomerID, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values.
cltv_df = pd.DataFrame()
cltv_df["CustomerID"] = df2["CustomerID"]

cltv_df["recency_cltv_weekly"] = np.NaN
for i in cltv_df.index:
    cltv_df["recency_cltv_weekly"][i] = ((max(df2["InvoiceDate"][i]) - min(df2["InvoiceDate"][i])).days) / 7

cltv_df["T_weekly"] = np.NaN
for i in cltv_df.index:
    cltv_df["T_weekly"][i] = ((analysis_date - min(df2["InvoiceDate"][i])).days) / 7

cltv_df["frequency"] = np.NaN
for i in cltv_df.index:
    cltv_df["frequency"][i] = len(df2["InvoiceNo"][i])

cltv_df["monetary_cltv_avg"] = np.NaN
for i in cltv_df.index:
    cltv_df["monetary_cltv_avg"][i] = sum(df2["TotalPrice"][i]) / len(df2["InvoiceNo"][i])

cltv_df = cltv_df[(cltv_df['recency_cltv_weekly'] > 0)]
cltv_df = cltv_df[(cltv_df['T_weekly'] > 0)]
cltv_df = cltv_df[(cltv_df['frequency'] > 0)]
cltv_df = cltv_df[(cltv_df['monetary_cltv_avg'] > 0)]

cltv_df.head()

# BG / NBD Model
# Forecast of expected sales
# It is performed over frequency, weekly recency and weekly tenure. (may be monthly instead of weekly)
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])
# expected number of sales for 1 week for each customer
cltv_df["exp_sales_1_week"] = bgf.predict(1, cltv_df['frequency'], cltv_df['recency_cltv_weekly'],
                                          cltv_df['T_weekly'])
print(cltv_df.head())

# expected number of sales for 1 month for each customer
cltv_df["exp_sales_1_month"] = bgf.predict(4 * 1, cltv_df['frequency'], cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])
print(cltv_df.head())

# top 10 customer  expected to make the most sales in a week
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency_cltv_weekly'],
                                                        cltv_df['T_weekly']).sort_values(ascending=False).head()

# expected number of sales of the entire company in a month
bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly']).sum()
print(cltv_df.head(5))

# GAMMA-GAMMA Model
# Expected average profitability.
# It is done using # frequency and monetary_avg variables.
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary_cltv_avg']).head()

ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary_cltv_avg']).sort_values(
    ascending=False).head(5)

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])

cltv_df.sort_values("exp_average_value", ascending=False).head()

# Calculation of CLTV with BG-NBD and GG model
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()

cltv_final = pd.merge(cltv_df, cltv, left_index=True, right_index=True)
cltv_final.sort_values(by="clv", ascending=False).head()

del cltv_final["index"]

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 3, labels=["C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(10)

cltv_final.groupby("segment").agg(
    {"mean", "std", "median"})
