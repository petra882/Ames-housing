import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

"""
Средняя и медианная стоимости, гистограмма распределения цен
train_db = pl.read_csv("train.csv", infer_schema_length = 100000,
    null_values = ["NA"])
result = train_db.select(
    pl.col("SalePrice").mean().alias("avg"),
    pl.col("SalePrice").median().alias("median"),
)
plt.figure()
sns.histplot(train_db["SalePrice"], bins = 50)
plt.show()
"""

"""
Медиана по каждой категории
train_db = pl.read_csv("train.csv", infer_schema=10000, null_values=["NA"])
result = train_db.group_by(
    pl.col("OverallQual")
).agg(pl.col("SalePrice").mean().alias("avg"))
sns.barplot(result, x = "OverallQual", y = "avg")
plt.show()
"""

"""
Анализ выбросов по графику
id выбросов - [1299, 524, 692, 1183]
train_db = pl.read_csv("train.csv", infer_schema=10000, null_values=["NA"])
result = train_db.select(
    pl.col("SalePrice"),
    pl.col("GrLivArea")
)
sns.scatterplot(result, x = "GrLivArea", y = "SalePrice")
plt.show()
"""

"""
Средняя стоимость домов в зависимости от года ремонта (декады)
train_db = pl.read_csv("train.csv", infer_schema=10000, null_values=["NA"], try_parse_dates=True)
result = train_db.with_columns(
    ((pl.col("YearRemodAdd")//10)*10).alias("decade")
).group_by(pl.col("decade")).agg(pl.col("SalePrice").mean().alias("mean")).sort(pl.col("decade"))

sns.relplot(result, x = "decade", y = "mean", kind="line")
plt.show()
print(result)
"""

"""

train_db = pl.read_csv("train.csv", infer_schema=10000, null_values=["NA"], try_parse_dates=True)
result = train_db.with_columns(
    (pl.col("YrSold") - pl.col("YearBuilt")).alias("House_age")
)
result = result.select(
    pl.col("House_age"),
    pl.col("OverallQual"),
    pl.col("SalePrice")
).group_by(["OverallQual", "House_age"]).agg(pl.col("SalePrice").mean().alias("mean_price")).pivot(
        index="OverallQual",
        columns="House_age",
        values="mean_price"
    ).sort(pl.col("OverallQual"))

result = result.to_pandas().set_index("OverallQual")

plt.figure(figsize=(14, 6))
sns.heatmap(result, cmap="YlOrRd", annot=False)
plt.title("Средняя цена: Качество vs Возраст дома")
plt.xlabel("Возраст дома на момент продажи (лет)")
plt.ylabel("OverallQual")
plt.show()
"""


