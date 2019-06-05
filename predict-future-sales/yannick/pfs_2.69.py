from pandas import read_csv
df = read_csv("datasets/sales_train_v2.csv")
items_df = read_csv("datasets/items.csv")
items_df.drop(columns = ["item_name"], inplace = True)
items_df.set_index("item_id", inplace = True)
df = df.join(items_df, on="item_id")
df["n_products_sold"] = df.apply(lambda row: row["item_cnt_day"] if row["item_cnt_day"] > 0 else 0, axis = 1)
df.to_csv("datasets/data_train_reviewed.csv")

df = read_csv("datasets/data_train_reviewed.csv")

val_df = read_csv("prediction/test.csv")
val_df = val_df.join(items_df, on="item_id")

number_of_item_per_categories = items_df["item_category_id"].value_counts()

result_df = val_df[["ID"]]
from pandas import Series

results = list()
for idx, val_row in val_df.iterrows():
    val_row_df = df.loc[(df["shop_id"] == val_row["shop_id"]) & (df["item_id"] == val_row["item_id"])]
    m = 1
    if val_row_df.shape[0] == 0:
        val_row_df = df.loc[(df["shop_id"] == val_row["shop_id"]) & (df["item_category_id"] == val_row["item_category_id"])]
        m = number_of_item_per_categories.loc[val_row["item_category_id"]]
    selling_start = val_row_df["date_block_num"].min() if val_row_df.shape[0] != 0 else 0.3
    results.append(round(val_row_df["n_products_sold"].sum()/((34 - selling_start)*m)))

result_df["item_cnt_month"] = Series(results)
result_df.set_index("ID", inplace = True)
result_df.to_csv("submission01.csv")

