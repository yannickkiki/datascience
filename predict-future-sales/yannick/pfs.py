from pandas import read_csv
df = read_csv("datasets/sales_train_v2.csv")
items_df = read_csv("datasets/items.csv")
items_df.drop(columns = ["item_name"], inplace = True)
items_df.set_index("item_id", inplace = True)
df = df.join(items_df, on="item_id")

val_df = read_csv("prediction/test.csv")
val_df = val_df.join(items_df, on="item_id")
del items_df

from numpy import in1d
in1d(val_df["item_category_id"].unique(), df["item_category_id"].unique(),
     assume_unique=True)

#just for make submission
result = 37.2
r = val_df[["ID"]]
from pandas import Series
r["item_cnt_month"] = Series([result]*len(val_df))
r.set_index("ID", inplace = True)
r.to_csv("submission00")
