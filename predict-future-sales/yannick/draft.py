from numpy import in1d

in1d(val_df["shop_id"].unique(), df["shop_id"].unique(),
     assume_unique=True)
in1d(val_df["item_id"].unique(), df["item_id"].unique(),
     assume_unique=True)
a = in1d(val_df["item_category_id"].unique(), df["item_category_id"].unique(),
     assume_unique=True)

# tous les shops de val_df sont dans df
# pareil pour les categories d'items
# mais tous les item de val_df ne se retrouvent pas dans df

#just for make submission
result = 37.2
r = val_df[["ID"]]
from pandas import Series
r["item_cnt_month"] = Series([result]*len(val_df))
r.set_index("ID", inplace = True)
r.to_csv("submission02")


#----------------------------------------#
submission_df = read_csv("submission01.csv")
submission_df["item_cnt_month"].fillna(0.3, inplace = True)
submission_df.set_index("ID", inplace = True)
submission_df.to_csv("submission01.csv")
