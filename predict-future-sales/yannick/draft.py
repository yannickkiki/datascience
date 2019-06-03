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
