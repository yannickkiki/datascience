# import pandas as pd
#
# filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
#
# headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration",
#            "num-of-doors", "body-style",
#            "drive-wheels", "engine-location", "wheel-base", "length", "width", "height",
#            "curb-weight", "engine-type",
#            "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
#            "compression-ratio", "horsepower",
#            "peak-rpm", "city-mpg", "highway-mpg", "price"]
#
# df = pd.read_csv(filename, names=headers)
#
# df.head()
#
# import numpy as np
#
# df.replace("?", np.nan, inplace=True)
# df.head(5)
#
# missing_data = df.isnull()
# missing_data.head(5)
#
# for column in missing_data.columns.values.tolist():
#     print(column)
#     print(missing_data[column].value_counts())
#     print("")
#
# avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
# print("Average of normalized-losses:", avg_norm_loss)
#
# df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
#
# avg_bore = df['bore'].astype('float').mean(axis=0)
# print("Average of bore:", avg_bore)
#
# df["bore"].replace(np.nan, avg_bore, inplace=True)
#
# avg_stroke = df["stroke"].astype("float").mean(axis=0)
# df["stroke"].replace(np.nan, avg_stroke, inplace=True)
#
# avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
# print("Average horsepower:", avg_horsepower)
#
# df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
#
# avg_peakrpm = df['peak-rpm'].astype('float').mean(axis=0)
# print("Average peak rpm:", avg_peakrpm)
#
# df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)
#
# df['num-of-doors'].value_counts()
#
# df['num-of-doors'].value_counts().idxmax()
#
# df["num-of-doors"].replace(np.nan, "four", inplace=True)
#
# df.dropna(subset=["price"], axis=0, inplace=True)
#
# df.reset_index(drop=True, inplace=True)
#
# df.head()
#
# df.dtypes
#
# df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
# df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
# df[["price"]] = df[["price"]].astype("float")
# df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
#
# df.dtypes
#
# df.head()
#
# df['city-L/100km'] = 235 / df["city-mpg"]
#
# df.head()
#
# df['highway-mpg'] = 235 / df["highway-mpg"]
# df.rename(columns={'highway-mpg': 'highway-L/100km'}, inplace=True)
# df.head()
#
# df['length'] = df['length'] / df['length'].max()
# df['width'] = df['width'] / df['width'].max()
#
# df['height'] = df['height'] / df['height'].max()
#
# df[["length", "width", "height"]].head()
#
# df["horsepower"] = df["horsepower"].astype(int, copy=True)
#
# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib as plt
#
# plt.pyplot.hist(df["horsepower"])
#
# plt.pyplot.xlabel("horsepower")
# plt.pyplot.ylabel("count")
# plt.pyplot.title("horsepower bins")
#
# bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
# bins
#
# group_names = ['Low', 'Medium', 'High']
#
# df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names,
#                                  include_lowest=True)
# df[['horsepower', 'horsepower-binned']].head(20)
#
# df["horsepower-binned"].value_counts()
#
# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib as plt
# from matplotlib import pyplot
#
# pyplot.bar(group_names, df["horsepower-binned"].value_counts())
#
# plt.pyplot.xlabel("horsepower")
# plt.pyplot.ylabel("count")
# plt.pyplot.title("horsepower bins")
#
# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib as plt
#
# plt.pyplot.hist(df["horsepower"], bins=3)
#
# plt.pyplot.xlabel("horsepower")
# plt.pyplot.ylabel("count")
# plt.pyplot.title("horsepower bins")
#
# df.columns
#
# dummy_variable_1 = pd.get_dummies(df["fuel-type"])
# dummy_variable_1.head()
#
# dummy_variable_1.rename(columns={'gas': 'fuel-type-gas', 'diesel': 'fuel-type-diesel'},
#                         inplace=True)
# dummy_variable_1.head()
#
# df = pd.concat([df, dummy_variable_1], axis=1)
#
# df.drop("fuel-type", axis=1, inplace=True)
#
# df.head()
#
# df["aspiration"].value_counts()
#
# df_dummies_aspiration = pd.get_dummies(df["aspiration"])
# df_dummies_aspiration.head()
#
# df_dummies_aspiration.rename(
#     columns={'std': 'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
#
# df_dummies_aspiration.head()
#
# df = pd.concat([df, df_dummies_aspiration], axis=1)
# df.head()
#
# df.drop("aspiration", axis=1, inplace=True)
#
# df.to_csv('clean_df.csv')
