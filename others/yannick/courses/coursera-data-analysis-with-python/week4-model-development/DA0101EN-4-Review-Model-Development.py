# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
# df = pd.read_csv(path)
# df.head()
#
# from sklearn.linear_model import LinearRegression
#
# lm = LinearRegression()
# lm
#
# X = df[['highway-mpg']]
# Y = df['price']
#
# lm.fit(X, Y)
#
# Yhat = lm.predict(X)
# Yhat[0:5]
#
# lm.intercept_
#
# lm.coef_
#
# lm1 = LinearRegression()
#
# X = df[['engine-size']]
# Y = df['price']
# lm1.fit(X, Y)
#
# lm1.coef_
#
# lm1.intercept_
#
# Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
#
# lm.fit(Z, df['price'])
#
# lm.intercept_
#
# lm.coef_
#
# import seaborn as sns
#
# get_ipython().run_line_magic('matplotlib', 'inline')
#
# width = 12
# height = 10
# plt.figure(figsize=(width, height))
# sns.regplot(x="highway-mpg", y="price", data=df)
# plt.ylim(0, )
#
# plt.figure(figsize=(width, height))
# sns.regplot(x="peak-rpm", y="price", data=df)
# plt.ylim(0, )
#
# df[["peak-rpm", "highway-mpg", "price"]].corr()
#
# width = 12
# height = 10
# plt.figure(figsize=(width, height))
# sns.residplot(df['highway-mpg'], df['price'])
# plt.show()
#
# Y_hat = lm.predict(Z)
#
# plt.figure(figsize=(width, height))
#
# ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
# sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values", ax=ax1)
#
# plt.title('Actual vs Fitted Values for Price')
# plt.xlabel('Price (in dollars)')
# plt.ylabel('Proportion of Cars')
#
# plt.show()
# plt.close()
#
#
# def PlotPolly(model, independent_variable, dependent_variabble, Name):
#     x_new = np.linspace(15, 55, 100)
#     y_new = model(x_new)
#
#     plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
#     plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
#     ax = plt.gca()
#     ax.set_facecolor((0.898, 0.898, 0.898))
#     fig = plt.gcf()
#     plt.xlabel(Name)
#     plt.ylabel('Price of Cars')
#
#     plt.show()
#     plt.close()
#
#
# x = df['highway-mpg']
# y = df['price']
#
# f = np.polyfit(x, y, 3)
# p = np.poly1d(f)
# print(p)
#
# PlotPolly(p, x, y, 'highway-mpg')
#
# np.polyfit(x, y, 3)
#
# f1 = np.polyfit(x, y, 11)
# p1 = np.poly1d(f1)
# print(p1)
# PlotPolly(p1, x, y, 'Highway MPG')
#
# from sklearn.preprocessing import PolynomialFeatures
#
# pr = PolynomialFeatures(degree=2)
# pr
#
# Z_pr = pr.fit_transform(Z)
#
# Z.shape
#
# Z_pr.shape
#
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
#
# Input = [('scale', StandardScaler()),
#          ('polynomial', PolynomialFeatures(include_bias=False)),
#          ('model', LinearRegression())]
#
# pipe = Pipeline(Input)
# pipe
#
# Z = Z.astype(float)
# pipe.fit(Z, y)
#
# ypipe = pipe.predict(Z)
# ypipe[0:4]
#
# lm.fit(X, Y)
# print('The R-square is: ', lm.score(X, Y))
#
# Yhat = lm.predict(X)
# print('The output of the first four predicted value is: ', Yhat[0:4])
#
# from sklearn.metrics import mean_squared_error
#
# mse = mean_squared_error(df['price'], Yhat)
# print('The mean square error of price and predicted value is: ', mse)
#
# lm.fit(Z, df['price'])
# print('The R-square is: ', lm.score(Z, df['price']))
#
# Y_predict_multifit = lm.predict(Z)
#
# print('The mean square error of price and predicted value using multifit is: ',
#       mean_squared_error(df['price'], Y_predict_multifit))
#
# from sklearn.metrics import r2_score
#
# r_squared = r2_score(y, p(x))
# print('The R-square value is: ', r_squared)
#
# mean_squared_error(df['price'], p(x))
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# get_ipython().run_line_magic('matplotlib', 'inline')
#
# new_input = np.arange(1, 100, 1).reshape(-1, 1)
#
# lm.fit(X, Y)
# lm
#
# yhat = lm.predict(new_input)
# yhat[0:5]
#
# plt.plot(new_input, yhat)
# plt.show()
