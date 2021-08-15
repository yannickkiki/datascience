# Python packages for data science

* Scientific computing libraries:
  * Pandas(dataframes), Numpy(arrays), Scipy(integral, differential equations)

* Visualization libraries:
  * Matplotlib(plot, maps), Seaborn(time series, etc)

* Algorithmic libraries:
  * Scikit learn: machine learning classification regression, 
      built on numpy, scipy, matplotli
  * Statsmodels: help perform statistics tests


# Importing and exporting data in Python
* Formats: .csv, .json, .xlsx, .hdf
* File path, can be on computer on Internet
* df.head(n) to show first n lines, df.tail(n) to show last n lines
* pd.read_csv(header=None) when there is no header in the csv
* df.columns = column_list if needed
* pandas support read_{x}, to_{x} for x in [csv, excel, json, sql]

# Getting Started Analyzing Data in Python
* df.dtypes to check types
* df.describe() to get quick stats on df, option include='all'
* df.info() to view the column names and data types.

# Accessing Databases with Python
* Python DB-API
  * Connection objects: Dataase connection, manage transactions
  * cursor objects: database queries
  * connection methods: cursor(), commit(), rollback(), close()
