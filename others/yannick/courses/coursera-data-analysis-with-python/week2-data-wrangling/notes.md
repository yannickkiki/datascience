# Pre-processing Data in Python
Data pre-processing (or data handling, or data wrangling is the process of converting)
the data from initial "raw" format to another format that can be used for further analysis
* Identify and handle missing values
* Data formatting
* Data normalization (centering/scaling)
* data binning (create bigger categories from set of numerical values)
* convert categorical values to numerical values to mae statistical modeling easier
* some possible operations: `df[column_name] = df[column_name] + 1`

# Dealing with missing values in Python
* can be represented with NaN, "?" or 0
## How to deal with missing data
* Check with data collection source
* Drop missing values: drop the data entry/drop the variable
* Replace the missing values: possibility to replace with average for number/ mode (most 
  frequent) for string or others
* Leave it as missing data: maybe useful
* df.dropna() to remove missing values (NaN), axis=0 remove all the row, axis=1 remove
 all the column, inplace=True to modify inplace, subset to select columns where to 
 check NaN
* df.replace(value_old, value_new)
* `df.[column_name].replace(np.nan, df[column_name].mean())`

# Data Formatting in Python
* bring data into a common standard in order to make meaningful comparison: 
  * Ex: N.Y/New York
  * use same units: g/L vs kg/L
* `df.rename(columns={old_name: new_name})`
* check and fix wrong data types
* objects, int64, float64
* `df[column_name] = df[column_name].astype('int')`

# Data Normalization in Python
* allow the data to have consistent values and make the same impact
  * age: 0-100, income: 10k-100k, not good, to normalize
  * set values to range `[0-1]`
* Normalization methods:
  * Simple feature scaling: x/x_max: [0,1]
  * Min max: x_x_min/x_max-x_min: [0,1]
  * z-score: x-avg/std: generally [-3,3] but can be +/-
* `df[column_name].max(),.mean(),.std()`

# Binning in Python
* price [5000-45000] can be categorized in low, medium, width
* can help model accuracy and check data distribution more easily

# Turning categorical variables into quantitative variables in Python
* most statistical can't take objects/string, just numbers
* one-hot encoding: fuel(gas, diesel) -> gas(0,1), diesel(0,1)
* pd.get_dummies()
