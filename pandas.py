"""
summary of pandas functionality from Kaggle pandas course
"""

import pandas as pd

# creating simple df from dict
pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})

# creating a df with custom index
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])

# pandas series
pd.Series([1, 2, 3, 4, 5])

# series with a custom index
pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')

# initiating df from csv
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")

# dumping df as csv
csvDump = animals.to_csv('cows_and_goats.csv')

# calling a column by name
df.colName
df['colName']

# getting a specific row based on DF INDEX
df.iloc[0]

# iloc uses the Python stdlib indexing scheme, where the first element of the range is included and the 
# last one excluded. So 0:10 will select entries 0,...,9. loc, meanwhile, indexes inclusively. 
# So 0:10 will select entries 0,...,10

# get a specific row and col
df.iloc[row_indx, col_indx]
# or 
df.iloc[row_indx, 'col_name']

# conditional selection
df.loc[df.col_name == 'col_param where true']
# note below returns array of booleans where condition is true
df.col_name == 'col_param where true'

# multiple conditions
df.loc[(df.col_name == 'col_param where true') & (df.col_name >= 4)] # or use | pipe for or

# built in condition functions
df.loc[df.col_name.isin(['col_param1', 'col_param2'])]

# returns non-null entries
df.loc[df.col_name.notnull()]

# displays df column stats
df.col_name.describe()

# returns series with unique column values abd their counts
df.col_name.value_counts()

# transforming col values via a map
df.col_name.map(lambda p: p - df.col_name.mean()) # de-means data

# applying a custom function
def remean_points(row):
	"""
	Removes mean
	"""
    row.points = row.points - review_points_mean
    return row
df.apply(remean_points, axis='columns')

# just fyi pre-built
points_mean = df.col_name.mean()
df.col_name - points_mean

# equal length series can be combined
df.country + " - " + df.region
#example out: England - Berkshire

# === grouping and sorting ===
# groupby condition
df.groupby('col_name1').col_name2.min()
# shows min value of col_name2 parameter for each unique col_name1 entry

# can also be used with the apply function
df.groupby('col_name1').apply(lambda df: df.loc[df.col_name2.idxmax()])
#  selects max col2 parameter for each unique col_name1 param

# for each unique col_name1 param dispalys len, min and max of col_name2 params
df.groupby(['col_name1']).col_name2.agg([len, min, max])

# multi indexing, returns MultiIndex object!
df.groupby(['col_name1', 'col_name2']).col_name3.agg([len])
# First index grouped col_name1, second index col_name2 and value corresponding to number of col_name3 params for each

# converts from multiIndex back to df
mi.reset_index() 

# sort values by number of repeating entries in col_name
df.sort_values(by='col_name')
# by can take in multiple col_names as an array

# === Data types and missing entries
# col d type or all col d types, respectively
df.col_name.dtype
df.dtypes # returns series

# casting
df.col_name.astype('float64')

# return nan/null entries for a column
df[pd.isnull(df.col_name)]

# fill in nan/null with a constant value; n.b. returns series of col_name
df.col_name.fillna("Unknown")

# replaces re-like defined value
df.col_name.replace("value_to_be_replaced", "replacing_val")

# === Renaming and combining

# rename col name
df.rename(columns={'old_name': 'new_name'})

# rename index
df.rename(index={0: 'firstEntry', 1: 'secondEntry'})

# stacking (same columns)
df1 = pd.read_csv("../input")
df2 = pd.read_csv("../input")
pd.concat([df1, df2])

# join two tables on some common index
left = df1.set_index(['uniqueIndexID'])
right = df2.set_index(['uniqueIndexID'])
combinedDF = left.join(right)

# alternative to scikit test/train split
df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)

# to iterate through the df based on condition (instead of using boolean condition)
# define output df
filtered_df = pd.DataFrame()
# Iterate over each row in the DataFrame
for index, row in magneticDF.iterrows():
    if ((row['gmag']+3.75*row['BPRP']-13.6) > 0) and ((row['gmag']-(8/3)*row['BPRP']-11.7) > 0) and ((row['gmag']+3.75*row['BPRP']-15.7) < 0) and ((row['gmag'] -(8/3)*row['BPRP']-13) < 0):
        # Append the row to the filtered DataFrame
        filtered_df = filtered_df.append(row)
# Reset the index of the filtered DataFrame
filtered_df.reset_index(drop=True, inplace=True)