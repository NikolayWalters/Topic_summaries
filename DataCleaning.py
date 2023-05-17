"""
Kaggle course notes on data cleaning
"""

# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]

# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)

# remove all the rows that contain a missing value
nfl_data.dropna()

# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()

# replace all NA's with 0
subset_nfl_data.fillna(0)

# replace all NA's the value that comes directly after it in the same column, 
# then replace all the remaining na's with 0
subset_nfl_data.fillna(method='bfill', axis=0).fillna(0)
# Example:
df = pd.DataFrame({'A': [1, 2, np.nan, 4, 5],
                   'B': [6, np.nan, 8, np.nan, 10],
                   'C': [11, 12, 13, 14, 15]})

#     A     B   C
#0  1.0   6.0  11
#1  2.0   NaN  12
#2  NaN   8.0  13
#3  4.0   NaN  14
#4  5.0  10.0  15

df['A'] = df['A'].fillna(method='bfill')

#     A     B   C
#0  1.0   6.0  11
#1  2.0   8.0  12
#2  4.0   8.0  13
#3  4.0  10.0  14
#4  5.0  10.0  15

# === Scaling and normalisation ===
# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])

# normalising in this context = changing to normal distribution
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# full example because boxcox only works with positive values
# get the index of all positive pledges (Box-Cox only takes positive values)
index_of_positive_pledges = kickstarters_2017.pledged > 0
# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.pledged.loc[index_of_positive_pledges]
# normalize the pledges (w/ Box-Cox)
normalized_pledges = pd.Series(stats.boxcox(positive_pledges)[0], 
                               name='pledged', index=positive_pledges.index)

# === Parsing dates ===

1/17/07 has the format "%m/%d/%y"
17-1-2007 has the format "%d-%m-%Y"

# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")

# try to automatically infere
landslides['date_parsed'] = pd.to_datetime(landslides['Date'], infer_datetime_format=True)

# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day

# === Character Encodings ===

# start with a string
before = "This is the euro symbol: €"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))

# We've lost the original underlying byte string! It's been 
# replaced with the underlying byte string for the unknown character :(

# >> This is the euro symbol: ?

# look at the first ten thousand bytes to guess the character encoding
with open("../input/kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:
    result = charset_normalizer.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)

before = sample_entry.decode("big5-tw")
new_entry = before.encode() # default utf-8

# === Inconsistent Data Entry ===

# get all the unique values in the 'Country' column
countries = professors['Country'].unique()

# sort them alphabetically and then take a closer look
countries.sort()

# convert to lower case
professors['Country'] = professors['Country'].str.lower()
# remove trailing white spaces
professors['Country'] = professors['Country'].str.strip()

# fuzzy matching
import fuzzywuzzy
from fuzzywuzzy import process

# get the top 10 closest matches to "south korea"
matches = fuzzywuzzy.process.extract("south korea", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio = 47):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match
    
    # let us know the function's done
    print("All done!")

# use the function we just wrote to replace close matches to "south korea" with "south korea"
replace_matches_in_column(df=professors, column='Country', string_to_match="south korea")