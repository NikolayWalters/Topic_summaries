"""
Notes on Intro to SQL from Kaggle
"""

from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client() # needed for retrieval
# Client objects hold projects and a connection to the BigQuery service

# Construct a reference to the "hacker_news" dataset
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")
# a project is a collection of datasets
# hacker_news is a dataset (a collection of tables)

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# List all the tables in the "hacker_news" dataset
tables = list(client.list_tables(dataset))

# Print names of all tables in the dataset (there are four!)
for table in tables:  
    print(table.table_id)
#>> comments
#>> full
#>> full_201510
#>> stories

# Construct a reference to the "full" table
table_ref = dataset_ref.table("full")

# API request - fetch the table
table = client.get_table(table_ref)

# Print information on all the columns in the "full" table in the "hacker_news" dataset
table.schema
#>> [SchemaField('title', 'STRING', 'NULLABLE', 'Story title', (), None),
#>>  SchemaField('url', 'STRING', 'NULLABLE', 'Story url', (), None), ...

# Preview the first five lines of the "full" table
client.list_rows(table, max_results=5).to_dataframe()

# Preview the first five entries in the "by" column of the "full" table
client.list_rows(table, selected_fields=table.schema[:1], max_results=5).to_dataframe()

# === Basic queries ===
# Query to select all the items from the "city" column where the "country" column is 'US'
query = """
        SELECT city
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """
# Create a "Client" object
client = bigquery.Client()
# Set up the query
query_job = client.query(query)
# API request - run the query, and return a pandas DataFrame
us_cities = query_job.to_dataframe()

# to estimate query size
# Query to get the score column from every row where the type column has value "job"
query = """
        SELECT score, title
        FROM `bigquery-public-data.hacker_news.full`
        WHERE type = "job" 
        """
# Create a QueryJobConfig object to estimate size of query without running it
dry_run_config = bigquery.QueryJobConfig(dry_run=True)
# API request - dry run query to estimate costs
dry_run_query_job = client.query(query, job_config=dry_run_config)
print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))
#>> This query will process 553320240 bytes.

# or by setting a custom limit
# Only run the query if it's less than 1 MB
ONE_MB = 1000*1000
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_MB)
# Set up the query (will only run if it's less than 1 MB)
safe_query_job = client.query(query, job_config=safe_config)
# API request - try to run the query, and return a pandas DataFrame
safe_query_job.to_dataframe()

# === Group by, having, count ===

# count - count no. of entries in a col; select count(col_name) from table

# GROUP BY takes the name of one or more columns, and treats all rows with the same value 
# in that column as a single group when you apply aggregate functions like COUNT().
# e.g. select col_name, count(col_name2) from table group by col_name
# returns all col_name entries with a count of col_name2 corresponding to that col_name entry

# HAVING is used in combination with GROUP BY to ignore groups that don't meet certain criteria.
# e.g. select col_name, count(col_name2) from table group by col_name having count(col_name2) > 1

# using AS to make col name more descriptive
query_improved = """
                 SELECT parent, COUNT(1) AS NumPosts
                 FROM `bigquery-public-data.hacker_news.comments`
                 GROUP BY parent
                 HAVING COUNT(1) > 10
                 """

# === Order by ===

# select col1, col2, col3 from table order by col1
# or descending:
# # select col1, col2, col3 from table order by col1 desc

# extract
# for date in format 2019-04-18
# select col, extract(day from DATE) as day from table
# extracts day only from date col
# can also extract eg week or dayofweek
# example
# Query to find out the number of accidents for each day of the week
query = """
        SELECT COUNT(consecutive_number) AS num_accidents, 
               EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        GROUP BY day_of_week
        ORDER BY num_accidents DESC
        """
# nb dayofweek = integer between 1 (Sunday) and 7 (Saturday)

# === as & with ===
# with as creates a CTE (a temporary table)

# Query to select the number of transactions per date, sorted by date
query_with_CTE = """ 
                 WITH time AS 
                 (
                     SELECT DATE(block_timestamp) AS trans_date
                     FROM `bigquery-public-data.crypto_bitcoin.transactions`
                 )
                 SELECT COUNT(1) AS transactions,
                        trans_date
                 FROM time
                 GROUP BY trans_date
                 ORDER BY trans_date
                 """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 10 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query_with_CTE, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
transactions_by_date = query_job.to_dataframe()

# Print the first five rows
transactions_by_date.head()

#plots
transactions_by_date.set_index('trans_date').plot()

# another example
# 3600 * SUM... line calculates average speed
speeds_query = """
               WITH RelevantRides AS
               (
                   SELECT EXTRACT(hour from trip_start_timestamp) as hour_of_day,
                   trip_miles,trip_seconds
                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                   WHERE trip_start_timestamp > '2016-01-01' 
                   AND trip_start_timestamp < '2016-04-01'
                   AND trip_seconds > 0 AND trip_miles > 0
               )
               SELECT hour_of_day, COUNT(1) AS num_trips,
               3600 * SUM(trip_miles) / SUM(trip_seconds) AS avg_mph 
               FROM RelevantRides
               GROUP BY hour_of_day
               ORDER BY hour_of_day
               """

# Set up the query
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
speeds_query_job = client.query(speeds_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
speeds_result = speeds_query_job.to_dataframe() # Your code here

# View results
print(speeds_result)

# === Joining data ===

# Query to determine the number of files per license, sorted by number of files
query = """
        SELECT L.license, COUNT(1) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` AS sf
        INNER JOIN `bigquery-public-data.github_repos.licenses` AS L 
            ON sf.repo_name = L.repo_name
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 10 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
file_count_by_license = query_job.to_dataframe()