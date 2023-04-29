"""
Kagle Advanced SQL course notes
"""

# === Join and union ===
# inner join
name1	val1
name2	val2

# left join
name1	val1
name2	NULL

# right join
name1	val1
NULL	val2

# full join
name1	val1
NULL	val2
name3	NULL

# union = vertially concat columns from two tables
"""
SELECT Age FROM pets UNION ALL SELECT Age FROM owners
"""
# out
# Age
# 20
# 1 ...
# 20
# UNION ALL includes duplicate values
# Can use UNION DISTINCT instead

# === Analytic functions ===
# All analytic functions have an OVER clause, which defines the sets of rows used in each calculation
# three (optional) parts:
PARTITION BY # divides the rows of the table into different groups
ORDER BY
ROWS BETWEEN 1 PRECEDING AND CURRENT ROW # window frame

# Example window frame clauses
ROWS BETWEEN 1 PRECEDING AND CURRENT ROW # the previous row and the current row.
ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING # the 3 previous rows, the current row, and the following row.
ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING # all rows in the partition.

# Navigation functions
FIRST_VALUE() or LAST_VALUE() # Returns the first (or last) value in the input
LEAD() or LAG() # Returns the value on a subsequent (or preceding) row

# Analytic numbering functions
ROW_NUMBER() # Returns the order in which rows appear in the input (starting with 1)
RANK() # All rows with the same value in the ordering column receive the same rank value, 
# where the next row receives a rank value which increments by the number of rows with the previous rank value.

# Example
# Query to count the (cumulative) number of trips per day
num_trips_query = """
                  WITH trips_by_day AS
                  (
                  SELECT DATE(start_date) AS trip_date, # count up number of trips for a specific date
                      COUNT(*) as num_trips
                  FROM `bigquery-public-data.san_francisco.bikeshare_trips`
                  WHERE EXTRACT(YEAR FROM start_date) = 2015
                  GROUP BY trip_date
                  )
                  SELECT *,
                      SUM(num_trips) 	# and sum them up for a total
                          OVER (
                               ORDER BY trip_date
                               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                               ) AS cumulative_trips
                      FROM trips_by_day
                  """

# Run the query, and return a pandas DataFrame
num_trips_result = client.query(num_trips_query).result().to_dataframe()
num_trips_result.head()

# Example 2
# Query to track beginning and ending stations on October 25, 2015, for each bike
start_end_query = """
                  SELECT bike_number,
                      TIME(start_date) AS trip_time,
                      FIRST_VALUE(start_station_id)
                          OVER (
                               PARTITION BY bike_number
                               ORDER BY start_date
                               ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                               ) AS first_station_id,
                      LAST_VALUE(end_station_id)
                          OVER (
                               PARTITION BY bike_number
                               ORDER BY start_date
                               ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                               ) AS last_station_id,
                      start_station_id,
                      end_station_id
                  FROM `bigquery-public-data.san_francisco.bikeshare_trips`
                  WHERE DATE(start_date) = '2015-10-25' 
                  """


# Example 3
# Average number of trips for a given day
avg_num_trips_query = """
                      WITH trips_by_day AS
                      (
                      SELECT DATE(trip_start_timestamp) AS trip_date,
                          COUNT(*) as num_trips
                      FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                      WHERE trip_start_timestamp > '2016-01-01' AND trip_start_timestamp < '2016-04-01'
                      GROUP BY trip_date
                      )
                      SELECT trip_date,
                          AVG(num_trips)
                          OVER (
                               ORDER by trip_date
                               ROWS between 3 PRECEDING and following
                               ) AS avg_num_trips
                      FROM trips_by_day
                      """

# === Nested and repeated data ===
# Nested columns have type STRUCT (or type RECORD)
# REPEATED for repeated data (more than one value for each row is permitted)

# Can use UNNEST() to flatten repeated data

# Example
max_commits_query = """ SELECT committer.name AS committer_name,
                    COUNT(*) as num_commits
                    FROM `bigquery-public-data.github_repos.sample_commits`
                    WHERE committer.date >= '2016-01-01' AND committer.date < '2017-01-01'
                    GROUP BY committer_name
                    ORDER BY num_commits DESC
            """

# Example 2
pop_lang_query = """SELECT language.name AS language_name,
                    COUNT(*) AS num_repos
                    FROM `bigquery-public-data.github_repos.languages`,
                    UNNEST(language) AS language
                    GROUP BY language_name
                    ORDER BY num_repos DESC
                 """

# === On Efficiency ===
# 1) SELECT only what you need
# 2) read less
# 3) Avoid N:N JOINs