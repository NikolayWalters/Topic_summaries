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
