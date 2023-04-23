"""
Some quick notes on seaborn library from kaggle course on data visualisation
"""

import seaborn as sns
data = pd.df

# simple line plot that plots from a df
sns.lineplot(data=data) # i think when loading in DF index col was specified
# so i assume index defaults to x

# pyplot figsize still changes fig size along with other pyplot commandds
plt.figure(figsize=(14,6))

# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")

# Add label for horizontal axis
plt.xlabel("Date")

# bar chart + separating df into x and y explicitly
sns.barplot(x=flight_data.index, y=flight_data['NK'])

# heatmap
# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=flight_data, annot=True) # annotate ads numerical values to each box

# scatterplot
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])

# plots a best-fit line with uncertainty from data (also includes the scatterplot)
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])

# adding colour based on a value of some col
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])

# plots MULTIPLE best-fit lines based on value of some other col
sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)

# swarmplot - scatterplot for categorical data
sns.swarmplot(x=insurance_data['smoker'],
              y=insurance_data['charges'])

# Histogram 
sns.histplot(iris_data['Petal Length (cm)'])

# Histograms for each species
sns.histplot(data=iris_data, x='Petal Length (cm)', hue='Species')

# KDE plot 
sns.kdeplot(data=iris_data['Petal Length (cm)'], fill=True)

# KDE plots for each species
sns.kdeplot(data=iris_data, x='Petal Length (cm)', hue='Species', fill=True)

# 2D KDE plot
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")

# Change the style of the figure to the "dark" theme
sns.set_style("dark")
#Seaborn has five different themes: (1)"darkgrid", (2)"whitegrid", (3)"dark", (4)"white", and (5)"ticks"