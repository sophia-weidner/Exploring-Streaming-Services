#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# In[47]:


# Uploading dataframe.
df = pd.read_csv('/Users/sophiaweidner/Downloads/disney_plus_titles.csv')
df.head(5)


# In[48]:


# Checking NaN values.
df[df.isna().any(axis=1)]


# NaN values in director, country, date_added, rating.
# 
# - I am going to replace NaN values in director with "No director listed"
# - I am going to remove the country column. This is a column that is not necessary for my study on variety.
# - I am going to remove the date_added column. This is the date that the movie was added to Disney plus. This is not necessary for my study on variety.
# - I am going to replace NaN values in rating column with "unknown".

# In[49]:


# Dropping country and date_added columns.
df = df.drop(['country'], axis = 1)
df = df.drop(['date_added'], axis = 1)


# In[50]:


df.head()


# In[51]:


# Replacing NaN values in director column with 'no director listed'
df['director'] = df['director'].fillna('No Director Listed')


# In[52]:


df.head()


# In[53]:


# Replacing NaN values in rating column with 'unknown'
df['rating'] = df['rating'].fillna('Unknown')


# In[54]:


# Checking NaN values again.
df[df.isna().any(axis=1)]


# In[55]:


# Replacing NaN in cast column with 'no cast listed'
df['cast'] = df['cast'].fillna('No Cast Listed')


# In[56]:


# Checking NaN values again.
df[df.isna().any(axis=1)]


# Since there are not any NaN values listed in the dataframe anymore, I am going to move forward with creating visualizations.
# - I want an overall look at number of movies versus number of television shows.
# - I want an overall look at the directors and how many movies have the same directors.
# - I want an overall look at the release years.
# - I want an overall look at the categories of movies and tv shows.

# In[57]:


# Number of movies and televsion shows.
movies = len(df['type']=='Movie')
shows = len(df['type']=='TV Show')

print(movies, shows)


# In[58]:


# Creating height of bar chart
y_pos = np.arange(len(df['type']))


# In[59]:


# Creating bar chart of Movies vs. TV Shows
plt.bar(df['type'], y_pos, color ='blue',
        width = 0.4)

plt.xlabel("Televison or Movies")
plt.ylabel("Amount of Media")
plt.title("Type of Media on Disney Plus")

plt.show()


# In[60]:


df['director'].tolist()


# In[61]:


df_dir = df[df.director != "No Director Listed"]


# In[62]:


df_dir.head()


# In[63]:


df_dir = df_dir[df_dir['type'] == 'Movie'] 


# In[64]:


df_dir.head()


# In[67]:


# Creating a graph with top 20 directors 
top_20 = (df_dir['director'].value_counts()).iloc[:20]


# In[75]:


top_20.plot(kind='bar', xlabel = 'Director', 
            ylabel = 'Number of Movies', title = 'Top 20 Directors of Movies on Disney Plus', color = 'maroon')


# In[77]:


# Categories of movies and tv shows. listed_in
df['listed_in'].tolist()


# In[78]:


df_cat_movie = df[df.type != "TV Show"]
df_cat_show = df[df.type != "Movie"]


# In[79]:


df_cat_movie.head()


# In[81]:


# Splitting listed_in category for movies by ','.
df_cat_movie[['cat1', 'cat2', 'cat3']] = df_cat_movie.listed_in.str.split(",", expand = True)


# In[117]:


counts = df_cat_movie['cat1'].value_counts()

# Create a pie chart
fig, ax = plt.subplots()
textprops = {'fontsize': 10, 'color': 'black'}
ax.pie(counts, labels=counts.index, autopct='%1.0f%%', textprops=textprops, labeldistance=1.1, pctdistance=0.7)
ax.set_title('Pie Chart First Category listed for each Movie on Disney Plus')
ax.legend(labels=counts.index, loc='upper left', bbox_to_anchor=(1.2, 0.5))

# Show the chart
plt.show()


# In[118]:


counts2 = df_cat_movie['cat2'].value_counts()

# Create a pie chart
fig, ax = plt.subplots()
textprops = {'fontsize': 10, 'color': 'black'}
ax.pie(counts2, labels=counts2.index, autopct='%1.0f%%', textprops=textprops, labeldistance=1.1, pctdistance=0.7)
ax.set_title('Pie Chart Second Category listed for each Movie on Disney Plus')
ax.legend(labels=counts2.index, loc='upper left', bbox_to_anchor=(1.2, 0.5))

# Show the chart
plt.show()


# In[119]:


counts3 = df_cat_movie['cat3'].value_counts()

# Create a pie chart
fig, ax = plt.subplots()
textprops = {'fontsize': 10, 'color': 'black'}
ax.pie(counts3, labels=counts3.index, autopct='%1.0f%%', textprops=textprops, labeldistance=1.1, pctdistance=0.7)
ax.set_title('Pie Chart Third Category listed for each Movie on Disney Plus')
ax.legend(labels=counts3.index, loc='upper left', bbox_to_anchor=(1.2, 0.5))

# Show the chart
plt.show()


# In[129]:


# Splitting release years to before and after 1980.
before_1980 = df[df.release_year <= 1980]
after_1980 = df[df.release_year > 1980]


# In[133]:


y = before_1980['release_year'].value_counts().sort_index()

plt.bar(y.index, y.values, color='green', width=0.4)

plt.xlabel("Release Year")
plt.ylabel("Amount of Media")
plt.title("Year of Release before 1980 for TV Shows or Movies")

plt.show()


# In[141]:


y2 = after_1980['release_year'].value_counts().sort_index()

plt.bar(y2.index, y2.values, color='green', width=0.4)

plt.xlabel("Release Year")
plt.ylabel("Amount of Media")
plt.title("Year of Release after 1980 for TV Shows or Movies")

plt.show()


# # Data Analysis: Linear Regression Analysis
# 
# For my business problem, I want to address the variety of media provided by Disney plus for its consumers. I was left deciding between Regression Analysis and Cluster Analysis, and ultimately decided to try Regression analysis first. The reason for this is Regression analysis could help to identify any relationships between factors, such as the year of release or rating. Since the data is largely categorical, I am going to have to create dummy variables.

# In[149]:


df.head()


# In[150]:


# dropping show_id, cast and description columns. We will not be analyzing variety based on these factors, 
# and because I am creating dummy variables, I don't want an overload of datapoints that are unncessary to
# the analysis.

df = df.drop(['show_id'], axis = 1)
df = df.drop(['cast'], axis = 1)
df = df.drop(['description'], axis = 1)


# In[151]:


df.head(2)


# In[165]:


# Creating dummy variables for the entire dataframe.
df_dummies = pd.get_dummies(df)


# In[166]:


from sklearn.linear_model import LinearRegression


# In[167]:


df_dummies.columns.tolist()


# In[177]:


from sklearn.metrics import mean_squared_error, r2_score

# Splitting DF into independent and dependent variables.

X_lr = df_dummies.drop('release_year', axis=1)  # X contains all independent variables
y_lr = df_dummies['release_year']  # y contains the dependent variable

# Split the data into training and test sets randomly
X_train, X_test, y_train, y_test = train_test_split(X_lr, y_lr, test_size=0.3, random_state=42)


# In[178]:


# Train a linear regression model on the training set
lm.fit(X_train, y_train)

# Use the model to make predictions on the test set
y_pred = lm.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R-squared:", r2)

