import sqlite3
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

#Load the cleaned CSV
df = pd.read_csv("genre_movies.csv")
# Filter: Movies with significant votes (let's say > 50,000 votes)
popular_movies = df[df['Votes'] > 50000]
# Sort by Rating (descending) and pick Top 10
top_movies = popular_movies.sort_values(by='Rating', ascending=False).head(10)
# Plot: Bar chart of Top 10 Movies
plt.figure(figsize=(12, 8))
plt.barh(top_movies['MovieName'], top_movies['Rating'], color='orange')
plt.xlabel('Rating', fontsize=12)
plt.title('Top 10 Movies by Rating (With Votes > 50,000)', fontsize=16)
plt.gca().invert_yaxis()  # Highest rating at top
plt.tight_layout()
plt.show()
# Optional: View the DataFrame
print(top_movies[['MovieName', 'Genre', 'Rating', 'Votes']])




from matplotlib import pyplot as plt
df= pd.read_csv("genre_movies.csv")
# Count number of movies in each genre and plot it
df['Genre'].value_counts().plot(kind='bar', figsize=(10, 6), color='skyblue')
#Customize chart
plt.title('Genre Distribution')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.tight_layout()
#Display chart
plt.show()
#Assuming you already have your cleaned DataFrame `df`
df=pd.read_csv("genre_movies.csv")


# Group by Genre and compute average Duration
avg_duration = df.groupby('Genre')['DurationMinutes'].mean().sort_values()
#2. Plot
plt.figure(figsize=(10, 6))  # Set figure size before plotting
avg_duration.plot(kind='barh', color='skyblue')  # Horizontal bar for better readability
plt.title('Average Movie DurationMinutes by Genre', fontsize=16)
plt.xlabel('Average DurationMinutes (minutes)', fontsize=12)   # X-axis is duration
plt.ylabel('Genre', fontsize=12)
plt.tight_layout()
plt.show()


#Step 4: Group and calculate average votes
#avg_votes = df.groupby('Genre')['Votes'].mean().reset_index()
avg_votes = avg_votes.sort_values(by='Votes', ascending=False)
#Step 6: View average votes result
print(avg_votes)
#Plot using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_votes, x='Genre', y='Votes', palette='viridis')
plt.title('Average Votes by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Votes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Rating'], color='lightgreen')
plt.title('Boxplot of Movie Ratings')
plt.xlabel('Rating')
plt.show()



# Top-rated movie for each genre
top_movies = df.loc[df.groupby('Genre')['Rating'].idxmax()][['Genre', 'MovieName', 'Rating']]
#View the result
print(top_movies.reset_index(drop=True))



# Group by Genre and sum the cleaned votes
genre_votes = df.groupby('Genre')['Votes'].sum().sort_values(ascending=False)
#Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(genre_votes, labels=genre_votes.index, autopct='%1.1f%%', startangle=140)
plt.title('Most Popular Genres by Total Votes')
plt.axis('equal')  # Equal aspect ratio ensures pie is a circle.
plt.tight_layout()
plt.show()

#Find shortest and longest movies
shortest = df.loc[df['DurationMinutes'].idxmin(), ['MovieName', 'DurationMinutes']]
longest = df.loc[df['DurationMinutes'].idxmax(), ['MovieName', 'DurationMinutes']]
#Display
duration_extremes = pd.DataFrame([shortest, longest], index=['Shortest', 'Longest'])
print(duration_extremes)
#Shortest= 'A Christmas Castle Proposal: A Royal in Paradise...'0min
#Longest='13 Lentes De Um Final Feliz (Documentário Completo)'410


# Step 1: Group by Genre and compute average Rating
genre_avg_rating = df.groupby('Genre')['Rating'].mean().reset_index()
# Step 2: Convert to pivot format for heatmap
# We set Genre as index and Rating as value column
heatmap_data = genre_avg_rating.pivot_table(index='Genre', values='Rating')
#Step 3: Plot heatmap
plt.figure(figsize=(6, 8))  # Taller to fit genre labels
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', linewidths=0.5, cbar_kws={'label': 'Average Rating'})
plt.title('Average Ratings by Genre')
plt.xlabel('')  # No x-axis label needed
plt.ylabel('Genre')
plt.tight_layout()
# plt.show()


# plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Votes', y='Rating')
plt.title('Correlation Between Votes and Ratings')
plt.xlabel('Total Votes')
plt.ylabel('Rating')
plt.grid(True)
plt.tight_layout()
# plt.show()




import streamlit as st
import sqlite3
import pandas as pd
# Title
st.title("🎬 Top 10 Movies by Rating & Voting Count")
# Connect to SQLite database (or load CSV)
conn = sqlite3.connect("films.db")  # Replace with your DB path
query = """
SELECT [MovieName], Rating, Votes
FROM film
WHERE Votes > 10000
ORDER BY Rating DESC, Votes DESC
LIMIT 10;
"""
df = pd.read_sql_query(query, conn)
conn.close()
# Show in Streamlit
st.write("Here are the top 10 movies with highest ratings and strong voting engagement:")
st.dataframe(df)




import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
# Title
st.title("Genre Distribution from SQL Database")
# Connect to the SQLite database
conn = sqlite3.connect('films.db')  # Make sure your database is named correctly
# SQL Query to get Genre Count
query = """
SELECT Genre, COUNT(*) as film_count
FROM film
GROUP BY Genre
ORDER BY film_count DESC;
"""
df = pd.read_sql_query(query, conn)
# Display DataFrame
st.write("Movie Genre Counts:", df)
# Plot
fig, ax = plt.subplots()
ax.bar(df['Genre'], df['film_count'], color='skyblue')
ax.set_xlabel("Genre")
ax.set_ylabel("Number of Movies")
ax.set_title("Genre Distribution")
st.pyplot(fig)
# Close the connection
conn.close()




import streamlit as st
import pandas as pd
import sqlite3
# Title
st.title("🎬 Duration Insights: Average Movie Duration by Genre")
# Connect to the SQLite database (replace with your actual database connection)
conn = sqlite3.connect('films.db')  # Replace 'movies.db' with your database
cursor = conn.cursor()
# SQL Query: Average Duration by Genre
query = """
SELECT Genre, ROUND(AVG(Duration), 2) as average_Duration
FROM film
GROUP BY Genre
ORDER BY average_Duration DESC;
"""
# Execute Query and Load Data
df = pd.read_sql_query(query, conn)
# Close connection (optional)
conn.close()
# Display the DataFrame
st.dataframe(df, use_container_width=True)
# Optional: Add Bar Chart
st.bar_chart(df.set_index('Genre')['average_Duration'])




import streamlit as st
import pandas as pd
import sqlite3
# App title
st.title("⭐ Genres with the Highest Average Voting Counts")
# Connect to the database
conn = sqlite3.connect('films.db')  # Replace with your actual database
cursor = conn.cursor()
# SQL Query: Average Voting Counts by Genre
query = """
SELECT Genre, ROUND(AVG(votes), 2) as average_votes
FROM film
GROUP BY Genre
ORDER BY average_votes DESC;
"""
# Run query and load data
df = pd.read_sql_query(query, conn)
# Close the connection
conn.close()
# Display results
st.dataframe(df, use_container_width=True)
# Optional: Plot as bar chart
st.bar_chart(df.set_index('Genre')['average_votes'])





import streamlit as st
import pandas as pd
import sqlite3
from matplotlib import pyplot as plt
# Streamlit Title
st.title(" Dominant IMDb 2024 Genres by Movie Count (Matplotlib)")
# Connect to database
conn = sqlite3.connect('films.db')
# SQL Query to get dominant genres
query = """
SELECT Genre, COUNT(*) AS MovieCount
FROM film
GROUP BY Genre
ORDER BY MovieCount DESC
"""
# Run query
df = pd.read_sql_query(query, conn)
conn.close()
# Display data
st.dataframe(df)
# Plot using matplotlib
fig, ax = plt.subplots(figsize=(10,6))
ax.bar(df['Genre'], df['MovieCount'], color='skyblue')
ax.set_title('Most Frequent Movie Genres in 2024')
ax.set_xlabel('Genre')
ax.set_ylabel('Number of Movies')
plt.xticks(rotation=45)
st.pyplot(fig)





import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
st.title("Distribution of Movie Ratings")
conn = sqlite3.connect('films.db')
query = "SELECT Rating FROM film"
df = pd.read_sql_query(query, conn)
conn.close()
fig, ax = plt.subplots()
ax.hist(df['Rating'], bins=10, color='skyblue', edgecolor='black')
ax.set_xlabel('Rating')
ax.set_ylabel('Number of Movies')
ax.set_title('Distribution of Movie Ratings')
st.pyplot(fig)





import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
st.title("Average Ratings by Genre")
conn = sqlite3.connect('films.db')
query = """
SELECT Genre, AVG(Rating) AS AvgRating
FROM film
GROUP BY Genre
ORDER BY AvgRating DESC
"""
df = pd.read_sql_query(query, conn)
conn.close()
fig, ax = plt.subplots()
ax.bar(df['Genre'], df['AvgRating'], color='orange', edgecolor='black')
ax.set_xlabel('Genre')
ax.set_ylabel('Average Rating')
ax.set_title('Average Ratings by Genre')
plt.xticks(rotation=45)
st.pyplot(fig)




import streamlit as st
import pandas as pd
import sqlite3
st.title("Shortest and Longest Movies in 2024")
conn = sqlite3.connect('films.db')
# Shortest movie query excluding Duration = 0
query_shortest = """
SELECT MovieName, Duration
FROM film
WHERE Duration = (SELECT MIN(Duration) FROM film WHERE Duration > 0)
"""
#Longest movie query
query_longest = """
SELECT MovieName, Duration
FROM film
WHERE Duration = (SELECT MAX(Duration) FROM film)
"""
df_shortest = pd.read_sql_query(query_shortest, conn)
df_longest = pd.read_sql_query(query_longest, conn)
conn.close()
st.subheader("🎬 Shortest Movie(s)")
st.dataframe(df_shortest)
st.subheader("🎬 Longest Movie(s)")
st.dataframe(df_longest)





import streamlit as st
import sqlite3
import pandas as pd
from matplotlib import pyplot as plt
st.title("Top 10 Movies with Highest Voting Counts")
# Connect to the database
conn = sqlite3.connect('films.db')
# SQL query to get top 10 movies by Votes
query = """
SELECT MovieName, Votes
FROM film
ORDER BY Votes DESC
LIMIT 10
"""
# Execute query and load into DataFrame
df_top10 = pd.read_sql_query(query, conn)
conn.close()
# Display in Streamlit
st.dataframe(df_top10)








import streamlit as st
import sqlite3
import pandas as pd
from matplotlib import pyplot as plt
st.title("📊 Ratings vs. Votes Scatter Plot (Matplotlib)")
# Connect to database
conn = sqlite3.connect('films.db')
# SQL: Filter out rows where Rating, Votes, or Duration are zero or null
query = """
SELECT MovieName, Genre, Rating, Votes, Duration
FROM film
WHERE 
    CAST(Rating AS REAL) > 0 AND 
    CAST(Votes AS INTEGER) > 0 AND 
    CAST(Duration AS INTEGER) > 0
"""
df = pd.read_sql_query(query, conn)
conn.close()
# Convert to numeric (string to numbers safely)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
# Filter: Keep rows where Rating, Votes, Duration > 0 (but no column drop)
filtered_df = df[(df['Rating'] > 0) & (df['Votes'] > 0) & (df['Duration'] > 0)]
# Scatter plot with Matplotlib
if not filtered_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(filtered_df['Votes'], filtered_df['Rating'], alpha=0.7)
    ax.set_xlabel('Votes')
    ax.set_ylabel('Rating')
    ax.set_title('Ratings vs. Votes')
    ax.grid(True)

    st.pyplot(fig)
else:
    st.info("No movies available after filtering.")
    
    
    
    
import streamlit as st
import sqlite3
import pandas as pd
# Connect to database
conn = sqlite3.connect('films.db')
cursor = conn.cursor()
st.title(" Interactive Filtering Functionality")
# Sidebar Filters (No Expander)
st.sidebar.header(" Apply Filters")
# Genre filter
cursor.execute("SELECT DISTINCT Genre FROM film")
genres = [row[0] for row in cursor.fetchall()]
selected_genre = st.sidebar.selectbox("Genre:", ["All"] + genres)
# Duration filter
duration_option = st.sidebar.radio("Duration:", ["All", "< 2 hrs", "2–3 hrs", "> 3 hrs"])
# Rating filter
min_rating = st.sidebar.slider("Minimum IMDb Rating:", 0.0, 10.0, 0.0, 0.1)
# Votes filter
min_votes = st.sidebar.number_input("Minimum Votes:", min_value=0, value=1, step=1000)
# Build SQL WHERE conditions
where_clauses = []
if selected_genre != "All":
    where_clauses.append(f"Genre = '{selected_genre}'")
if duration_option == "< 2 hrs":
    where_clauses.append("Duration < 120")
elif duration_option == "2–3 hrs":
    where_clauses.append("Duration >= 120 AND Duration <= 180")
elif duration_option == "> 3 hrs":
    where_clauses.append("Duration > 180")
# Ratings and votes must always be > 0
where_clauses.append("Rating > 0")
where_clauses.append("Votes > 0")
if min_rating > 0:
    where_clauses.append(f"Rating >= {min_rating}")
if min_votes > 0:
    where_clauses.append(f"Votes >= {min_votes}")
# Final Query
where_query = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
final_query = f"""
    SELECT MovieName, Genre, Rating, Votes, Duration 
    FROM film 
    {where_query}
"""

# Query execution
df = pd.read_sql_query(final_query, conn)
# Safe casting
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(0)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0).astype(int)
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce').fillna(0).astype(int)
# Final post-filter to avoid zero ratings/votes
df = df[(df['Rating'] > 0) & (df['Votes'] > 0)]
# Display filtered results
st.dataframe(df)
# Show total only (No extra titles or headers)
st.write(f" Total Movies Found: {len(df)}")
conn.close()




