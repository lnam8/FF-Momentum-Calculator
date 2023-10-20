# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:32:37 2023

@author: asann
"""

# =============================================================================
# Source: https://www.kaggle.com/code/abhish92sme/kmean-clustering-on-football-player-season-19-20
#
# There are some warnings that arise (at least for me) Regarding memory leaks
# with kMeans. Not sure how to rectify.
#
# Another warning says 'Polyfit may be poorly conditioned', that is just because
# the trend-finding function attempts to polyfit week 1's data.
#
# =============================================================================




import pandas as pd 
import plotly.express as px
from plotly.offline import plot
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =============================================================================
# Got copy/slice errors I did not feel like rectifying
# =============================================================================
pd.options.mode.chained_assignment = None


ff = pd.read_csv('weekly_data.csv')


# =============================================================================
# Calculate standard points and ppr points
# =============================================================================
ff = ff.assign(standard_points=lambda x: (x.receiving_yards*0.1)\
                                 + (x.rushing_yards*0.1) + (x.receiving_touchdowns*6)\
                                 + (x.rushing_touchdowns*6))

ff = ff.assign(ppr_points=lambda x: (x.receiving_yards*0.1) + (x.receptions*1)\
                                 + (x.rushing_yards*0.1) + (x.receiving_touchdowns*6)\
                                 + (x.rushing_touchdowns*6))
    
# =============================================================================
# Drop some duplicate data that appeared for the key players
# =============================================================================
ff = ff.drop_duplicates(subset=['player_name', 'week']).reset_index(drop=True)


# =============================================================================
# Just an Interesting plot of the point distributions. Interactive via plotly
# and should open in browser
# =============================================================================
fig = px.scatter(ff, x='ppr_points', y = 'week', color='player_name')
fig.update_traces(textposition='top center')
plot(fig)


# =============================================================================
# This should probably be refined. Just a quick attempt at indetifying if a
# player is trending up or down based on weekly performance
# =============================================================================
def trenddetector(list_of_index, array_of_data, order=1):
    result = np.polyfit(list_of_index, list(array_of_data), order)
    slope = result[-2]
    return float(slope)



# =============================================================================
# Below the 'slope' column is made which corresponds to how the player is 
# trending as of that week
# =============================================================================
df = ff.copy()
df['slope'] = df.apply(lambda x: trenddetector(ff[(ff.player_name == x.player_name) & (ff.week <= x.week)].week.to_numpy(),
                                            ff[(ff.player_name == x.player_name) & (ff.week <= x.week)].ppr_points.to_numpy(),
                                            1), axis =1)


###############################################################################
###############################################################################
# K-Means Starts Here                                                         #
###############################################################################
###############################################################################


# =============================================================================
# DataFrame consisting of only the latest weeks data
# =============================================================================
latest = df[df.week == max(df.week)]


# =============================================================================
# Only looking at PPR and slope data to cluster on
# =============================================================================
X = latest.iloc[:, [15,16]].values
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    

# =============================================================================
# Elbow plot to determine how many clusters. Was sort of on the fence of 3 & 4
# =============================================================================
plt.plot(range(1,11), wcss,)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

kmmodel = KMeans(n_clusters=3 , init='k-means++', random_state=0)
y_kmeans= kmmodel.fit_predict(X)
labels = kmmodel.labels_


# =============================================================================
# Add some labels
# =============================================================================
latest["label"]=labels
latest.loc[latest['label'] == 2, 'Category'] = '2'
latest.loc[latest['label'] == 1, 'Category'] = '1'
latest.loc[latest['label'] == 0, 'Category'] = '0'


# =============================================================================
# Plot again via plotly, should load in browser
# =============================================================================
fig = px.scatter(latest,x="ppr_points",y="slope", color ='Category', text="player_name", title="K-mean clustering of PPR Points vs Slope")
fig.update_traces(textposition='top center')
plot(fig)
