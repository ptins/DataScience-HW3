# CSE40647/CSE60647 - HW3

This is the third assignment for my Data Science Class.

## Getting Started

This project uses Python 3.6.4


## Task 1

Use Python to do K Means Clustering with two features (1) #Wins in 2015 and (2) Wins in 2017. Suppose the number of clusters is K = 2. Use Euclidean distance as the distance metric. Initialize your algorithm with the following centroids:

```
(7,7) and (14,14).
(7,7) and (7,14).
```

Do they generate the same result? Which initialization do you prefer and why?

Code:
```
df = pd.read_csv('data.txt', delimiter='\t')
df.set_index('College', inplace=True)
df.head()

df_t1 = df.drop(['Rank_2015','Rank_2017'], axis=1)
df_t1.columns = ['x','y']

k = 2

# centroids = {
#     1: np.array([7,7]),
#     2: np.array([14,14])
# }

centroids = {
    1: np.array([7,7]),
    2: np.array([7,14])
}

colmap = {1: 'r', 2: 'b'}

df_t1 = assignment(df_t1, centroids, 'euclidean')

counter = 0
while True:
    counter += 1
    print('Iteration #{}'.format(counter))
    closest_centroids = df_t1['closest'].copy(deep=True)
    centroids = update(df_t1, centroids)
    df_t1 = assignment(df_t1, centroids, 'euclidean')
    if closest_centroids.equals(df_t1['closest']):
        break

fig = plt.figure()
plt.scatter(df_t1['x'], df_t1['y'], color=df_t1['color'], alpha=0.3, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.show()
```

The graphs produced are the same; the initializations generate the same result. However, we prefer the second initialization to the first because it converges after only one iteration while the other converges after two.

## Task 2

Use Python to do K Means Clustering with two features (1) Ranking in 2015 and (2) Ranking in 2017. Suppose the number of clusters is K = 2. Use Manhattan distance as the distance metric. Initialize your algorithm with the centroids (1,1) and (25,25). Compared with cluster results in Question 1, do you prefer the clustering based on these two new features more or less?

Code:
```
df = pd.read_csv('data.txt', delimiter='\t')
df.set_index('College', inplace=True)
df.head()

df_t2 = df.drop(['Win_2015','Win_2017'], axis=1)
df_t2.columns = ['x','y']

k = 2

centroids = {
    1: np.array([1,1]),
    2: np.array([25,25])
}

colmap = {1: 'r', 2: 'b'}

df_t2 = assignment(df_t2, centroids, 'manhattan')

counter = 0
while True:
    counter += 1
    print('Iteration #{}'.format(counter))
    closest_centroids = df_t2['closest'].copy(deep=True)
    centroids = update(df_t2, centroids)
    df_t2 = assignment(df_t2, centroids, 'manhattan')
    if closest_centroids.equals(df_t2['closest']):
        break

fig = plt.figure()
plt.scatter(df_t2['x'], df_t2['y'], color=df_t2['color'], alpha=0.3, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.show()
```

Between the first and second problems, we prefer the first two features because it takes fewer iterations to converge, and the plot suggests K=3 might be more appropriate; this can be seen in the blue cluster with the centroid rather far from all points in said cluster.


## Task 3

Use Python to do K Means Clustering with two features (1) Ranking in 2015 and (2) Ranking in 2017. Suppose the number of clusters is K = 3. Use Manhattan distance as the distance metric.

Code:
```
df = pd.read_csv('data.txt', delimiter='\t')
df.set_index('College', inplace=True)
df.head()

df_t3 = df.drop(['Win_2015','Win_2017'], axis=1)
df_t3.columns = ['x','y']

k = 3

# plt.scatter(df_t3['x'], df_t3['y'], alpha=0.3, edgecolor='k')

centroids = {
    1: np.array([5,5]),
    2: np.array([8,15]),
    3: np.array([20,15])
}

colmap = {1: 'r', 2: 'b', 3: 'g'}

df_t3 = assignment(df_t3, centroids, 'manhattan')

counter = 0
while True:
    counter += 1
    print('Iteration #{}'.format(counter))
    closest_centroids = df_t3['closest'].copy(deep=True)
    centroids = update(df_t3, centroids)
    df_t3 = assignment(df_t3, centroids, 'manhattan')
    if closest_centroids.equals(df_t3['closest']):
        break

fig = plt.figure()
plt.scatter(df_t3['x'], df_t3['y'], color=df_t3['color'], alpha=0.3, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.show()
```

Upon inspecting the first plot, we choose the initial centroids to be (5,5), (8,15), and (20,15); the resulting clusters can be seen in the second plot. Compared to K=2, we prefer K=3 because it converges after one iteration, and the clusters more appropriately describe the trends in the data. By adding a third cluster, we see centroids that are much closer to the data points, especially in the green cluster, which spread out the blue cluster in the plot from Problem 2.
