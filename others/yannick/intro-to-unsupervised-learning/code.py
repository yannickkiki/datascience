from sklearn import datasets

# Loading dataset
iris_df = datasets.load_iris()

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(iris_df.data)
predictions = model.predict(iris_df.data)
