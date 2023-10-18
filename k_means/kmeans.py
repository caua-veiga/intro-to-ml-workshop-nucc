import numpy as np
import matplotlib.pyplot as plt

class Kmeans():

    def __init__(self, X, K, initializer='++'):
        assert initializer == '++' or initializer == 'standard', f'{initializer} is initializer  method is not implemented.' \
                                                                 f'Please use either ++ or standard'

        self.X = np.array(X)
        self.K = K
        self.num_examples, self.num_features = X.shape
        self.initializer = '++' #initializer  # The standard method has yet to be implemented

    def initialize_centroids(self):
        centroids = np.zeros((self.K, self.num_features))

        if self.initializer == 'standard':
            # TODO - Ask the professor because I didn't really get the ideia here. Should it be random initialized wit
            #  with what constrains? In the entire space? As a point from the data? etc...
            centroids = np.random.rand(self.K, self.num_features)

        elif self.initializer == '++':
            possible_ind = np.arange(self.num_examples)

            pseudo_ind = np.random.choice(len(possible_ind), 1, replace=False)
            first_centroid_ind = possible_ind[pseudo_ind]
            centroids[0] = self.X[first_centroid_ind]
            deleted_ind = [first_centroid_ind]
            possible_ind = np.delete(possible_ind, first_centroid_ind)

            for k in range(1, self.K):
                indep_sqr_dist = np.array([self.EuclideanDistance(centroids[k-1], x)**2 for i,x in enumerate(self.X) if i not in deleted_ind])
                probs = indep_sqr_dist/np.sum(indep_sqr_dist)

                pseudo_ind = np.random.choice(len(possible_ind), 1, p=probs, replace=False)
                centroid_ind = possible_ind[pseudo_ind]

                centroids[k] = self.X[centroid_ind]

                deleted_ind.append(centroid_ind)
                possible_ind = np.delete(possible_ind, pseudo_ind)

        return centroids

    def get_cluster(self, centroids, X):
        clusters = [[] for _ in range(self.K)]

        for point_ind, point in enumerate(X):
            closest_centroid = np.argmin(np.sqrt(np.sum( (point-centroids)**2, axis=1)))
            clusters[closest_centroid].append(point_ind)

        return clusters

    def get_centroids(self, clusters, X):
        X_clustered = []

        for cluster in clusters:
            X_clustered.append(np.mean(X[cluster], axis=0))

        X_clustered = np.array(X_clustered)
        return X_clustered

    def plot2D(self, centroids, clusters):
        colors = ['green', 'red', 'black',  'blue', 'cyan', 'pink']


        x = self.X[:, 0]
        y = self.X[:, 1]

        plt.figure(figsize=(14,8))
        for i, cluster in enumerate(clusters):
            plt.scatter(x[cluster], y[cluster], c=colors[i], alpha=.5)
            centroid_x = centroids[i][0]
            centroid_y = centroids[i][1]
            plt.scatter(centroid_x, centroid_y, c=colors[i], marker='x')

        plt.show()

    def fit(self):
        centroids = self.initialize_centroids()
        new_centroids = centroids * 2 # just for the first iteration of the while loop
        Error = self.Error(centroids, new_centroids)
        while Error > 1e-15:
            clusters = self.get_cluster(centroids, self.X)
            self.plot2D(centroids, clusters)
            new_centroids = self.get_centroids(clusters, self.X)
            Error = self.Error(centroids, new_centroids)
            #print(abs(new_centroids - centroids))
            centroids = new_centroids

        return centroids, clusters


    def EuclideanDistance(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.sqrt(np.sum((a - b) ** 2))

    def Error(self, old_centroids, new_centroids):
        return np.max(abs(new_centroids - old_centroids))



if __name__ == '__main__':
    from additional_stuf import simulated_dataset
    #X = np.random.rand(450,2) * 25

    X, _ = simulated_dataset()
    plt.figure(figsize=(14,8))
    x = X[:, 0]
    y = X[:, 1]
    plt.scatter(x,y)
    plt.show()
    test = Kmeans(X=X, K=4, initializer='standard')
    test.fit()