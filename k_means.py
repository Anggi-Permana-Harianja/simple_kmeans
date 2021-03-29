#Simple K-Means functions that implemented similary with scikit-learn module--------
#methods:
#   - fit: train the model
#       - parameters
#           - n_clusters = number of cluster/centroids to be generated
#           - max_iter = maximum iteration
#           - plot_every = plot every n iteratiion
#       - return
#           - closest_centroid = closest centroid where the sample belongs to
#           - centroids = centroid's last position
#   - predict: predict new sample
#       - parameters
#           - new sample to be predicted
#       - return
#           - predict_clusters = clusters given to each of predict sample
#   - plot: plot the current position of centroids and clusters
#------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
sns.set(rc={'figure.figsize':(20,20)})
import matplotlib.pyplot as plt 
from math import sqrt
import random



class KMeans:
    #initalize instance variables
    def __init__(self, n_clusters = 2, max_iter = 10, plot_every = 0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.plot_every = plot_every


    #train method
    def fit(self, X):
        self.X = X #so that training sample could be used for all methods
        global find_closest_centroid #so it could be used for prediction purposes
        #set initial centroids
        ##get min/max value of col x and col y so that initial centroids could be sparse enough
        max_x_value = X['x'].max()
        min_x_value = X['x'].min()
        max_y_value = X['y'].max()
        min_y_value = X['y'].min()
        ##set random initial centroids position
        self.centroids = pd.DataFrame(data = {'x' : [random.uniform(min_x_value, max_x_value) for _ in range(self.n_clusters)],
                                              'y' : [random.uniform(min_y_value, max_y_value) for _ in range(self.n_clusters)]})
        

        #find closet centroid function
        def find_closest_centroid(X, centroids):
            all_distances = []

            #get shape of row        
            self.nrow = self.X.shape[0]

            for i in range(0, self.nrow):
                distance = np.zeros(self.n_clusters)
                for j in range(0, self.n_clusters):
                    #calculate euclidean distance for each sample on each centroid
                    euclid_distance = sqrt((self.X.iloc[i][0] - centroids.iloc[j][0])**2 + (self.X.iloc[i][1] - centroids.iloc[j][1])**2)
                    distance[j] = euclid_distance
                all_distances.append(distance.tolist())

            #return idx of closest centroid for each sample
            returned_idx = [all_distances[i].index(min(all_distances[i])) for i in range(0, len(all_distances))]

            return returned_idx

        #compute new centroids movement
        def compute_centroids(X, closest_centroid, nk):
            for i in range(0, self.n_clusters):
                #get sample idx that assigned to closest_centroid
                closest_sample = [j for j in range(0, len(closest_centroid)) if i == closest_centroid[j]]
                mean_x_centroid = self.X.iloc[closest_sample]['x'].mean()
                mean_y_centroid = self.X.iloc[closest_sample]['y'].mean()
                
                #assign the new value to existing centroid
                self.centroids.loc[i, 'x'] = mean_x_centroid
                self.centroids.loc[i, 'y'] = mean_y_centroid
            
            return self.centroids
            
        #main part of training--------------------------------------------------
        for i in range(0, self.max_iter):
            print('training iteration: {}'.format(i + 1))

            #give each sample centroid id with the nearest distance
            self.closest_centroid = find_closest_centroid(self.X, self.centroids)
            
            #given the membership, compute new centroids position
            self.centroids = compute_centroids(self.X, self.closest_centroid, self.n_clusters)
            
            #run the animation visualization
            if(self.plot_every > 0) and ((i + 1) % self.plot_every == 0):
                self.plot()

        #this last one is for closest centroid after the last compute_centroids
        self.closest_centroid = find_closest_centroid(self.X, self.centroids)

        print('finished training...')
        return self.closest_centroid, self.centroids
        #-----------------------------------------------------------------------

    def predict(self, new_X):
        #get the closests centroid using this predict sample
        predict_clusters = find_closest_centroid(new_X, self.centroids)

        return predict_clusters

    def plot(self):
        print('Plotting current clusters and centroid(s) position')
        concat_ = pd.concat([self.X.assign(cluster = self.closest_centroid), self.centroids.assign(cluster = 'centroids')])
        sns.scatterplot(x = 'x', y = 'y', 
                        data = concat_, 
                        hue = 'cluster', 
                        s = 100)
        plt.pause(0.05)
        plt.show()

        return

            
 