from k_means import * 
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set(rc={'figure.figsize':(20,20)})


# all samples must be in float 
training_sample = pd.DataFrame(data = {'x' : [random.uniform(0, 1) for _ in range(100)],
                                        'y' : [random.uniform(1, 5) for _ in range(100)]})



#create KMeans object
max_iters = 3
k_means_obj = KMeans(n_clusters=4, 
                     max_iter=max_iters, plot_every = 1)
# #train the KMeans object
clusters, centroids = k_means_obj.fit(training_sample)
print(clusters)
print(centroids)

new_sample = pd.DataFrame(data = {'x' : [random.uniform(0, 1) for _ in range(10)],
                                  'y' : [random.uniform(1, 5) for _ in range(10)]})

predicted_clusters = k_means_obj.predict(new_sample)
print(predicted_clusters)

k_means_obj.plot()