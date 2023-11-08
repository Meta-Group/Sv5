import pandas as pd
from os.path import join
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score, adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans

import numpy as np

import random
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

def _algorithm_space_minibatch(benchmarks_list):
    output_path = "clustering_solutions.csv"
    
    for idx, file_name in enumerate(benchmarks_list):
        print(str(idx), "/", str(len(benchmarks_list)))
        file_name = file_name.replace("'","_")
        print(file_name)
        
        file_path = join('Sv5_b', file_name)
        try:
            
            data = pd.read_csv(file_path)
            
            
            X = data.drop(columns=['y'])
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            
            
            n_clusters = len(set(data['y']))
            
            random_numbers = []
            for i in range(15):
                min_value = 2
                max_value = n_clusters + 15
                if i == 0:
                    # The first number is the central number itself
                    random_numbers.append(n_clusters)
                else:
                    # Generate random numbers with progressively larger distance
                    distance = i * 2  # You can adjust the step size if needed
                    random_value = random.randint(n_clusters - distance, n_clusters + distance)
                    random_value = max(min_value, min(random_value, max_value))  # Ensure the value is within the desired range
                    random_numbers.append(random_value)
            
            
            sil = silhouette_score(X, list(data['y']))
            dbs = davies_bouldin_score(X, list(data['y']))
            ari = adjusted_rand_score(list(data['y']), list(data['y']))
            datasets = ([file_name]+["Ground Truth"]+[np.round(sil, decimals=2)]+[np.round(dbs, decimals=2)]+[np.round(ari, decimals=2)]+[abs(n_clusters-n_clusters)]+[n_clusters]+[n_clusters])
            pd.DataFrame(datasets).T.to_csv(output_path, mode='a', index=False, header=False)
            for rep in random_numbers:
                cluster_algo = [
                                MiniBatchKMeans(n_clusters=rep),
                                ]
                try:
                    cluster_labels = None
                    while cluster_labels is None or len(set(cluster_labels)) == 1:
                        selected_model = random.choice(cluster_algo)
                        cluster_labels = selected_model.fit_predict(X)
                    sil = silhouette_score(X, cluster_labels)
                    dbs = davies_bouldin_score(X, cluster_labels)
                    ari = adjusted_rand_score(list(data['y']), cluster_labels)    
                    datasets = ([file_name]+[type(selected_model).__name__]+[np.round(sil, decimals=2)]+[np.round(dbs, decimals=2)]+[np.round(ari, decimals=2)]+[abs(rep-n_clusters)]+[rep]+[n_clusters])    
                    pd.DataFrame(datasets).T.to_csv(output_path, mode='a', index=False, header=False)
                except Exception as e:
                    print(f"{e} for >>>> {selected_model}")
        except Exception as e:
            print(e)
            pass

metadatabase = pd.read_csv("metadatabase_surrogate_Sv5_b.csv")

metadatabase = metadatabase.drop(columns=["algorithm","sil","dbs","cluster_diff"])
metadatabase = metadatabase.drop_duplicates(subset=['Dataset']).reset_index(drop=True)

_algorithm_space_minibatch(list(metadatabase['Dataset']))