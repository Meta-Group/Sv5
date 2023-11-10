# imports
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import itertools
from sklearn.ensemble import IsolationForest
import neptune
import sklearn.metrics as metrics
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn_extra.cluster import KMedoids

from os import listdir
from os.path import isfile, join
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# paths
PATH = os.getcwd()
VERSION = "Sv5_minibatches"
FILE = "metadatabase_surrogate_"+VERSION+".csv"
LOG_FILE = "log_ml2dac.csv"
PROBLEM_FILE = "validation_ml2dac.csv"


par_ini = 1
par_fin = 24


# sil_ini_list = np.arange(0.2, 0.6, 0.05)
# dbs_max_list = np.arange(0.4, 0.8, 0.1)
# dist_sil_dbs_list = np.arange(0.15, 0.35, 0.05)

sil_ini_list = np.arange(0, 0.6, 0.1)
dbs_max_list = np.arange(0.5, 3, 0.5)
dist_sil_dbs_list = np.arange(0.5, 3, 0.5)


seed_list = range(int(par_ini),int(par_fin))
# distribution_list = [["exponential", 'gumbel', 'normal'], ["exponential", 'normal'], ['gumbel', 'normal']]

algorithms = [RandomForestRegressor]
qtd_arvores = [100]
threshold_qtd_on_training = 120
contamination = [0, 0.1, 0.2]

features_4bench = ['attr_conc.mean','attr_conc.sd','attr_ent.mean',
    'attr_ent.sd','attr_to_inst','cohesiveness.mean','cohesiveness.sd',
    #'cor.mean',#'cor.sd',
    'cov.mean',#'cov.sd',
    'eigenvalues.mean','eigenvalues.sd',
    'inst_to_attr','iq_range.mean','iq_range.sd',
    #'kurtosis.mean','kurtosis.sd',
    'mad.mean','mad.sd',
    #'max.mean','max.sd','mean.mean','mean.sd',
    'median.mean',
    'median.sd',#'min.mean','min.sd',
    'nr_attr','nr_cor_attr','nr_inst','one_itemset.mean',
    'one_itemset.sd',
    #'range.mean','range.sd',
    'sd.mean','sd.sd'
    #,'skewness.mean', 'skewness.sd'
    ,'sparsity.mean','sparsity.sd','t2','t3','t4','t_mean.mean',
    't_mean.sd','two_itemset.mean','two_itemset.sd','var.mean','var.sd',
    'wg_dist.mean','wg_dist.sd',
    'sil', 'dbs', 'predicted_cluster', 'cluster_diff'
]
datasets_selected_benchmarking = os.listdir(join(PATH,"benchmarks_dataset"))

features = features_4bench
original = pd.read_csv(f"{PATH}/{FILE}")
combined_lists = [sil_ini_list, dbs_max_list, dist_sil_dbs_list, seed_list, algorithms, contamination, qtd_arvores]

all_combinations = list(itertools.product(*combined_lists))

def filter_sil_bds(database_input, min_sil, max_dbs):
  database_input = database_input[database_input.sil>=min_sil]
  database_input = database_input[database_input.dbs<=max_dbs]
  return database_input

def filter_dist_sil_bds(database_input, dist):
  database_input = database_input[(database_input.dbs-database_input.sil) <= dist]
  return database_input

def filter_samples_isolation(database_input, contamination):
    if contamination>0:
        iso = IsolationForest(contamination=contamination, random_state=39)
        yhat = iso.fit_predict(database_input._get_numeric_data())

        mask = yhat != -1
        database_input = database_input.iloc[mask, :]
    return database_input

def run_exp(model_input):
  # remove 'cluster_diff'
  meta_features_names = features_4bench[:-1]
  validation_set = pd.read_csv(PROBLEM_FILE, header=0)
  
  filtered_validation_set = validation_set[meta_features_names]
  filtered_validation_set = filtered_validation_set.dropna()
  
  yhat = model_input.predict(filtered_validation_set)
  
  validation_set['yhat'] = yhat
  
  return validation_set.filter(["file_name", "algorithm", "sil", "dbs", "ari", "predicted_cluster", "clusters", "yhat"])

def minimizing_logging(model_regressor, features, sil_ini, dbs_max, dist_s_d, seed, qtd, run, progress, contamin, qtd_arvores):
  datasets = run_exp(model_regressor)
  x = pd.DataFrame(datasets)
  
  x.columns = ["Dataset", "Algorithm", "sil", "dbs", "ari", "k_candidate", "k_expected", "yhat"]
  x.to_csv("x.csv", index=None)

  min_data = pd.DataFrame(x.groupby(["Dataset"])["yhat"].min().reset_index())
  min_data.to_csv("min_data.csv", index=None)
  
  # TODO - criterio de empate
  # result = pd.merge(x, min_data, on="yhat").drop_duplicates(subset='Dataset_x', keep="first")
  result = pd.merge(x, min_data, on=["Dataset","yhat"])
  # result.to_csv("result.csv", index=None)

  grouped_stats = result.groupby("Dataset")["ari"].agg(['max','min','mean','median','std','count'])
  
  ari_max = np.round(grouped_stats['max'].mean(), 2)
  print("ARI_max",ari_max)
  ari_min = np.round(grouped_stats['min'].mean(), 2)
  print("ARI_min",ari_min)
  ari_mean = np.round(grouped_stats['mean'].mean(), 2)
  print("ARI_mean",ari_mean)
  ari_median = np.round(grouped_stats['median'].mean(), 2)
  print("ARI_median",ari_median)
  ari_count = np.round(grouped_stats['count'].mean(), 2)
  print("ARI_count",ari_count)
  ari_std = np.round(grouped_stats['std'].dropna().mean(), 2)
  print("ARI_std",ari_std)
  
  mae = np.round(metrics.mean_absolute_error(result.k_candidate, result.k_expected),2)
  print("MAE",mae)
  mse = np.round(metrics.mean_squared_error(result.k_candidate, result.k_expected),2)
  print("MSE",mse)
  rmse = np.round(np.sqrt(mse),2) # or mse**(0.5)
  print("RMSE",rmse)
  r2 = np.round(metrics.r2_score(result.k_candidate,result.k_expected),2)
  print("R2",r2)
  acc = np.round(metrics.accuracy_score(result.k_candidate,result.k_expected),2)
  print("ACC",acc)

  std =  np.round(result.yhat.std(),2)
  print("Std Deviation", std)

  contamination = np.round(contamin,2)

  print(result.groupby(['Dataset'])[['ari','k_candidate', 'k_expected']]
        .mean())

  log = [mae, mse, rmse, r2, acc, ari_max, ari_min, ari_mean, ari_median, ari_count, ari_std, std, type(model_regressor).__name__, seed, sil_ini, dbs_max, dist_s_d, qtd, contamination, qtd_arvores]
  log = pd.DataFrame(log).T
  log.columns = ["MAE", "MSE", "RMSE", "R2", "ACC", "ARI_max","ARI_min","ARI_mean","ARI_median","ARI_count","ARI_std","STD all clusterers", "algorithm", "seed",  "sil_ini", "dbs_max", "distance_s_d", "qtd samples", "contamination", "qtd_arvores"]
  log.to_csv(f"{PATH}/{LOG_FILE}", mode='a', index=False, header=False)

  run["model_regressor"] = "RandomForest"
  run["sil_ini"] = sil_ini  # time cost
  run["dbs_max"] = dbs_max  # parsimonly?
  run["dist_s_d"] = dist_s_d
  run["seed"] = seed
  run["Qtd Samples"] = qtd
  run["MAE"] = mae
  run["MSE"] = mse
  run["RMSE"] = rmse
  run["R2"] = r2
  run["ACC"] = acc
  run["ARI_max"] = ari_max
  run["ARI_min"] = ari_min
  run["ARI_mean"] = ari_mean
  run["ARI_median"] = ari_median
  run["ARI_count"] = ari_count
  run["ARI_std"] = ari_std
  run["STD all clusterers"] = std
  run["Progress"] = progress
  run["Contamination"] = contamination
  run["qtd_arvores"] = qtd_arvores

for index, combination in enumerate(all_combinations):
    
    df_surrogate = original
    progress = (index + 1) / len(all_combinations) * 100
    print("\n\n\n")
    print(f"Progress: {progress:.2f}%")
    print(f"Total runs {len(all_combinations)}")
    sil_ini, dbs_max, dist_s_d, seed, regressor, contamin, qtd_arvores = combination
    
    df_surrogate = filter_sil_bds(df_surrogate, sil_ini, dbs_max)
    df_surrogate = filter_dist_sil_bds(df_surrogate, dist_s_d)


    if(df_surrogate.shape[0]>threshold_qtd_on_training):
        df_surrogate = filter_samples_isolation(df_surrogate, contamin)

        df_surrogate = df_surrogate[features]

        data = df_surrogate
        x_train, y_train = data.values[:, :-1], data.values[:, -1]

        model_regressor = regressor(random_state=seed, n_estimators=qtd_arvores, n_jobs=-1)
        model_regressor.fit(x_train, y_train)
        
        run = neptune.init_run(
          project="MaleLab/SurrogateSelectionML2DAC",
          api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODNjNDRiNS02MDM4LTQ2NGEtYWQwMC00OGRhYjcwODc0ZDIifQ==",
        )    
        
        run['progress'] = round(progress,2)
        run["sys/tags"].add("randomsearch")
        # run = None
        minimizing_logging(model_regressor, features, sil_ini, dbs_max, dist_s_d, seed, df_surrogate.shape[0], run, progress, contamin, qtd_arvores)

        run.sync()
        run.stop()
