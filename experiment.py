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
VERSION = "Sv5_b"
# FILE = "metadatabase_surrogate_"+VERSION+".csv"
FILE = "minibatch_problem_space.csv"
LOG_FILE = "log_13_07_b.csv"
PROBLEM_FILE = "New_Problem_Benchmarks.csv"


par_ini = 1
par_fin = 10


# sil_ini_list = np.arange(0.2, 0.6, 0.05)
# dbs_max_list = np.arange(0.4, 0.8, 0.1)
# dist_sil_dbs_list = np.arange(0.15, 0.35, 0.05)

sil_ini_list = np.arange(0, 0.6, 0.05)
dbs_max_list = np.arange(0.4, 3, 0.2)
dist_sil_dbs_list = np.arange(0.15, 0.35, 0.05)


seed_list = range(int(par_ini),int(par_fin))
# distribution_list = [["exponential", 'gumbel', 'normal'], ["exponential", 'normal'], ['gumbel', 'normal']]

algorithms = [RandomForestRegressor]
qtd_arvores = [100,250]
threshold_qtd_on_training = 50
contamination = [0, 0.05, 0.1, 0.15]

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
  
  min_data = pd.DataFrame(x.groupby(["Dataset"])["yhat"].min().reset_index())
  result = pd.merge(x, min_data, on="yhat").drop_duplicates(subset='Dataset_x', keep="first")
  # TODO - criterio de empate
  ari = np.round(result.ari.mean(), 2)
  print("ARI", ari)
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

  print(result.groupby(['Dataset_x'])[['ari','k_candidate', 'k_expected']]
        .mean())

  log = [mae, mse, rmse, r2, acc, ari, std, type(model_regressor).__name__, seed, sil_ini, dbs_max, dist_s_d, qtd, contamination, qtd_arvores]
  log = pd.DataFrame(log).T
  log.columns = ["MAE", "MSE", "RMSE", "R2", "ACC", "ARI","STD all clusterers", "algorithm", "seed",  "sil_ini", "dbs_max", "distance_s_d", "qtd samples", "contamination", "qtd_arvores"]
  log.to_csv(f"{PATH}/{LOG_FILE}", mode='a', index=False, header=False)

  run["model_regressor"] = "RandomForest"
  run["features"] = features
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
  run["ARI"] = ari
  run["STD all clusterers"] = std
  run["Progress"] = progress
  run["Contamination"] = contamination
  run["qtd_arvores"] = qtd_arvores

all_combinations = all_combinations[:3]
for index, combination in enumerate(all_combinations):
    run = neptune.init_run(
        project="MaleLab/SV5minibatch",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODNjNDRiNS02MDM4LTQ2NGEtYWQwMC00OGRhYjcwODc0ZDIifQ==",
    )
    df_surrogate = original
    progress = (index + 1) / len(all_combinations) * 100
    print("\n\n\n")
    print(f"Progress: {progress:.2f}%")
    run['progress'] = round(progress,2)
    sil_ini, dbs_max, dist_s_d, seed, regressor, contamin, qtd_arvores = combination
    
    # df_surrogate = filter_sil_bds(df_surrogate, sil_ini, dbs_max)
    # df_surrogate = filter_dist_sil_bds(df_surrogate, dist_s_d)
    # df_surrogate = filter_samples_isolation(df_surrogate, contamin)


    if(df_surrogate.shape[0]<threshold_qtd_on_training):
        print("Invalid Combination: ",df_surrogate.shape )
        continue
    print("Samples", df_surrogate.shape)

    # df_surrogate['clusters'] = df_surrogate.apply(lambda row: int(row.Dataset.split('-')[1].replace("clusters","")), axis=1)
    
    df_surrogate = df_surrogate[features]

    data = df_surrogate
    x_train, y_train = data.values[:, :-1], data.values[:, -1]

    model_regressor = regressor(random_state=seed, n_estimators=qtd_arvores, n_jobs=-1)
    model_regressor.fit(x_train, y_train)
    
    # run["name"] = "A_"+str(round(contamin,2))+"_"+str(seed)+"_"+str(round(sil_ini,2))+"_"+str(round(dbs_max,2))+"_"+str(round(dist_s_d,2))
    run["sys/tags"].add("randomsearch")
    run = None
    minimizing_logging(model_regressor, features, sil_ini, dbs_max, dist_s_d, seed, df_surrogate.shape[0], run, progress, contamin, qtd_arvores)

    run.sync()
    run.stop()
