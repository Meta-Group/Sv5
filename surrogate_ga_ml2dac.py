import random
from deap import creator, base, tools
from deap import algorithms
import deap

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
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# paths
PATH = os.getcwd()
VERSION = "Sv5_minibatches"
FILE = "metadatabase_surrogate_"+VERSION+".csv"
LOG_FILE = "log_ml2dac_ga.csv"
PROBLEM_FILE = "validation_ml2dac.csv"
SEED = 1

algorithms = [RandomForestRegressor]
qtd_arvores = [100]
threshold_qtd_on_training = 120

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
















def filter_sil_bds(database_input, min_sil, max_sil, min_dbs, max_dbs):
  database_input = database_input[database_input.sil>=min_sil]
  database_input = database_input[database_input.sil<=max_sil]
  database_input = database_input[database_input.dbs>=min_dbs]
  database_input = database_input[database_input.dbs<=max_dbs]
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

def minimizing_logging(model_regressor, sil_min, sil_max, dbs_min, dbs_max, seed, qtd, run, contamin, qtd_arvores):
  datasets = run_exp(model_regressor)
  x = pd.DataFrame(datasets)
  x.columns = ["Dataset", "Algorithm", "sil", "dbs", "ari", "k_candidate", "k_expected", "yhat"]
  
  min_data = pd.DataFrame(x.groupby(["Dataset"])["yhat"].min().reset_index())
  result = pd.merge(x, min_data, on="yhat").drop_duplicates(subset='Dataset_x', keep="first")


  print(" ")
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

  print("QTD samples",qtd)

  contamination = np.round(contamin,2)

  #print(result.groupby(['Dataset_x'])[['ari','k_candidate', 'k_expected']]
  #      .mean())

  log = [mae, mse, rmse, r2, acc, ari, std, type(model_regressor).__name__, seed, sil_min, sil_max, dbs_min, dbs_max, qtd, contamination, qtd_arvores]
  log = pd.DataFrame(log).T
  log.columns = ["MAE", "MSE", "RMSE", "R2", "ACC", "ARI","STD all clusterers", "algorithm", "seed",  "sil_min", "sil_max", "dbs_min", "dbs_max", "qtd samples", "contamination", "qtd_arvores"]
  log.to_csv(f"{PATH}/{LOG_FILE}", mode='a', index=False, header=False)

  return ari 

def mutate(individual):
    gene = random.randint(0,9) #select which parameter to mutate
    if gene == 0:
         individual[gene] = random.uniform(sil_min, sil_max)
    elif gene == 1:
         individual[gene] = random.uniform(sil_min, sil_max)
    elif gene == 2:
         individual[gene] = random.uniform(dbs_min, dbs_max)
    elif gene == 3:
         individual[gene] = random.uniform(dbs_min, dbs_max)
    elif gene == 4:
         individual[gene] = random.uniform(contamination_min, contamination_max)
    return individual,


def evaluate(individual):
    sil_min = individual[0]
    sil_max = individual[1]
    dbs_min = individual[2]
    dbs_max = individual[3]
    contamination = individual[4]


    df_surrogate = original
    df_surrogate = filter_sil_bds(df_surrogate, sil_min, sil_max, dbs_min, dbs_max)

    x = 0
    if(df_surrogate.shape[0]>threshold_qtd_on_training):
        df_surrogate = filter_samples_isolation(df_surrogate, individual[4])

        df_surrogate = df_surrogate[features]

        data = df_surrogate
        x_train, y_train = data.values[:, :-1], data.values[:, -1]

        model_regressor = RandomForestRegressor(random_state=SEED, n_estimators=100, n_jobs=-1)
        model_regressor.fit(x_train, y_train)
        x = minimizing_logging(model_regressor, sil_min, sil_max, dbs_min, dbs_max, get_SEED(), df_surrogate.shape[0], run,  contamination, qtd_arvores)
    return x,


toolbox = base.Toolbox()

#GA DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximise the fitness function value
creator.create("Individual", list, fitness=creator.FitnessMax)

# Possible parameter values
sil_max = 1.0
sil_min = 0.0
dbs_max = 3.0
dbs_min = 0.0
contamination_min = 0
contamination_max = 0.3 

toolbox.register("attr_sil_max", random.uniform, sil_min, sil_max)
toolbox.register("attr_sil_min", random.uniform, sil_min, sil_max)
toolbox.register("attr_dbs_max", random.uniform, dbs_min, dbs_max)
toolbox.register("attr_dbs_min", random.uniform, dbs_min, dbs_max)
toolbox.register("attr_contamination", random.uniform, contamination_min, contamination_max)

# This is the order in which genes will be combined to create a chromosome
N_CYCLES = 1
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_sil_min, toolbox.attr_sil_max,
                  toolbox.attr_dbs_min, toolbox.attr_dbs_max,
                  toolbox.attr_contamination), n=N_CYCLES)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)



toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate",mutate)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("evaluate", evaluate)

population_size = 150
crossover_probability = 0.7
mutation_probability = 0.1
number_of_generations = 30

pop = toolbox.population(n=population_size)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

for SEED in np.arange(1,20):

    def get_SEED():
        return SEED
 
    print(">>>>>>>>>>>>>>>>>>",get_SEED())
    run = neptune.init_run(
    project="MaleLab/GASv5ML2DAC",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2YTE5Zjg5NC1mMjk0LTRlN2UtYjgxMC03OTE1ZWJiYjliNTQifQ==",
    )    
 
    #run["name"] = "GA_seed"+str(SEED)+"_sil_"+str(round(sil_min,2))+"_"+str(sil_max)+"_dbs_"+str(round(dbs_min,2))+"_"+str(round(dbs_max,2))+"_pop_"+str(round(population_size,2)+"_gen_"+str(round(number_of_generations)))
    run["sys/tags"].add("randomsearch")

    pop, log = deap.algorithms.eaSimple(pop, toolbox, cxpb=crossover_probability, stats = stats, 
                                mutpb = mutation_probability, ngen=number_of_generations, halloffame=hof, 
                                verbose=True) 

    best_parameters = hof[0] # save the optimal set of parameters

    print(best_parameters)

    run["qtd_samples"] = best_parameters[1]
    run["attr_sil_max"] = best_parameters[1]
    run["attr_sil_min"] = best_parameters[0]
    run["attr_dbs_max"] = best_parameters[3]
    run["attr_dbs_min"] = best_parameters[1]
    run["attr_contamination"] = best_parameters[4]
    run["seed"] = SEED
    run["population_size"] = population_size
    run["crossover_probability"] = crossover_probability
    run["mutation_probability"] = mutation_probability
    run["number_of_generations"] = number_of_generations
    run["ARI"] = max(log.select("max"))

 
    gen = log.select("gen")
    max_ = log.select("max")
    avg = log.select("avg")
    min_ = log.select("min")

    evolution = pd.DataFrame({'Generation': gen,
                            'Max X': max_,
                            'Average':avg,
                            'Min X': min_})

    
    plt.title('Parameter Optimisation')
    plt.plot(evolution['Generation'], evolution['Min X'], 'b', color = 'C1', label = 'Min')
    plt.plot(evolution['Generation'], evolution['Average'], 'b', color = 'C2', label = 'Average')
    plt.plot(evolution['Generation'], evolution['Max X'], 'b', color = 'C3', label= 'Max')

    plt.legend(loc = 'lower right')
    plt.ylabel('ARI (min)')
    plt.xlabel('Generation')
    plt.xticks([0,5,10,15,20])

    plt.savefig(PATH+'/parameter_optimization_plot.png') 
    run["GA_CHART"].upload(PATH+'/parameter_optimization_plot.png')
    plt.close()

    run.sync()
    run.stop()