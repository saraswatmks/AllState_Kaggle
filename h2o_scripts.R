path <- "/home/manish/Kaggle"
setwd(path)

#load libraries and data
library(data.table)
library(caret)
library(mlr)

library(h2o)
localH2o <- h2o.init(nthreads = -1)

#load data in h2o
trainh2o <-as.h2o(train)
testh2o <- as.h2o(test)

y <- "loss"
x <- setdiff(colnames(train),y)


#Grid Search
#Hyperparameter Tuning
nfolds <- 5
search_criteria <- list(strategy = "RandomDiscrete",max_models=50)

#GBM Hyper
learn_rate_opt <- seq(0,1,0.01)
max_depth_opt <- seq(1,20,1)
sample_rate_opt <- seq(0.1,0.9,0.1)
col_sample_rate_opt <- seq(0.1,0.9,0.1)
ntrees_opt <- seq(10,1000,50)

#set hyperparams
hyper_params <- list(
  learn_rate = learn_rate_opt,
  max_depth = max_depth_opt,
  sample_rate = sample_rate_opt,
  col_sample_rate = col_sample_rate_opt,
  ntrees = ntrees_opt
)

#started at 10:10
system.time(
gbm_grid <- h2o.grid("gbm",x=x,y=y,
                     training_frame = train,
                     #ntrees = 400,
                     seed=7007,
                     model_id="gbm_grid21",
                     nfolds=nfolds,
                     fold_assignment="Modulo",
                     keep_cross_validation_predictions = T,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)
)

best_model <- h2o.getGrid(grid_id = "Grid_GBM_train_model_R_1478233313069_2",
                          sort_by="mae",
                          decreasing=T)
                          
#########Incomplete Script###########################################333
