path <- "/home/manish/Kaggle"
setwd(path)

#correlated variables already removed
train <- fread("clean_train.csv",stringsAsFactors = T)
test <- fread("clean_test.csv",stringsAsFactors = T)

#taking log on skewed variable
train$loss <- log(train$loss+50)

#load data in MLR
test$loss <- sample(1:100,size = nrow(test),replace = T)
traintask <- makeRegrTask(data = train,target = "loss")
testtask <- makeRegrTask(data = test,target = "loss")

#remove variables with 1% threshold - removed 31 variables
traintask <- removeConstantFeatures(obj = traintask,perc = 0.01,show.info = T)

#normalize cont5, cont8
traintask <- normalizeFeatures(obj = traintask,method = "standardize",cols = c("cont5","cont8"))
testtask <- normalizeFeatures(obj = testtask,method = "standardize",cols = c("cont5","cont8"))

#check variable importance
feature_imp <- generateFilterValuesData(task = traintask,method = c("information.gain","chi.squared"))
plotFilterValues(feature_imp,feat.type.cols = T)

#check cross validation results
listLearners(obj = "regr",check.packages = T)[c("class","package")]

#ridge
#lasso
#xgboost
#nnet
#glmboost

#CV1
rdesc <- makeResampleDesc("CV",iters=5)
r <- resample(learner = "regr.xgboost",task = traintask,resampling = rdesc,measures = mae,show.info = T)
r$aggr #5.030

r <- resample(learner = "regr.rpart",task = traintask,resampling = rdesc,measures = mae,show.info = T)
r$aggr #0.521

r <- resample(learner = "regr.ctree",task = traintask,resampling = rdesc,measures = mae,show.info = T)
r$aggr #0.456

#try xgboost and try to achieve score below 1118
#lets compare scores with converting categories into integers
names <- colnames(train)[11:ncol(train)]
train[,(names) := lapply(.SD,as.integer),.SDcols=names]
sapply(train,class)

#use xgboost
install.packages("xgboost", repos=c("http://dmlc.ml/drat/", getOption("repos")), type="source")
library(xgboost)
install.packages("Metrics")
library(Metrics)

x <- createDataPartition(y = train$loss,p = 0.70,list = F)
dtrain <- train[x]
dval <- train[-x]

dtrain <- xgb.DMatrix(data = as.matrix(dtrain[,-c("loss"),with=F]),label=as.matrix(dtrain[,.(loss)]))
dval <- xgb.DMatrix(data = as.matrix(dval[,-c("loss"),with=F]),label=as.matrix(dval[,.(loss)]))
dtest <- xgb.DMatrix(data = as.matrix(test[,-c("loss","id"),with=F]))

watchlist <- list(val=dval,train=dtrain)

xgb_params <- list(
  seed = 1101,
  colsample_bytree = 0.6,
  subsample=0.6,
  eta=0.1,
  objective="reg:linear",
  max_depth=6,
  num_parallel_tree=1,
  min_child_weight=10
)

xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

xgcv <- xgb.cv(params = xgb_params,
               data = dtrain,
               nrounds = 750,
               nfold = 6,
#               early_stopping_rounds = 25,
               print_every_n = 10,
               verbose = 1,
               feval = xg_eval_mae,
               maximize = F)
#to check best round
min.error.nround <- which.min(xgcv$evaluation_log$test_error_mean)
nrounds <- min.error.nround #320
xgcv$evaluation_log$test_error_mean[320] #it gives 1149.57


bst1 <- xgb.train(params = xgb_params,
                  data = dtrain,
                  nrounds = nrounds, #set num_rounds from cv
                  watchlist = watchlist,
                  feval = xg_eval_mae,
                  print_every_n = 10,
                  verbose = 1,
                  early_stopping_rounds = 10,
                  maximize = F)
bst.pred <- predict(bst1,newdata = as.matrix(test[,-c("loss","id"),with=F]))
