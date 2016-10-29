path <- "/home/manish/Kaggle"
setwd(path)

#load libraries and data
library(data.table)
library(caret)
library(mlr)

train <- fread("clean_train.csv",stringsAsFactors = T)
test <- fread("clean_test.csv",stringsAsFactors = T)

str(train)
dim(test)

#combine levels with minimal observations
colnames(train)[11:126]
colnames(test)[10:125]

for(i in names(train)[11:126]){
  p <- 5/100
  ld <- names(which(prop.table(table(train[[i]]))<p))
  levels(train[[i]])[levels(train[[i]]) %in% ld] <- "Any"
}

for(i in names(test)[10:125]){
  p <- 5/100
  ld <- names(which(prop.table(table(test[[i]]))<p))
  levels(test[[i]])[levels(test[[i]]) %in% ld] <- "Any"
}


#create a validation set
split <- createDataPartition(train$loss,p = 0.7,list = FALSE)

d_train <- train[split,]
d_val <- train[-split,]

d_train$loss <- log(d_train$loss)
summary(d_train$loss)

#mlr tandav
traintask <- makeRegrTask(data = d_train,target = "loss")
valtask <- makeRegrTask(data = d_val,target = "loss")

#check variable importance
var_imp <- generateFilterValuesData(traintask,method = "gain.ratio")
plotFilterValues(var_imp) #cat57,80,7 most important

#Feature Selection - No. of features to tune
getParamSet("regr.h2o.gbm")
gbm.lrn <- makeLearner(cl = "regr.h2o.gbm",predict.type = "response")
#lrner <- makeLearner(cl = "regr.randomForest",predict.type = "response")

lrn <- makeFilterWrapper(learner = gbm.lrn, fw.method = "gain.ratio")
ps <- makeParamSet(
  makeDiscreteParam("fw.abs",values = seq(1,15,1))
  # makeIntegerParam("ntree",lower = 100,upper = 1000),
  # makeIntegerParam("mtry",lower = 3,upper = 100),
  # makeIntegerParam("nodesize",lower = 50,upper = 1000),
  # makeLogicalParam("importance",default = TRUE)
)

library(parallelMap)
library(parallel)
parallelStartSocket(cpus = detectCores())

rdesc <- makeResampleDesc("CV",iters=10)
ctrl <- makeTuneControlGrid()
res <- tuneParams(learner = lrn,task = traintask,resampling = rdesc,par.set = ps,control = ctrl,measures = mae)

#Other feature selection methods to try
#add one by one variable until accuracy reduces
ctrl <- makeFeatSelControlSequential(method = "sfs",alpha = 0.02)
#or
ctrl2 <- makeFeatSelControlExhaustive(maxit = 100L)
sfeats <- selectFeatures(learner = gbm.lrn,task = traintask,resampling = rdesc,measures = mae,control = ctrl)
sfeats
analyzeFeatSelResult(sfeats)

#model multiplexer
#http://mlr-org.github.io/mlr-tutorial/devel/html/advanced_tune/index.html














