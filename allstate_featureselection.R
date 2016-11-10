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


#check unique levels per column
sapply(train[11:126],unique)

for(i in names(train)[83:126]){
  p <- 5/100
  ld <- names(which(prop.table(table(train[[i]]))<p))
  levels(train[[i]])[levels(train[[i]]) %in% ld] <- "Any"
}

for(i in names(test)[82:125]){
  p <- 5/100
  ld <- names(which(prop.table(table(test[[i]]))<p))
  levels(test[[i]])[levels(test[[i]]) %in% ld] <- "Any"
}

#create a validation set
train[,loss := log(loss + 1)]
summary(train$loss)

ggplot(train,aes(loss))+geom_density(color="red")

#split <- createDataPartition(train$loss,p = 0.7,list = FALSE)

#d_train <- train[split,]
#d_val <- train[-split,]

#mlr tandav
library(parallelMap)
library(parallel)
parallelStartSocket(cpus = detectCores())
parallelStop()

traintask <- makeRegrTask(data = train,target = "loss")
#valtask <- makeRegrTask(data = d_val,target = "loss")
test[,loss := 0L]
testtask <- makeRegrTask(data = test,target = "loss")

traintask <- removeConstantFeatures(obj = traintask)
testtask <- removeConstantFeatures(obj = testtask)

all(getTaskFeatureNames(traintask) %in% getTaskFeatureNames(testtask))

#check variable importance with gain ratio
var_imp <- generateFilterValuesData(traintask,method = "gain.ratio")
plotFilterValues(var_imp,sort = "dec",feat.type.cols = T,n.show = 50) #cat57,80,7 most important

#Feature Selection
listLearners(obj = "regr",check.packages = T)[c("class","package")]

#Decision Tree
getParamSet("regr.rpart")
tree.lrn <- makeLearner(cl = "regr.rpart",predict.type = "response")

lrn <- makeFilterWrapper(learner = tree.lrn, fw.method = "gain.ratio",fw.abs = 20)

#check cross validation score
rdesc <- makeResampleDesc("CV",iters=10)
r <- resample(learner = lrn,task = traintask,resampling = rdesc,measures = mae,models = T)
r$aggr #1505 MAE or 0.535 with 20 features

sfeats <- sapply(r$models,getFilteredFeatures)
table(sfeats)

#Tune the size of feature subset
#use previous learner tree.lrn
lrn <- makeFilterWrapper(learner = tree.lrn,fw.method = "information.gain")
ps <- makeParamSet(
  makeDiscreteParam("fw.abs",values = seq(5,50,1))
)
rdesc <- makeResampleDesc("CV",iters=5)
ctrl <- makeTuneControlRandom(maxit = 50)
res <- tuneParams(learner = lrn,task = traintask,resampling = rdesc,par.set = ps,control = ctrl,measures = mae)

#Other feature selection methods to try
#add one by one variable until accuracy reduces
ctrl <- makeFeatSelControlSequential(method = "sfs",alpha = 0.02)
#or
ctrl2 <- makeFeatSelControlExhaustive(maxit = 100L)
sfeats <- selectFeatures(learner = gbm.lrn,task = traintask,resampling = rdesc,measures = mae,control = ctrl)
sfeats
analyzeFeatSelResult(sfeats)
