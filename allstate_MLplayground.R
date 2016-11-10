path <- "/home/manish/Kaggle"
setwd(path)

#load libraries and data
library(data.table)
library(caret)
library(mlr)

#use randomforest and GBM
#use glmnet and xgboost
#use extraTrees and rpart

base.learners <- list(
  makeLearner("regr.randomForest"),
  makeLearner("regr.gbm")
)

lrn <- makeModelMultiplexer(base.learners)


ps <- makeModelMultiplexerParamSet(lrn,
                      makeIntegerParam("ntree",lower = 30L,upper = 800L),
                      makeIntegerParam("mtry",lower = 5L,upper = 50L),
                      makeIntegerParam("sampsize",lower = 100L,upper = 10000L),
                      makeIntegerParam("nodesize",lower = 10L,upper = 500L),
                      makeIntegerParam("n.trees",lower = 30L,upper = 800L),
                      makeIntegerParam("interaction.depth",lower = 3L,upper = 20L),
                      makeIntegerParam("n.minobsinnode",lower = 10L,upper = 500L),
                      makeNumericParam("shrinkage",lower = 0.001,upper = 0.1)
)

rdesc <- makeResampleDesc("CV",iters=5L)
ctrl <- makeTuneControlRandom(maxit = 10L)

parallelStartSocket(cpus = detectCores())

res <- tuneParams(learner = lrn,task = traintask,resampling = rdesc,measures = mae,par.set = ps,control = ctrl,show.info = T)

parallelStop()

#Neural Network averaging in R
#https://www.kaggle.com/tobikaggle/allstate-claims-severity/h2o-dnn-averaging-in-r/discussion


#check variable importance with gain ratio
var_imp <- generateFilterValuesData(traintask,method = "gain.ratio")
plotFilterValues(var_imp,sort = "dec",feat.type.cols = T) #cat57,80,7 most important

#use only top 30 features - gives 1340 on LB
filtered.data <- filterFeatures(traintask,method = "gain.ratio",abs = 30)

#use only top 50 features
filtered.data <- filterFeatures(traintask,method = "information.gain",abs = 50)



# Tune GBM ----------------------------------------------------------------
f_gbm <- makeLearner("regr.gbm",predict.type = "response")

#set search space
set.seed(7007)
getParamSet("regr.gbm")
params <- makeParamSet(
  makeIntegerParam("n.trees",lower = 30L,upper = 800L),
  makeIntegerParam("interaction.depth",lower = 3L,upper = 20L),
  makeIntegerParam("n.minobsinnode",lower = 10L,upper = 500L),
  makeNumericParam("shrinkage",lower = 0.001,upper = 0.1)
)

rdsc <- makeResampleDesc(method = "Holdout")
ctrl <- makeTuneControlRandom(maxit = 10L)

parallelStartSocket(cpus = detectCores())
res <- tuneParams(learner = f_gbm,task = traintask,resampling = rdsc,measures = mae,par.set = params,control = ctrl,show.info = T)
parallelStop()
res$y
res$x
head(as.data.frame(res$opt.path))

#set hyper params
gbm_lrn <- setHyperPars(learner = f_gbm,par.vals = res$x)

#train model
gbmmodel <- train(gbm_lrn,traintask)

#test model
gbmpred <- predict(gbmmodel,testtask)
head(gbmpred$data)
gbmpredmain <- exp(gbmpred$data$response)-1

#submission
myfile <- data.table(id=test$id,loss=gbmpredmain)
write.csv(myfile,"first_gbm.csv",row.names = F) #1340.64
write.csv(myfile,"second_gbm.csv",row.names = F) #1148.42



# Tune Extratrees ---------------------------------------------------------

getParamSet("regr.extraTrees")
ext.lrn <- makeLearner("regr.extraTrees",predict.type = "response")

#create a task of numeric matrix
install.packages("dummies")
library(dummies)

my_train <- dummy.data.frame(data = train,dummy.classes = "factor",omit.constants = T)
my_test <- dummy.data.frame(data = test,dummy.classes = "factor",omit.constants = T)

#set integer to numeric
for(i in colnames(my_train)){
  if (class(my_train[[i]]) == "integer"){
    my_train[[i]] <- as.numeric(my_train[[i]])

  }
}

for(i in colnames(my_test)){
  if (class(my_test[[i]]) == "integer"){
    my_test[[i]] <- as.numeric(my_test[[i]])
    
  }
}

extraTrain <- makeRegrTask(data = my_train,target = "loss") 
extraTest <- makeRegrTask(data = my_test,target = "loss")

params <- makeParamSet(
  makeIntegerParam("ntree",lower = 100L,upper = 1000L),
  makeIntegerParam("mtry",lower = 5L,upper = 500L),
  makeIntegerParam("nodesize",lower = 5L,upper = 1000L),
  makeIntegerParam("numRandomCuts",lower = 1L,upper = 50L)
)

rdsc <- makeResampleDesc(method = "Holdout")
ctrl <- makeTuneControlRandom(maxit = 20L)

parallelStartSocket(cpus = detectCores())
parallelStop()

ext.tune <- tuneParams(learner = ext.lrn,
                       task = traintask,
                       resampling = rdsc,
                       measures = mae,
                       par.set = params,
                       control = ctrl,
                       show.info = T
)


y <- train$loss
train2 <- copy(train)

train2[,loss := NULL]
x <- data.matrix(train2)

tid <- test$id
test[,id := NULL]

test1 <- copy(test)
test1[,loss := NULL]

install.packages("extraTrees")
library(extraTrees)
extraTree <- extraTrees::extraTrees(x = x,y=y,ntree=1000,mtry=10,nodesize=20,numThreads=20,numRandomCuts=6,evenCuts=F)
ex.predict <- predict(extraTree,newdata = data.matrix(test1))

mypred <- exp(ex.predict)-1
mysub <- data.table(id=tid,loss=mypred)
write.csv(mysub,"third_extraTrees.csv",row.names = F) #1175.35
