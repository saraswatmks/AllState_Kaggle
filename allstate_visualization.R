path <- "/home/manish/Kaggle"
setwd(path)
source("t1.R")

#data is loaded from t1.R

#train data
dim(train)
dim(test)

#exploratory data analysis with target variable
#let's plot a variable with target
library(ggplot2)

ggplot(train,aes(loss))+geom_density()
ggplot(train,aes(cont1))+geom_line(stat="density",color="red")

line_plot <- function(a) {
  ggplot(train,aes(x = a,y=..density..))+
    geom_histogram(bins = 50,fill="navyblue")+
    geom_density(color="red",fill="grey",alpha=0.2)
}

line_plot(normalize(train$cont5,method = "standardize"))
line_plot(train$cont8)
#cont 5 is skewed
#cont 8 is skewed


#density plot on a histogram
ggplot(train,aes(x=cont1,y=..density..))+
  geom_histogram(bins = 100,fill="black")+
  geom_density(color="red")+
  scale_x_continuous(breaks = seq(0,1,0.05))

#density plot with histogramwith categorical variables
ggplot(train,aes(x=loss,fill=cat112))+geom_density(alpha=0.1)

#density plot with histograms separated by categorical variables
ggplot(train,aes(x=loss,y=..density..))+
  geom_histogram(bins = 100,fill="navyblue")+
  geom_density(color="red")+
  facet_grid(cat10~.)

#bar plot for categorical variables
dare1 <- function(x){
ggplot(train,aes(x))+geom_bar(fill="lightblue",color="black")+
  geom_text(stat='count',aes(label=..count..),vjust=-0.2)
}

dare2 <- function(x){
  ggplot(test,aes(x))+geom_bar(fill="lightblue",color="black")+
    geom_text(stat='count',aes(label=..count..),vjust=-0.2)
}

dare1(train$cat15)
dare2(test$cat15)

dare1(train$cat16)
dare2(test$cat16)

dare1(train$cat30)
dare2(train$cat30)

dare1(train$cat90)
dare2(test$cat90)

dare1(train$cat91)
dare2(test$cat91)

var22 <- train[,.(cat15,cat16,loss)]

doodo <- makeRegrTask(data = train,target = "loss")
getTaskFeatureNames(doodo)

doodo <- removeConstantFeatures(obj = doodo,show.info = T,perc = 0.01)
getTaskFeatureNames(doodo)

#with 5% threshold
# Removing 54 columns: cat7,cat14,
# cat15,cat16,cat17,cat18,cat19,cat20,
# cat21,cat22,cat24,cat28,cat29,cat30,
# cat31,cat32,cat33,cat34,cat35,cat39,
# cat40,cat41,cat42,cat43,cat45,cat46,
# cat47,cat48,cat49,cat51,cat52,cat54,
# cat55,cat56,cat57,cat58,cat59,cat60,
# cat61,cat62,cat63,cat64,cat65,cat66,
# cat67,cat68,cat69,cat70,cat74,cat76,
# cat77,cat78,cat85,cat89

#with 1% threshold
#Removing 31 columns: cat15,cat17,cat18,
# cat19,cat20,cat21,cat22,cat32,cat33,cat34,
# cat35,cat42,cat46,cat47,cat48,cat51,cat55,cat56,
# cat58,cat59,cat60,cat61,cat62,cat63,cat64,cat67,
# cat68,cat69,cat70,cat77,cat78


#create a separate data from these 
#near zero variance factors and extract one feature

zero_var <- train[,c(paste0("cat",c(7,14:22,24,28:35,39:43,45:49,51,52,54:70,74,76:78,85,89))),with=F]
zero_var$loss <- train$loss

sapply(zero_var,levels)

dare1(zero_var$cat77)
dare1(zero_var$cat85)

dare1(train$cat107)

a <- train[,median(loss),by=cat107]
b <- test[,median(loss),by=cat107]
colnames(b)[1] <- "name"

cd <- data.table(name = a$cat107,train107 = a$V1)
cd <- b[cd,on="name"]
cd[,diff := V1 - train107]

zero_var[,median(loss),by=cat89]
zero_var[,median(loss),by=cat85]
zero_var[,median(loss),by=cat77]

#write test file
sol <- fread("second_gbm.csv")
test$loss <- sol$loss

test[,median(loss),by=cat85]
