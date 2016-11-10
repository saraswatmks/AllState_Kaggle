path <- "C:/Users/manish/Desktop/Data/Amonth_Wise/October 2016/Kaggle"
setwd(path)

#load libraries and data
library(data.table)
library(caret)
library(mlr)

train <- fread("train.csv",stringsAsFactors = T)
test <- fread("test.csv",stringsAsFactors = T)

dim(train)
dim(test)

str(train)
train[1:3]
train[1:50,.(loss)]

#check column classes
train[,lapply(.SD, class)]

#split categorical and continuous variables
x <- split(names(train), sapply(train, function(x)class(x)))

X_num <- train[,x$numeric,with=FALSE]
X_cat <- train[,x$factor,with=FALSE]

y <- split(names(test), sapply(test, function(x) class(x)))

Y_num <- test[,y$numeric,with=FALSE]
Y_cat <- test[,y$factor,with=FALSE]

#correlation
X_mynum <- X_num[,-c("loss"),with=FALSE] 
d <- cor(as.matrix(X_mynum))
de <- findCorrelation(x = d,cutoff = 0.7,names = T)
de

e <- cor(as.matrix(Y_num))
ef <- findCorrelation(x = e, cutoff = 0.7,names = T)
ef

#remove correlated columns
X_num <- X_num[,-de,with=FALSE]
Y_num <- Y_num[,-ef, with=FALSE]

#create df1
a1 <- summarizeColumns(X_cat)[c("name","nlevs")]
a2 <- summarizeColumns(Y_cat)[c("name","nlevs")]

colnames(a1)[[2]] <- "train_lev"
colnames(a2)[[2]] <- "test_lev"

a2 <- setDT(a2)[a1,on="name"]
rm(a1)

a2$train_lev_ac <- lapply(X_cat,levels)
a2$test_lev_ac <- sapply(Y_cat,levels)

#till cat88 all levels OK
a3 <- a2[89:116]
rm(a2,a3)
#set levels as equal
#convert factors to characters and subset
cnames <- names(X_cat)
X_cat[,(cnames) := lapply(.SD,as.character),.SDcols = cnames]
Y_cat[,(cnames) := lapply(.SD,as.character),.SDcols = cnames]


#combine train data subsets
X_cat <- cbind(X_num,X_cat)


# X_cat <- subset(X_cat, !(cat89 %in% c("I")))
# a3 <- X_cat[,89:90,with=FALSE]
X_cat <- X_cat[!(X_cat$cat89 %in% c("I"))]
X_cat <- X_cat[!(X_cat$cat90 %in% c("G"))]
X_cat <- X_cat[!(X_cat$cat92 %in% c("F"))]
X_cat <- X_cat[!(X_cat$cat101 %in% c("N","U"))]
X_cat <- X_cat[!(X_cat$cat102 %in% c("H","J"))]
X_cat <- X_cat[!(X_cat$cat105 %in% c("R","S"))]
X_cat <- X_cat[!(X_cat$cat109 %in% c("AG","AK","B","BF","BM","BP","BT","BV","BY","CJ","J"))]                                 
X_cat <- X_cat[!(X_cat$cat110 %in% c("AF","AN","BD","BI","BK","CN","CB","DV","EH","EI","H"))]                                 
X_cat <- X_cat[!(X_cat$cat111 %in% c("D"))]                                 
X_cat <- X_cat[!(X_cat$cat113 %in% c("AC","BE","T"))]                                 
X_cat <- X_cat[!(X_cat$cat114 %in% c("X"))]                                 
X_cat <- X_cat[!(X_cat$cat116 %in% c("AB","AH","AM","AP","AS","AT","AY","BF","BI","BL",
                                     "C","CC","DQ","DY","EQ","EV","FN","FO","FS","GQ","HO",
                                     "HU","IB","IK","IO","IX","JD","JI","JN","JO","JT",
                                     "MB","MF","MT","P","V","W","X"))]                                 

#combine levels with less than 4% observations
#convert character to factors
cnames <- names(X_cat)[11:126]
for(i in cnames) set(X_cat,j=i,value = factor(X_cat[[i]]))


for(i in names(X_cat)[11:126]){
        p <- 4/100
        ld <- names(which(prop.table(table(X_cat[[i]])) < p))
        levels(X_cat[[i]])[levels(X_cat[[i]]) %in% ld] <- "Other"
}

cnames <- names(Y_cat)
for(i in cnames) set(Y_cat,j=i,value = factor(Y_cat[[i]]))

for(i in names(Y_cat)){
        p <- 4/100
        ld <- names(which(prop.table(table(Y_cat[[i]])) < p))
        levels(Y_cat[[i]])[levels(Y_cat[[i]]) %in% ld] <- "Other"
}

table(Y_cat$cat87)
table(X_cat$cat87)

str(Y_cat)

# Y_cat <- subset(Y_cat, !(cat89 %in% c("F")&
#                                  cat92 %in% c("E","G")&
#                                  cat96 %in% c("H")&
#                                  cat99 %in% c("U")&
#                                  cat103 %in% c("N")&
#                                  cat106 %in% c("R")&
#                                  cat109 %in% c("AD")&
#                                  cat110 %in% c("BH","CA","EN")&
#                                  cat111 %in% c("D")&
#                                  cat113 %in% c("AA","R")&
#                                  cat116 %in% c("A","AI","AQ","BE","BH","BJ","BN","BR",
#                                                "DB","EM","ER","ET","EX","FY","HS","IS",
#                                                "IW","JS","KO","LP","LS","MX","N")
#                          ))


#combine data sets
d_train <- copy(X_cat)
d_test <- cbind(Y_num,Y_cat)

d_test[,id := test$id]

#check id similarity in train and test
setkey(d_train,id)
setkey(d_test,id)
d_train[d_test,.N,by=.EACHI] #no test ID in train

table(d_test$id %in% d_train$id) #also verified by this step

#write clean data
write.csv(d_train,"clean_train.csv",row.names = F)
write.csv(d_test,"clean_test.csv",row.names = F)

