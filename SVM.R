##redict()是训练函数，plot()可视化数据，
##支持向量，决策边界(如果提供的话)。参数调整tune()。

install.packages("e1071")
library("e1071")


##自带数据实验
data(iris)
ir <- iris
set.seed(124)
count.test <- round(runif(50,1,150))
test <- ir[count.test,]
sv <- svm(Species~.,data=ir,cross=5,type='C-classification',kernel='sigmoid')
summary(sv)
pre <- predict(sv,test)
pre

dim(test[test$Species!=pre,])[1]/dim(test)[1]


##工程實例

#讓我們看一下如何使用支持向量機實現二元分類器，

#使用的數據是來自mass包的cats數據集。
#在本例中你將嘗試使用體重和心臟重量來預測一隻貓的性別。
#我們拿數據集中20%的數據點，用於測試模型的準確性（在其餘的80%的數據上建立模型）
#原文網址：https://read01.com/6B66xM.html
data(cats,package = "MASS")

# linear SVM
# linear svm, scaling turned OFF
inputData <- data.frame(cats[, c (2,3)], 
                        response = as.factor(cats$Sex)) # response as factor
svmfit <- svm(response ~ ., data = inputData, 
              kernel = "linear", cost = 10, scale = FALSE) 
print(svmfit)
plot(svmfit,inputData)
help("plot")
compareTable <- table (inputData$response, predict(svmfit))  # tabulate

# radial SVM

svmrad <- svm(response ~ ., data = inputData, kernel = "radial", 
              cost = 10, scale = FALSE) # radial svm, scaling turned OFF
print(svmrad)
plot(svmrad,inputData)
compareTable <- table (inputData$response, predict(svmrad))  # tabulate
mean(inputData$response != predict(svmrad))

##寻找最优参数，使用tune.svm()函数，来寻找svm()函数的最优参数

### Tuning
# Prepare training and test data
set.seed(100) # for reproducing results
rowIndices <- 1 : nrow(inputData)# prepare row indices
sampleSize <- 0.8 * length(rowIndices) # training sample size
trainingRows <- sample (rowIndices, sampleSize) # random sampling
trainingData <- inputData[trainingRows, ] # training data
testData <- inputData[-trainingRows, ] # test data
tuned <- tune.svm(response ~., data = trainingData, 
                  gamma = 10^(-6:-1), cost = 10^(1:2)) # tune
summary (tuned) # to select best gamma and cost


###test

install.packages("RSNNS")
library(Rcpp)
library(RSNNS)

data(iris)
iris = iris[sample(1:nrow(iris),length(1:nrow(iris))),1:ncol(iris)]
irisValues= iris[,1:4]
iris






