install.packages("randomForest")
library(readr)
library(mlbench)  
library(ggplot2)
library(lattice)
install.packages("Boruta")
library(Boruta)
library(caret) 
library(randomForest) 
library(readr)
#wine <- read_csv("~/gitclone/R/First_R/winequality-white.csv",header = T)
wine <- read.csv("~/gitclone/R/First_R/winequality-white.csv",header = T,stringsAsFactors = F)
str(wine)
names(wine) <- gsub("_", "", names(wine))
summary(wine)
####################################### feature choose ########################################
set.seed(123)
wine_label <- as.character(wine[,12])
wine_label <- as.factor(wine_label)
wine_label <- unlist(wine_label)
wine_label <- unlist(wine[,12])
boruta.train <- Boruta(wine[,1:11],wine[,12], doTrace = 2)
print(boruta.train)
plot(boruta.train, xlab = "", xaxt = "n")

lz <- lapply(1:ncol(boruta.train$ImpHistory),
             function(i)boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])

names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),at = 1:ncol(boruta.train$ImpHistory),
     cex.axis = 0.7)
####choose the final feature
final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)
getSelectedAttributes(final.boruta, withTentative = F)
############# all 11 features were choosed

#####################################    end  ##################################################


##################################### data processing ########################################
wine$quality[wine$quality == 4] <- 5
wine$quality[wine$quality == 6] <- 5
wine$quality[wine$quality == 7] <- 8
wine$quality[wine$quality == 9] <- 8


#####################################    end  ##################################################

library(nnet)
###### get the training set and testing set 
sub <- sample(1:nrow(wine),round(nrow(wine)*2/3))
data_train <- wine[sub,]
data_test <- wine[-sub,]

Quality <- factor(data_train$quality)
Parameters <- data_train[1:11]

df <- data.frame(Parameters,Quality)

NueralNet.Model1 <- nnet(Quality ~.,data = df, size = 22, maxit = 1000, decay = 5e-4)

summary(NueralNet.Model1)

predict1 <- predict(NueralNet.Model1,data_test,type = "class")

result1 <- table(predict1,data_test$quality) 
result1

accuracy <- sum(diag(result1))/sum(result1)
accuracy
######################################## END ##################################################

install.packages('neuralnet')
library(grid)
library(MASS)
library(neuralnet)
wine_white <- read.csv("~/gitclone/R/First_R/winequality-white.csv")
########################### combine dataset #####
wine_white$quality[wine_white$quality == 4] <- 5
wine_white$quality[wine_white$quality == 6] <- 5
wine_white$quality[wine_white$quality == 7] <- 8
wine_white$quality[wine_white$quality == 9] <- 8

########################### combine dataset #####


sub <- sample(1:nrow(wine_white),round(nrow(wine_white)*2/3))
data_train <- wine_white[sub,]
data_test <- wine_white[-sub,]
lm.fit <- glm(quality~., data=data_train)
summary(lm.fit)
pr.lm <- predict(lm.fit,data_test)
MSE.lm <- sum((pr.lm - data_test$quality)^2)/nrow(data_test)
MSE.lm
maxs <- apply(wine_white, 2, max)
mins <- apply(wine_white, 2, min)
scaled <- as.data.frame(scale(wine_white, center = mins, scale = maxs - mins))
train_ <- scaled[sub,]
test_ <- scaled[-sub,]
n <- names(train_)
f <- as.formula(paste("quality ~",paste(n[!n %in% "quality"],collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)
plot(nn)
pr.nn <- compute(nn,test_[,1:11])


# pr.nn_ <- pr.nn$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
pr.nn <- pr.nn$net.result*(max(wine_white$quality)-min(wine_white$quality))+min(wine_white$quality)
# test.r <- (test_$medv)*(max(data$medv)-min(data$medv))+min(data$medv)
test.r <- (test_$quality)*(max(wine_white$quality)-min(wine_white$quality))+min(wine_white$quality)

MSE.nn <- sum((test.r - pr.nn)^2)/nrow(test_)
MSE.nn
print(paste(MSE.lm,MSE.nn))
par(mfrow=c(1,2))
# plot(test$medv,pr.nn_,col=’red’,main=’Real vs predicted NN’,pch=18,cex=0.7)

plot(test_$quality,pr.nn,col = 'red',main = "Read vs pedicted NN",pch = 18, cex = 0.7)
abline(0,1,lwd=2)
# legend(‘bottomright’,legend=’NN’,pch=18,col=’red’, bty=’n’)
legend('bottomright',legend = 'NN',pch = 18,col = 'red',bty = 'n')
plot(test_$quality,pr.lm,col='blue',main='Real vs predicted lm',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend = 'LM',pch = 18,col = 'blue',bty = 'n')

plot(test_$quality,pr.nn,col = 'red',main = "Read vs pedicted NN",pch = 18, cex = 0.7)
points(test_$quality,pr.lm,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('NN','LM'),pch=18,col=c('red','blue'))

boxplot(cv.error,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)














































