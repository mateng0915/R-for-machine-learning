#import package
install.packages("Rcpp")
library(RSNNS)
library(readr)


#########
Diabetes <- read_csv("~/gitclone/R/First_R/Diabetes.csv")
#Disrupt the data sequence
# set.seed(12345) best
set.seed(12345)
Diabetes <- Diabetes[sample(1:nrow(Diabetes),length(1:nrow(Diabetes))),1:ncol(Diabetes)]

################### mix ################################
Diabetes$c1[Diabetes$c1 == 0] <- 3.8
Diabetes$c2[Diabetes$c2 == 0] <- 90.0
Diabetes$c3[Diabetes$c3 == 0] <- 65.0
Diabetes$c4[Diabetes$c4 == 0] <- 20.5
Diabetes$c5[Diabetes$c5 == 0] <- 115.2       #115.2
Diabetes$c6[Diabetes$c6 == 0] <-  32.0

################### end ################################

Diabetes_input <- Diabetes[,1:8]
#define output of grid
Diabetes_target <- Diabetes[,9]
Diabetes_target <- decodeClassLabels(unlist(Diabetes_target))

#split the test and training data
Diabetes_data <- splitForTrainingAndTest(Diabetes_input,Diabetes_target,ratio = 0.25)
#Data standardization
Diabetes_data <- normTrainingAndTestSet(Diabetes_data,type = '0_1')

####################################  RBF ###################################################
model1 <- rbf(Diabetes_data$inputsTrain, Diabetes_data$targetsTrain, size=10, maxit=100, 
             initFunc="RBF_Weights",
             initFuncParams=c(0, 1, 0, 0.01, 0.01), learnFunc="RadialBasisLearning",
             learnFuncParams=c(1e-8, 0, 1e-8, 0.1, 0.8),
             updateFunc="Topological_Order", updateFuncParams=c(0.0),
             shufflePatterns=TRUE,computeIterativeError=TRUE,linOut=TRUE, 
             inputsTest = Diabetes_data$inputsTest, Diabetes_data$targetsTest)


plotIterativeError(model1)

prediction1 <- predict(model1,Diabetes_data$inputsTest)

Freq_mlp1 <- confusionMatrix(Diabetes_data$targetsTest,prediction1)
Freq_mlp1

accuracy_mlp1 <-  sum(diag(Freq_mlp1))/sum(Freq_mlp1)
accuracy_mlp1
####################################  END ###################################################

####################################  BP ###################################################

model2 <- mlp(Diabetes_data$inputsTrain, Diabetes_data$targetsTrain, size=4, learnFuncParams=c(0.3,0.00001),
              maxit=100, inputsTest=Diabetes_data$inputsTest, targetsTest=Diabetes_data$targetsTest)

plotIterativeError(model2)

prediction2 <- predict(model2,Diabetes_data$inputsTest)

Freq_mlp2 <- confusionMatrix(Diabetes_data$targetsTest,prediction2)
Freq_mlp2

accuracy_mlp2 <-  sum(diag(Freq_mlp2))/sum(Freq_mlp2)
accuracy_mlp2
####################################  END ###################################################

####################################  elman ###################################################

model3 <- elman(Diabetes_data$inputsTrain, Diabetes_data$targetsTrain, 
                size = c(8,8), learnFuncParams = c(0.1), maxit = 500,
                inputsTest = Diabetes_data$inputsTest,targetsTest = Diabetes_data$targetsTest,
                linOut = FALSE)

plotIterativeError(model3)

prediction3 <- predict(model3,Diabetes_data$inputsTest)

Freq_mlp3 <- confusionMatrix(Diabetes_data$targetsTest,prediction3)
Freq_mlp3

accuracy_mlp3 <-  sum(diag(Freq_mlp3))/sum(Freq_mlp3)
accuracy_mlp3
####################################  end ###################################################

####################################  jordan ###################################################

modelJordan <- jordan(Diabetes_data$inputsTrain, Diabetes_data$targetsTrain, 
                      size=c(8), learnFuncParams=c(0.1), maxit=100,
                      inputsTest=Diabetes_data$inputsTest, 
                      targetsTest=Diabetes_data$targetsTest, linOut=FALSE)

plotIterativeError(modelJordan)
prediction4 <- predict(modelJordan,Diabetes_data$inputsTest)
Freq_mlp4 <- confusionMatrix(Diabetes_data$targetsTest,prediction4)
Freq_mlp4

accuracy_mlp4 <- sum(diag(Freq_mlp4))/sum(Freq_mlp4)
accuracy_mlp4

####################################  end ###################################################


################################### voting ################################################


prediction_final <- ( prediction2 + prediction4 +prediction1)/3

ifelse(prediction_final > 0.5,1,0)

# prediction_final[prediction_final >= 2] <- 3
# prediction_final[prediction_final < 2] <- 0
# prediction_final[prediction_final == 3] <- 1

Freq_mlp <- confusionMatrix(Diabetes_data$targetsTest,prediction_final)
Freq_mlp

accuracy_mlp <-  sum(diag(Freq_mlp))/sum(Freq_mlp)
accuracy_mlp

print("________________+++++++++++++++++++++____________________")
print("-----Average Accurancy:")
ac_av <- ((accuracy_mlp1 + accuracy_mlp4 +accuracy_mlp2 )/3)
ac_av



################################### end #####################################################









