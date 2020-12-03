library(rsample)# data splitting
library(caret)
library(tidyverse)
library(DT)
library(sjPlot)# to represent the results in nice way
library(dplyr)
library("ggpubr")
library(ranger)  
library(vip)# visualize feature importance 
library(pdp)# visualize feature effects
library(gbm)
library(xgboost)

# import dataset
coladm <- read.csv("C:\\Users\\ghadi\\Documents\\DataSet\\College_admission.csv")
dim(coladm)
names(coladm)
str(coladm)

# Structure Of The Dataset

# str(coladm) # show types before change type
coladm$admit=as.factor(coladm$admit)
coladm$ses=as.factor(coladm$ses)
coladm$Gender_Male=as.factor(coladm$Gender_Male)
coladm$Race=as.factor(coladm$Race)
coladm$rank=as.factor(coladm$rank)
# str(coladm) # show types after change type
summary(coladm) # show type & values for each column 
# Find the missing values & perform missing value treatment
coladm <- drop_na(coladm)



# Exploratory Data Analysis

datatable(head(coladm))

# Number of students
coladm %>% 
  count(admit)

ggplot(coladm, aes(admit,fill = admit)) +
  geom_bar()+
  scale_fill_brewer(palette="Green")+
  ggtitle("Number of Admissions")+ theme(plot.title=element_text(face="bold"))#bold title



# Outliers
## Find Outliers in gre
# find & perform outlier treatment 
# plot box plot to detect outliers in Graduate Record Exam Scores
ggplot(coladm, aes(x=admit, y=gre, fill=admit)) + 
  geom_boxplot(alpha=0.8) +
  theme(legend.position="none") +
  scale_fill_brewer(palette="Green")+
  ggtitle("GRE Before Remove Outliers")+     
  theme(plot.title=element_text(face="bold"))#bold title

# choose the outliers and show where they are in dataset
out <-boxplot.stats(coladm$gre)$out
out_ind <- which(coladm$gre %in% c(out))
out_ind #show index

# show outliers values
coladm[out_ind, ]  
# remove outlines using index
coladm<-coladm[!(row.names(coladm) %in% c('72','180','305','316')), ]

# boxplot after remove outliers
ggplot(coladm, aes(x=admit, y=gre, fill=admit)) + 
  geom_boxplot(alpha=0.8) +
  theme(legend.position="none") +
  scale_fill_brewer(palette="Green")+
  ggtitle("GRE Before Remove Outliers")+     
  theme(plot.title=element_text(face="bold"))#bold title

## Find Outliers in gpa
# plot box plot to detect outliers in Grade Point Average
ggplot(coladm, aes(x=admit, y=gpa, fill=admit)) + 
  geom_boxplot(alpha=0.8) +
  theme(legend.position="none") +
  scale_fill_brewer(palette="Green")+
  ggtitle("GPA Before Remove Outliers")+     
  theme(plot.title=element_text(face="bold"))#bold title

# choose the outliers and show where they are in dataset
out <-boxplot.stats(coladm$gpa)$out
out_ind <- which(coladm$gpa %in% c(out))
out_ind #show index
# show outliers values
coladm[out_ind, ]
# remove outlines using index
coladm<-coladm[!(row.names(coladm) %in% c('290')), ]

# boxplot after remove outliers
ggplot(coladm, aes(x=admit, y=gpa, fill=admit)) + 
  geom_boxplot(alpha=0.8) +
  theme(legend.position="none") +
  scale_fill_brewer(palette="Green")+
  ggtitle("GPA After Remove Outliers")+     
  theme(plot.title=element_text(face="bold"))#bold title


# Data Visualization

ggdensity(coladm$gpa, 
          main = "GPA Density Plot",
          xlab = "gpa")
ggdensity(coladm$gre, 
          main = "GRE Density Plot",
          xlab = "gre")

shapiro.test(coladm$gpa)
shapiro.test(coladm$gre)
# apply log
coladm$gpa = log(coladm$gpa)
coladm$gre = log(coladm$gre)
# shapiro test to check normality for gpa
shapiro.test(coladm$gpa)
shapiro.test(coladm$gre)
# plot shapes again 
ggdensity(coladm$gpa, 
          main = "Density plot of gpa",
          xlab = "gpa")
ggdensity(coladm$gre, 
          main = "Density plot of gre",
          xlab = "gre")

#linear regression to see relation between gpa & gre
relr <- lm(gre ~ gpa,data=coladm)
summary(relr)
# to represent the results in nice way
tab_model(relr)
ggplot(coladm, aes(x=gpa, y=gre)) +
  geom_point(size=3, shape=1,color='DarkGreen')

# Modeling
# split data into train & test
set.seed(123)
split  <- initial_split(coladm, prop = 0.7, strata = "admit")
coladm_train  <- training(split)
coladm_test   <- testing(split)

## Logistic Regression 
model_glm = glm(admit ~ . , family="binomial", data = coladm_train)
summary(model_glm)

model_glm2 = glm(admit ~ rank , family="binomial", data = coladm_train)
summary(model_glm2)

# Predictions on the training set
predictTrain = predict(model_glm2, data = coladm_train, type = "response")

# Confusion matrix on training data
table(coladm_train$admit, predictTrain >= 0.5)
(114+268)/nrow(coladm_train)

#Predictions on the test set
predictTest = predict(model_glm2, newdata = coladm_test, type = "response")

# Confusion matrix on test set
table(coladm_test$admit, predictTest >= 0.5)
158/nrow(coladm_test)


# GBM
set.seed(123)
coladm_gbm <- gbm(
  formula = admit ~ .,#have all features 
  data = coladm_train,
  distribution = "gaussian",
  n.trees = 100, #number of tree , i chose default number of trees 100 to avoid overfit
  shrinkage = 0.1, #learning rate
  interaction.depth = 1, # depth of each tree I kept the default value 1 so it's not able to capture any interaction effects
  n.minobsinnode = 10, # also here kept default value for tree terminal nodes 
  cv.folds = 8 # i decide to reduce folds to fit better with smaller data
)  

# find index for n trees with minimum CV error
min_MSE <- which.min(coladm_gbm$cv.error)

# model performance get MSE and compute RMSE
sqrt(coladm_gbm$cv.error[min_MSE])

gbm.perf(coladm_gbm, method = "cv")
#Tree Tuning strategy
set.seed(123)
coladm_gbm1 <- gbm(
  formula = admit ~ .,
  data = coladm_train,
  distribution = "gaussian", 
  n.trees = 100, #reduce No of tree to fit with learning rate
  shrinkage = 0.005, #reduce learning rate to increase accuracy 
  interaction.depth = 10, #<<
  n.minobsinnode = 10, #<<
  cv.folds = 15 #<<
)

# find index for n trees with minimum CV error
min_MSE <- which.min(coladm_gbm1$cv.error)

# get MSE and compute RMSE
sqrt(coladm_gbm1$cv.error[min_MSE])

gbm.perf(coladm_gbm1, method = "cv")

#Tree Tuning using Hyper grid
# search grid
hyper_grid <- expand.grid(
  n.trees = 20,
  shrinkage = .01,
  interaction.depth = c(3, 5, 7), 
  n.minobsinnode = c(5, 10, 15) 
)

model_fit <- function(n.trees, shrinkage, interaction.depth, n.minobsinnode) {
  set.seed(123)
  m <- gbm(
    formula = admit ~ .,
    data = coladm_train,
    distribution = "gaussian",
    n.trees = n.trees,
    shrinkage = shrinkage, #use function values to have best result for learning rate
    interaction.depth = interaction.depth, #use function values to have best result for number of iteration
    n.minobsinnode = n.minobsinnode,#use function values to have best result for number of tree terminal nodes
    cv.folds = 10
  )
  # compute RMSE
  sqrt(min(m$cv.error))
}

hyper_grid$rmse <- pmap_dbl(
  hyper_grid,
  ~ model_fit(
    n.trees = ..1,
    shrinkage = ..2,
    interaction.depth = ..3,
    n.minobsinnode = ..4
  )
)

arrange(hyper_grid, rmse)












# Random Forests

# number of features
features <- setdiff(names(coladm_train), "admit")

# perform basic random forest model
RFmodel <- ranger(
  formula    = admit ~ ., 
  data       = coladm_train, 
  num.trees  = length(features) * 10,
  mtry       = floor(length(features) / 3),
  respect.unordered.factors = 'order',
  verbose    = FALSE,
  seed       = 123
)
# look at results
RFmodel
# compute RMSE
sqrt(RFmodel$prediction.error)

# find important features 
coladm_train %>%
  summarise_if(is.factor, n_distinct) %>% 
  gather() %>% 
  arrange(desc(value))

# Tune Tree
# number of features
n_features <- ncol(coladm_train) - 1

# ranger function
oob_error <- function(trees) {
  fit <- ranger(
    formula    = admit ~ ., 
    data       = coladm_train, 
    num.trees  = trees, #<<
    mtry       = floor(n_features / 3),
    respect.unordered.factors = 'order',
    verbose    = FALSE,
    seed       = 123
  )
  
  sqrt(fit$prediction.error)
}

# tuning grid
trees <- seq(10, 1000, by = 20)

(rmse <- trees %>% purrr::map_dbl(oob_error))


# feature correlation
cor_matrix <- coladm_train %>%
  mutate_if(is.factor, as.numeric) %>%
  cor()

data_frame(
  row  = rownames(cor_matrix)[row(cor_matrix)[upper.tri(cor_matrix)]],
  col  = colnames(cor_matrix)[col(cor_matrix)[upper.tri(cor_matrix)]],
  corr = cor_matrix[upper.tri(cor_matrix)]
) %>%
  arrange(desc(abs(corr)))

# target correlation
data_frame(
  row  = rownames(cor_matrix)[row(cor_matrix)[upper.tri(cor_matrix)]],
  col  = colnames(cor_matrix)[col(cor_matrix)[upper.tri(cor_matrix)]],
  corr = cor_matrix[upper.tri(cor_matrix)]
) %>% filter(col == "admit") %>%
  arrange(desc(abs(corr)))

# Tune hyper grid
hyper_grid <- expand.grid(
  mtry            = floor(n_features * c(.05, .15, .25, .333, .4)),
  min.node.size   = c(1, 3, 5),
  replace         = c(TRUE, FALSE),
  sample.fraction = c(.5, .63, .8),
  rmse            = NA
)

# number of hyperparameter combinations
nrow(hyper_grid)

head(hyper_grid)

default_rmse <- sqrt(RFmodel$prediction.error)
#Tuning results 
hyper_grid %>%
  arrange(rmse) %>%
  mutate(perc_gain = (default_rmse - rmse) / default_rmse * 100) %>%
  head(10)
