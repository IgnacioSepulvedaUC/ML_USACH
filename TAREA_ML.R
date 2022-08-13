### CODIGO TAREA

#### Package
library(readr)
library(tidyverse)
library(gtools)
library(readxl)
library(modeldata)
library(caret)   
library(rsample) 
library(ggplot2)
library(dplyr)
library(gmodels) 
library(class) 
library(C50)
library(rpart)
library(rpart.plot)
library(fastDummies)


### PARTE I

#### Datos
getwd()
rm(list=ls())
df <- read_delim("MEF-USACH/ML/PARTE 2/bank-additional-full.csv", 
                 delim = ";")
## Drop pdays
df <- df %>% select(-pdays)

#### Str
str(df)

### NAS
colSums(is.na(df))

## Factor data.
factor_cols<- c('job','marital','education',
                'default','housing','loan',
                'contact','month','day_of_week',
                'poutcome','y')
df[factor_cols] <- lapply(df[factor_cols], factor)

## Columnas numéricas
numeric_cols <- names(df)[!names(df) %in% factor_cols]


## bloxplot
par(mfrow = c(3, 3), xaxs = "i")
for (i in numeric_cols){
  boxplot(df[[i]],col='lightblue',main = i,xlab = i)}

## hist
par(mfrow = c(3, 3), xaxs = "i")
for (i in numeric_cols){
  hist(df[[i]],col='lightblue',main = i,xlab = i)}

## Summaary
df %>% select(all_of(numeric_cols)) %>% summary()
df %>% select(all_of(factor_cols)) %>% summary()

## Table
for (i in factor_cols) {
  print(
    table(
      df[[paste(i)]],df$y))
}

## Bivariate
par(mfrow = c(3, 3), xaxs = "i")
for (i in numeric_cols){
  df[['numeric_y']]=as.numeric(df[['y']])
  plot(x=df[[i]],
       y=df$numeric_y,
       xlab=i,
       ylab='Si=2 y No=1',
       main=i,
       col='blue')
}


## Scatter
categorias <- data.frame(combinations(length(unique(numeric_cols)),2,unique(numeric_cols)))
for (i in 1:length(categorias$X1)){
  x=categorias[i,1]
  y=categorias[i,2]
  plot(df[[x]]
       ,df[[y]]
       ,col=df[['y']]
       ,xlab=x
       ,ylab=y,
       ,main=paste(c('Scatter ','Si=1 y No=0')))}


#### KNN

## NORMALIZAMOS
df_z <- df
df_z[all_of(numeric_cols)] <- df %>% 
  select(all_of(numeric_cols)) %>% 
  lapply(scale) %>% 
  as.data.frame()

## PROP
df$y %>% table/length(df$y)

#Split
split <- initial_split(df_z, prop = 0.8, strata = "y")
df_train <- training(split) 
df_test <- testing(split) 

#Valida
round(prop.table(table(df_train$y)),2)
round(prop.table(table(df_test$y)),2)

#Asigna
y_train <- df_train$y
y_test <- df_test$y

# Especificar metodo de remuestreo
cv <- trainControl(method = "cv", number = 10)


# Especificar parametros
hyper_grid <- expand.grid(k = seq(1, 60, by = 2))

# Entrenamiento
knn_fit <- train(y ~ ., data = df_train, method = "knn", 
                 trControl = cv, tuneGrid = hyper_grid,  metric = "Accuracy")

knn_fit
#plot
ggplot(knn_fit)


###Mejora KNN

# Especificar metodo de remuestreo
cv <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# Especificar parametros
hyper_grid <- expand.grid(k = seq(1, 60, by = 2))

# Entrenamiento
knn_fit <- train(y ~ ., data = df_train, method = "knn", 
                 trControl = cv, tuneGrid = hyper_grid,  metric = "Accuracy")

knn_fit
#Plot
ggplot(knn_fit)

### Prediccion
knn_model <- knn3(y~.,df_train, k = 30)
knn_model_result <- predict(knn_model, newdata = df_test, type = 'class')

## RESULTADOS
CrossTable(x = df_test$y, y = knn_model_result, prop.chisq = FALSE, 
           prop.c = FALSE, prop.r = FALSE)

## BASELINE
conf_matrix <- table(rep('no',length(df_test$y)), df_test$y)
conf_matrix
conf_matrix[1,1]/ length(as.numeric(knn_model_result))

## MODEL
conf_matrix <- table(knn_model_result, df_test$y)
conf_matrix
print(c('Accuracy:',round((conf_matrix[1,1] + conf_matrix[2,2]) / length(as.numeric(knn_model_result)),3)))


## DT
# Especificar parametros
hyper_grid_DT <- expand.grid(trials=c(1:100), model="tree", winnow = c(FALSE))

# Entrenamiento
DT_fit <- train(y ~ ., data = df_train, method = "C5.0", 
                trControl = cv, tuneGrid = hyper_grid_DT,  metric = "Accuracy")

DT_fit
#plot
ggplot(DT_fit)

# TRAIN
DT_model <- C5.0(df_train %>% select(-'y'), df_train$y, trials = 100)
# PREDICT
DT_result <- predict(DT_model, df_test %>%  select(-"y"))
### RESULTADO
conf_matrix <- table(DT_result, df_test$y)
conf_matrix
print(c('Accuracy:',round((conf_matrix[1,1] + conf_matrix[2,2]) / length(as.numeric(DT_result)),3)))

## MEJORA

## RANDOM FOREST
library('randomForest')
set.seed(120)  # Setting seed
classifier_RF = randomForest(x = df_train %>% select (-c(y)) ,
                             y = df_train$y,
                             ntree = 1000)

# Predicting the Test set results
y_pred = predict(classifier_RF, newdata = df_test %>% select(-c(y)))

# Confusion Matrix
conf_matrix = table(df_test$y, y_pred)
conf_matrix
print(c('Accuracy:',round((conf_matrix[1,1] + conf_matrix[2,2]) / length(as.numeric(y_pred)),3)))
