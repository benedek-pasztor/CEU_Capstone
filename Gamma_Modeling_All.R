### This is the master modeling file of the Menetrend project. It has two parts: 1. Modeling, 2. Forecasting, both are defined in a function which may be run.




############################ I. MODELING #############################

library(data.table)
library(tidyverse)
library(plyr)
library(lubridate)
library(zoo)
library(ranger)
library(caret)
library(e1071)
library(gbm)    
library(xgboost)
library(dplyr)

# Clear environment
rm(list=ls())

# FOLDER DEFINITIONS
dirs <- read.csv("C:/Users/B19883/Documents/R_AUTORUNS/SOURCE.csv", sep = ";", stringsAsFactors = F)
dir <- dirs[1,"Source"]
dir_upl <- dirs[2, "Source"]
dir_dashboard <- dirs[3, "Source"]


##subdirectories
raw <- paste0(dir, "raw/")
clean <- paste0(dir,"clean/")
output <- paste0(dir,"output/")


source(paste0(dir, "Gamma_Functions.R"))

DBRefresh()
DBMerge()



# DATA IMPORT
data_raw <- fread(paste0(clean, "Database.csv"))
data_raw$datetime <- as.POSIXct((data_raw$datetime), format = "%Y-%m-%d %T")

d <- data.table(data_raw)
d[ForecastOrNot == 0, max(datetime)]

data <- data_raw[!duplicated(data_raw)]

d <- data %>% filter(datetime > "2019-02-20") %>% as.data.table()
d[ForecastOrNot %in% c(0, 1), sum(P), by = .(as.Date(datetime), pp)]
data[pp == "szigetvar_gamma" & as.Date(datetime, tz = "CET") == "2019-03-04", ]


data_raw$month <- month(data_raw$datetime)


# 
# ##taking out 2019-03-08 and 03-15 for testing
# data_raw$date <- as.Date(data_raw$datetime, tz = "CET")
# data_test <- data_raw %>% filter(date %in% c(as.Date("2019-03-08", tz = "CET"),
#                                              as.Date("2019-03-15", tz = "CET"),
#                                              as.Date("2019-03-21", tz = "CET")))
# 
# '%notin%' <- function(x,y)!('%in%'(x,y))
# 
# data_raw <- data_raw %>% filter(date %notin% c(as.Date("2019-03-08", tz = "CET"),
#                                              as.Date("2019-03-15", tz = "CET"),
#                                              as.Date("2019-03-21", tz = "CET")))




data <- data_raw %>% select(P, month, cloudCover, temperature, humidity, icon, precipProbability, precipIntensity, dewPoint, apparentTemperature, Rad, ForecastOrNot,
                            GHI, TOA, month)

data$month <- month(data$month)


#data <- data %>% filter(P != 0)
# Creating training and testing samples
set.seed(20190115)
train_indices <- createDataPartition(data$P, p = 0.9, list = FALSE) #I don't need to put price specifically
data_train <- data[train_indices, ]
data_holdout <- data[-train_indices, ] 

#################### LINEAR MODEL ##################
train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)


set.seed(857)
linear_model <- train(P ~ .,
                      method = "lm",
                      data = data_train,
                      trControl = train_control)
linear_model



#################### SIMPLE TREE MODEL ##################

set.seed(857)
simple_tree_model <- train(P ~ .,
                           method = "rpart",
                           data = data_train,
                           tuneGrid = data.frame(cp = c(0.001,0.01, 0.02, 0.05)),
                           trControl = train_control,
                           control = list(maxdepth = 15))
simple_tree_model

rpart.plot::rpart.plot(simple_tree_model[["finalModel"]], cex=0.5)


#################### RANDOM FOREST ######################


# Simple model
set.seed(20190115)
tune_grid <- expand.grid(
  .mtry = 10,
  .splitrule = "variance",
  .min.node.size = c(10)
)

rf_model <- train(
  formula(P ~ .),
  data = data_train,
  method = "ranger",
  trControl = train_control,
  tuneGrid = tune_grid,
  importance = "permutation",
  preProcess = c("center", "scale")
)
rf_model

#################### GRADIENT BOOSTING ######################


gbm_grid <- expand.grid(n.trees = c(100, 500, 1000), 
                        interaction.depth = c(5), 
                        shrinkage = c(0.1),
                        n.minobsinnode = c(5))
set.seed(857)
gbm_model <- train(P ~ .,
                   method = "gbm",
                   data = data_train,
                   trControl = train_control,
                   preProcess = c("center", "scale", "pca"),
                   tuneGrid = gbm_grid,
                   verbose = FALSE # gbm by default prints too much output
)
gbm_model

#################### XG BOOSTING ######################


xgb_grid <- expand.grid(nrounds = c(500, 1000),
                        max_depth = c(5, 7),
                        eta = c(0.05, 0.07),
                        gamma = 0,
                        colsample_bytree = c(0.7, 0.8),
                        min_child_weight = 1, # similar to n.minobsinnode
                        subsample = c(0.5))
set.seed(857)
xgboost_model <- train(P ~ .,
                       method = "xgbTree",
                       data = data_train,
                       trControl = train_control,
                       preProcess = c("center", "scale", "pca"),
                       tuneGrid = xgb_grid)
xgboost_model


## MODEL SELECTION
resamples_object <- resamples(list("linear_model" = linear_model,
                                   "simple_tree_model" = simple_tree_model,
                                    "rf_model" = rf_model,
                                   "gbm_model" = gbm_model,
                                   "xgboost_model" = xgboost_model))
models <- summary(resamples_object)

models_cvrmse <- as.data.frame(models$statistics$RMSE)
models_cvrmse_means <- data.frame(model = row.names(models_cvrmse), rmse = models_cvrmse$Mean)

best_model <- as.character(models_cvrmse_means$model[which.min(models_cvrmse_means$rmse)])
best_model

save.image(file = paste0(dir, "models/", Sys.Date() - days(1), ".RData"))



data_test <- data_raw %>% filter(as.Date(datetime, tz = "CET") %in% c(as.Date("2019-03-08", tz = "CET"),
                                             as.Date("2019-04-03", tz = "CET"),
                                             as.Date("2019-04-04", tz = "CET")))


data_test <- data_test %>% filter(ForecastOrNot == 1)
## Checking for test

data_test <- data_test %>% 
  mutate(predicted_P_xgBoost = round(ifelse(TOA != 0, predict(xgboost_model, newdata = data_test), 0)), 0)


data_test <- data_test %>% 
  mutate(predicted_P_rf = round(ifelse(TOA != 0, predict(rf_model, newdata = data_test), 0)), 0)


data_test <- data_test %>% 
  mutate(predicted_P_gbm = round(ifelse(TOA != 0, predict(gbm_model, newdata = data_test), 0)), 0)
data_test$Error_xgBoost <- data_test$predicted_P_xgBoost-data_test$P
data_test$Error_rf <- data_test$predicted_P_rf-data_test$P
data_test$Error_gbm <- data_test$predicted_P_gbm-data_test$P

sqrt(mean(data_test$Error_xgBoost^2))
sqrt(mean(data_test$Error_rf^2))
sqrt(mean(data_test$Error_gbm^2))

    str(data_test)
    data_test_l <- melt(data_test, measure.vars = c("P", "predicted_P_rf", "predicted_P_xgBoost", "predicted_P_gbm")) %>% 
      filter(as.Date(datetime, tz = "CET") %in% c(as.Date("2019-03-08", tz = "CET")))  %>% filter(pp == "szigetvar_gamma")   
    ggplot(data_test_l, aes(datetime, value, color = variable)) + geom_path()

    
    data_test_l <- melt(data_test, measure.vars = c("P", "predicted_P_rf")) %>% 
      filter(as.Date(datetime, tz = "CET") %in% c(as.Date("2019-03-08", tz = "CET")))  %>% filter(pp == "szigetvar_gamma")   
    ggplot(data_test_l, aes(datetime, value, color = variable)) + geom_path()
    
    
rm(forecast_date)





data_holdout <- data_train

data_holdout <- data_holdout %>% 
  mutate(predicted_P_xgBoost = round(ifelse(GHI != 0, predict(xgboost_model, newdata = data_holdout), 0)), 0)
data_holdout <- data_holdout %>% 
  mutate(predicted_P_rf = round(ifelse(GHI != 0, predict(rf_model, newdata = data_holdout), 0)), 0)
data_holdout <- data_holdout %>% 
  mutate(predicted_P_gbm = round(ifelse(GHI != 0, predict(gbm_model, newdata = data_holdout), 0)), 0)
data_holdout <- data_holdout %>% 
  mutate(predicted_P_lm = round(ifelse(GHI != 0, predict(linear_model, newdata = data_holdout), 0)), 0)
data_holdout <- data_holdout %>% 
  mutate(predicted_P_cart = round(ifelse(GHI != 0, predict(simple_tree_model, newdata = data_holdout), 0)), 0)


data_holdout$Error_xgBoost <- data_holdout$predicted_P_xgBoost-data_holdout$P
data_holdout$Error_rf <- data_holdout$predicted_P_rf-data_holdout$P
data_holdout$Error_gbm <- data_holdout$predicted_P_gbm-data_holdout$P
data_holdout$Error_lm <- data_holdout$predicted_P_lm-data_holdout$P
data_holdout$Error_cart <- data_holdout$predicted_P_cart-data_holdout$P



sqrt(mean(data_holdout$Error_xgBoost^2))
sqrt(mean(data_holdout$Error_rf^2))
sqrt(mean(data_holdout$Error_gbm^2))
sqrt(mean(data_holdout$Error_lm^2))
sqrt(mean(data_holdout$Error_cart^2))

data_holdout_l <- melt(data_holdout, measure.vars = c("Error_xgBoost", "Error_gbm", "Error_lm"))
ggplot(data_holdout_l, aes(P, value, color = variable)) + geom_point(alpha = 0.07)



forecast_date <- "2019-03-1"
dir_upl <- "\\\\eon.ise.local/DFSRoot02442/TEAM/Kereskedelmi_osztály/PIACI_ÉRTÉKESÍTÉS/Szekunder/NapelemMenetrend/"
rm(dir, dir_dashboard, dir_upl, forecast_date_char, forecast_end_utc, forecast_start_utc, forecast_date)



forecast_date <- "2019-04-12"
forecast_gamma("2019-04-12")
####################### II.  F O R E C A S T I N G #######################
forecast_gamma <- function(forecast_date){
  
  library(darksky)
  library(data.table)
  library(tidyverse)
  library(plyr)
  library(lubridate)
  library(zoo)  
  library(jsonlite) 
  library(XML)
  library(dplyr)
  library(stringr)
  
  forecast_date <- as.Date(forecast_date, tz = "CET")
  
  
  # loading model
  # load(paste0(dir, "Model_2019-03-07.RData"))
  
  ## API call for weather
  url <- "https://api.darksky.net/forecast/b1b4a5b40f957cbaa84d80efcdaa3426/46.04865,17.80554?units=si&extend=hourly"
  darksky_raw <- readLines(url)
  darksky_list <- fromJSON(darksky_raw)
  
  darksky_hourly <- darksky_list$hourly$data
  
  darksky_hourly$datetime <- as.POSIXct(as.numeric(as.character(darksky_hourly$time)),origin="1970-01-01",tz="CET") 
  # darksky_hourly <- darksky_hourly %>% select(datetime, cloudCover, temperature, humidity, icon, precipProbability)
  
  
  
  datau <- fread(paste0(raw, "DarkSky_RawDatabase.csv"))
  datau$datetime <- as.POSIXct((datau$datetime), format = "%Y-%m-%d %T")
  
  datau$month <- month(datau$datetime)
  
  # Correcting any missing value   ## correcting possible missing icon
  # darksky_hourly_cor <- darksky_hourly
  # darksky_hourly_cor$iconmode <- mode(datau$icon)   
  # darksky_hourly_cor <- darksky_hourly %>% mutate(precipIntensity = ifelse(is.na(precipIntensity), 0, precipIntensity),
  #                                                 precipProbability = ifelse(is.na(precipProbability), 0, precipProbability),
  #                                                 ozone = ifelse(is.na(ozone), mean(ozone, na.rm = T), ozone),
  #                                                 cloudCover = ifelse(is.na(cloudCover), 0, cloudCover),
  #                                                 temperature = ifelse(is.na(cloudCover), 20, temperature),
  #                                                 apparentTemperature = ifelse(is.na(apparentTemperature), 20, apparentTemperature),
  #                                                 dewPoint = ifelse(is.na(dewPoint), 0, dewPoint),
  #                                                 humidity = ifelse(is.na(humidity), 20, humidity),
  #                                                 icon = ifelse(is.na(icon), iconmode, icon)
  # )
  # str(darksky_hourly_cor)
  # 
  # darksky_hourly <- darksky_hourly_cor
  # 
  
  ## Creating data frame for each quarter of hour
  # creating datetime for each 15 mins
  times <- data.frame(datetime = seq(min(darksky_hourly$datetime), max(darksky_hourly$datetime), by = '15 mins'))
  
  # joining and transformating to data table
  darksky_hourly <- (left_join(times, darksky_hourly, by = "datetime"))
  
  # getting variables of datau and their class
  vars <- data.frame(unlist(lapply(sapply(darksky_hourly, class),  `[[`, 1)), row.names = NULL)
  vars <- cbind(names(darksky_hourly), vars)
  names(vars) <- c("varnames", "class")
  
  #which no of columns are characters
  vars_char_ind <- which(vars$class %in% "character")
  vars_num_ind <- which(vars$class %in% c("numeric", "integer"))
  
  ## Linear interpolation for numbers
  for (i in vars_num_ind){
    interp <- approx(darksky_hourly$datetime,darksky_hourly[[i]],xout=darksky_hourly$datetime)
    darksky_hourly[[i]] <- interp$y
  }
  
  
  # setting the character variables
  for(j in vars_char_ind){
    if(class(darksky_hourly[[j]]) %in% "character"){
      darksky_hourly %>% set(, j, na.locf(darksky_hourly[[j]], fromLast = T))
    }
  }
  
  ## Splitting darksky data to quarter hours ends
  
  DS <- darksky_hourly
  
  
  Rad <- read.csv(paste0(raw, "Radiation_RawDatabase.csv"))
  Rad$datetime <- as.POSIXct((Rad$datetime), format = "%Y-%m-%d %T", tz = "CET")
  
  CAMS <- read.csv(paste0(raw, "CAMS_RawDatabase.csv"), sep = ";")
  CAMS$datetime <- as.POSIXct((CAMS$datetime), format = "%Y-%m-%d %T", tz = "CET")
  
  
  
  datau_seg <- data.table(merge(darksky_hourly, Rad, by = "datetime"))
  datau <- data.table(merge(datau_seg, CAMS, by = "datetime"))
  datau$ForecastOrNot <- 1
  
  #Day selection - day+1
  datau <- datau[as.Date(datetime, tz = "CET") == forecast_date]
  
  datau$month <- month(datau$datetime)
  # PREDICTION
  datau_w_prediction <- datau %>% 
    mutate(predicted_P = round(ifelse(GHI != 0, predict(rf_model, newdata = datau), 0)), 0) #### MELYIK MODELLL!!!!
  
  datau_w_prediction <- datau_w_prediction %>% mutate(ifelse(predicted_P < 0, 0, predicted_P))
  
  write.table(datau_w_prediction, paste0(raw, "ORIGINALS/DS/DS_Forecast_", forecast_date, "2.csv"), row.names = F, sep = ";")
  
  
  datau_w_prediction <- datau_w_prediction %>% select(datetime, predicted_P)

  ggplot(datau_w_prediction, aes(x = datetime, y = predicted_P)) +
    geom_line(alpha = 0.3) +
    theme_bw()
  
  
  
  # C R E A T I N G  X M L
  
  ## Parameter definition
  forecast_start_utc <- as.character(with_tz(as.POSIXct(paste0(forecast_date - days(1), "24:00")), "UTC")) # Possible parameter
  forecast_end_utc <- as.character(with_tz(as.POSIXct(paste0(forecast_date, "24:00")), "UTC")) # Possible parameter
  
  
  forecast_date_char <- as.character(forecast_date)
  ### Version automatic definition
  #listing files
  filesdf <- data.frame(filenames = list.files(paste0(dir_upl, "Feltoltesre_Benedek")))
  #from list to data.frame
  filesdf <- data.frame(t(data.frame(str_split(filesdf$filenames, "_"))))
  row.names(filesdf) <- NULL
  
  #renaming variables
  colnames(filesdf) <- c("pp", "date", "version")
  
  #version variable cleaning
  filesdf$version <- gsub("\\..*","",filesdf$version)
  filesdf <- filesdf %>% filter(date == gsub("-", "", forecast_date_char))
  
  if (nrow(filesdf) == 0) {
    version = 1
  }
  if (nrow(filesdf) != 0) {
    version = max(as.integer(filesdf$version), na.rm = T) + 1
  }
  
  
  ## Version check
  # read.xlsx(LotsofmydatainExcel.xlsm,sheetName="Deaths",as.data.frame=TRUE)
  
  
  forecast_interval <- gsub(" ", "T", paste0(substr(forecast_start_utc, 1, nchar(forecast_start_utc)-3), "Z/", 
                                             substr(forecast_end_utc, 1, nchar(forecast_end_utc)-3),"Z"))
  current_time <- gsub(" ", "T", paste0(with_tz(Sys.time(), tzone = "CET"), "Z"))
  
  
  datau_w_prediction_forXML <- data.frame(Pos = 1:nrow(datau_w_prediction), Qty = datau_w_prediction$predicted_P)
  
  
  ## XML Construction
  doc <- newXMLDoc()
  root <- newXMLNode("ScheduleMessage",
                     attrs = c(DtdRelease = 3 , DtdVersion = 3), 
                     doc=doc)
  
  newXMLNode("MessageIdentification", attrs = c(v = "15X-K-KILOGA---O_PROD"), parent = root)
  newXMLNode("MessageVersion", attrs = c(v = version), parent = root) ## VERSION IS A PARAMETER !!
  newXMLNode("MessageType", attrs = c(v = "Z71"), parent = root)
  newXMLNode("ProcessType", attrs = c(v = "A17"), parent = root)
  newXMLNode("ScheduleClassificationType", attrs = c(v = "A01"), parent = root)
  newXMLNode("SenderIdentification", attrs = c(v = "15X-K-KILOGA---O", codingScheme = "A01"), parent = root)
  newXMLNode("SenderRole", attrs = c(v = "A20"), parent = root)
  newXMLNode("ReceiverIdentification", attrs = c(v = "15X-KATENERGIA-V", codingScheme = "A01"), parent = root)
  newXMLNode("ReceiverRole", attrs = c(v = "A06"), parent = root)
  newXMLNode("MessageDateTime", attrs = c(v = current_time), parent = root)  
  newXMLNode("ScheduleTimeInterval", attrs = c(v = forecast_interval), parent = root) 
  ScheduleTimeSeries <- newXMLNode("ScheduleTimeSeries", parent = root)  
  newXMLNode("SendersTimeSeriesIdentification", attrs = c(v = "KILOGAMM-NAP-31"), parent = ScheduleTimeSeries)
  newXMLNode("SendersTimeSeriesVersion", attrs = c(v = version), parent = ScheduleTimeSeries) # VERSION IS A PARAMETER !!
  newXMLNode("BusinessType", attrs = c(v = "A01"), parent = ScheduleTimeSeries) 
  newXMLNode("Product", attrs = c(v = "8716867000016"), parent = ScheduleTimeSeries)
  newXMLNode("ObjectAggregation", attrs = c(v = "A02"), parent = ScheduleTimeSeries)
  newXMLNode("InArea", attrs = c(v = "10YHU-MAVIR----U", codingScheme = "A01"), parent = ScheduleTimeSeries)
  newXMLNode("MeteringPointIdentification", attrs = c(v = "HU000120B11-U-NAPE-SZIGVA-0400-32", codingScheme = "A01"), parent = ScheduleTimeSeries)
  newXMLNode("InParty", attrs = c(v = "15X-K-KILOGA---O", codingScheme = "A01"), parent = ScheduleTimeSeries)
  newXMLNode("MeasurementUnit", attrs = c(v = "KWT"), parent = ScheduleTimeSeries)
  Period <- newXMLNode("Period", parent = ScheduleTimeSeries)  
  newXMLNode("TimeInterval", attrs = c(v = forecast_interval), parent = Period)
  newXMLNode("Resolution", attrs = c(v = "PT15M"), parent = Period)
  
  for (j in 1:nrow(datau_w_prediction)){
    IntervalNode = newXMLNode("Interval", parent=Period)  
    newXMLNode("Pos", attrs = c(v = datau_w_prediction_forXML$Pos[j]), parent=IntervalNode)  
    newXMLNode("Qty", attrs = c(v = datau_w_prediction_forXML$Qty[j]), parent=IntervalNode)  
  }  
  
  saveXML(doc, file = paste0(dir_upl, "Feltoltesre_Benedek/15X-K-KILOGA---O_", 
                             gsub("-", "", forecast_date), "_", version,
                             ".xml"), encoding = "UTF-8", prefix = '<?xml version="1.0" encoding="UTF-8"?>\n')
  
  
  
  
  ### XML Verzio 2
  
  ## Parameter definition
  forecast_start_utc <- as.character(with_tz(as.POSIXct(paste0(forecast_date - days(1), "24:00")), "UTC")) # Possible parameter
  forecast_end_utc <- as.character(with_tz(as.POSIXct(paste0(forecast_date, "24:00")), "UTC")) # Possible parameter
  
  
  forecast_date_char <- as.character(forecast_date)
  ### Version automatic definition
  #listing files
  filesdf <- data.frame(filenames = list.files(paste0(dir_upl, "Feltoltesre_Benedek")))
  #from list to data.frame
  filesdf <- data.frame(t(data.frame(str_split(filesdf$filenames, "_"))))
  row.names(filesdf) <- NULL
  
  #renaming variables
  colnames(filesdf) <- c("pp", "date", "version")
  
  
  # dir_upl2 <- 
  #version variable cleaning
  filesdf$version <- gsub("\\..*","",filesdf$version)
  filesdf <- filesdf %>% filter(date == gsub("-", "", forecast_date_char))
  
  version <- 2
  
  
  ## Version check
  # read.xlsx(LotsofmydatainExcel.xlsm,sheetName="Deaths",as.data.frame=TRUE)
  
  
  forecast_interval <- gsub(" ", "T", paste0(substr(forecast_start_utc, 1, nchar(forecast_start_utc)-3), "Z/", 
                                             substr(forecast_end_utc, 1, nchar(forecast_end_utc)-3),"Z"))
  current_time <- gsub(" ", "T", paste0(with_tz(Sys.time(), tzone = "CET"), "Z"))
  
  
  datau_w_prediction_forXML <- data.frame(Pos = 1:nrow(datau_w_prediction), Qty = datau_w_prediction$predicted_P)
  
  
  ## XML Construction
  doc <- newXMLDoc()
  root <- newXMLNode("ScheduleMessage",
                     attrs = c(DtdRelease = 3 , DtdVersion = 3), 
                     doc=doc)
  
  newXMLNode("MessageIdentification", attrs = c(v = "15X-K-KILOGA---O_PROD"), parent = root)
  newXMLNode("MessageVersion", attrs = c(v = version), parent = root) ## VERSION IS A PARAMETER !!
  newXMLNode("MessageType", attrs = c(v = "Z71"), parent = root)
  newXMLNode("ProcessType", attrs = c(v = "A17"), parent = root)
  newXMLNode("ScheduleClassificationType", attrs = c(v = "A01"), parent = root)
  newXMLNode("SenderIdentification", attrs = c(v = "15X-K-KILOGA---O", codingScheme = "A01"), parent = root)
  newXMLNode("SenderRole", attrs = c(v = "A20"), parent = root)
  newXMLNode("ReceiverIdentification", attrs = c(v = "15X-KATENERGIA-V", codingScheme = "A01"), parent = root)
  newXMLNode("ReceiverRole", attrs = c(v = "A06"), parent = root)
  newXMLNode("MessageDateTime", attrs = c(v = current_time), parent = root)  
  newXMLNode("ScheduleTimeInterval", attrs = c(v = forecast_interval), parent = root) 
  ScheduleTimeSeries <- newXMLNode("ScheduleTimeSeries", parent = root)  
  newXMLNode("SendersTimeSeriesIdentification", attrs = c(v = "KILOGAMM-NAP-31"), parent = ScheduleTimeSeries)
  newXMLNode("SendersTimeSeriesVersion", attrs = c(v = version), parent = ScheduleTimeSeries) # VERSION IS A PARAMETER !!
  newXMLNode("BusinessType", attrs = c(v = "A01"), parent = ScheduleTimeSeries) 
  newXMLNode("Product", attrs = c(v = "8716867000016"), parent = ScheduleTimeSeries)
  newXMLNode("ObjectAggregation", attrs = c(v = "A02"), parent = ScheduleTimeSeries)
  newXMLNode("InArea", attrs = c(v = "10YHU-MAVIR----U", codingScheme = "A01"), parent = ScheduleTimeSeries)
  newXMLNode("MeteringPointIdentification", attrs = c(v = "HU000120B11-U-NAPE-SZIGVA-0400-32", codingScheme = "A01"), parent = ScheduleTimeSeries)
  newXMLNode("InParty", attrs = c(v = "15X-K-KILOGA---O", codingScheme = "A01"), parent = ScheduleTimeSeries)
  newXMLNode("MeasurementUnit", attrs = c(v = "KWT"), parent = ScheduleTimeSeries)
  Period <- newXMLNode("Period", parent = ScheduleTimeSeries)  
  newXMLNode("TimeInterval", attrs = c(v = forecast_interval), parent = Period)
  newXMLNode("Resolution", attrs = c(v = "PT15M"), parent = Period)
  
  for (j in 1:nrow(datau_w_prediction)){
    IntervalNode = newXMLNode("Interval", parent=Period)  
    newXMLNode("Pos", attrs = c(v = datau_w_prediction_forXML$Pos[j]), parent=IntervalNode)  
    newXMLNode("Qty", attrs = c(v = datau_w_prediction_forXML$Qty[j]), parent=IntervalNode)  
  }  
  
  saveXML(doc, file = paste0(dir_upl, "Feltoltesre_Benedek/masverziok/15X-K-KILOGA---O_", 
                             gsub("-", "", forecast_date), "_", version,
                             ".xml"), encoding = "UTF-8", prefix = '<?xml version="1.0" encoding="UTF-8"?>\n')
  
  
  
  ## Verzio 3
  
  ## Parameter definition
  forecast_start_utc <- as.character(with_tz(as.POSIXct(paste0(forecast_date - days(1), "24:00")), "UTC")) # Possible parameter
  forecast_end_utc <- as.character(with_tz(as.POSIXct(paste0(forecast_date, "24:00")), "UTC")) # Possible parameter
  
  
  
  forecast_date_char <- as.character(forecast_date)
  ### Version automatic definition
  
  #listing files
  filesdf <- data.frame(filenames = list.files(paste0(dir_upl, "Feltoltesre_Benedek")))
  #from list to data.frame
  filesdf <- data.frame(t(data.frame(str_split(filesdf$filenames, "_"))))
  row.names(filesdf) <- NULL
  
  #renaming variables
  colnames(filesdf) <- c("pp", "date", "version")
  
  #version variable cleaning
  filesdf$version <- gsub("\\..*","",filesdf$version)
  filesdf <- filesdf %>% filter(date == gsub("-", "", forecast_date_char))
  
  version <- 3
  
  
  ## Version check
  # read.xlsx(LotsofmydatainExcel.xlsm,sheetName="Deaths",as.data.frame=TRUE)
  
  
  forecast_interval <- gsub(" ", "T", paste0(substr(forecast_start_utc, 1, nchar(forecast_start_utc)-3), "Z/", 
                                             substr(forecast_end_utc, 1, nchar(forecast_end_utc)-3),"Z"))
  current_time <- gsub(" ", "T", paste0(with_tz(Sys.time(), tzone = "CET"), "Z"))
  
  
  datau_w_prediction_forXML <- data.frame(Pos = 1:nrow(datau_w_prediction), Qty = datau_w_prediction$predicted_P)
  
  
  ## XML Construction
  doc <- newXMLDoc()
  root <- newXMLNode("ScheduleMessage",
                     attrs = c(DtdRelease = 3 , DtdVersion = 3), 
                     doc=doc)
  
  newXMLNode("MessageIdentification", attrs = c(v = "15X-K-KILOGA---O_PROD"), parent = root)
  newXMLNode("MessageVersion", attrs = c(v = version), parent = root) ## VERSION IS A PARAMETER !!
  newXMLNode("MessageType", attrs = c(v = "Z71"), parent = root)
  newXMLNode("ProcessType", attrs = c(v = "A17"), parent = root)
  newXMLNode("ScheduleClassificationType", attrs = c(v = "A01"), parent = root)
  newXMLNode("SenderIdentification", attrs = c(v = "15X-K-KILOGA---O", codingScheme = "A01"), parent = root)
  newXMLNode("SenderRole", attrs = c(v = "A20"), parent = root)
  newXMLNode("ReceiverIdentification", attrs = c(v = "15X-KATENERGIA-V", codingScheme = "A01"), parent = root)
  newXMLNode("ReceiverRole", attrs = c(v = "A06"), parent = root)
  newXMLNode("MessageDateTime", attrs = c(v = current_time), parent = root)  
  newXMLNode("ScheduleTimeInterval", attrs = c(v = forecast_interval), parent = root) 
  ScheduleTimeSeries <- newXMLNode("ScheduleTimeSeries", parent = root)  
  newXMLNode("SendersTimeSeriesIdentification", attrs = c(v = "KILOGAMM-NAP-31"), parent = ScheduleTimeSeries)
  newXMLNode("SendersTimeSeriesVersion", attrs = c(v = version), parent = ScheduleTimeSeries) # VERSION IS A PARAMETER !!
  newXMLNode("BusinessType", attrs = c(v = "A01"), parent = ScheduleTimeSeries) 
  newXMLNode("Product", attrs = c(v = "8716867000016"), parent = ScheduleTimeSeries)
  newXMLNode("ObjectAggregation", attrs = c(v = "A02"), parent = ScheduleTimeSeries)
  newXMLNode("InArea", attrs = c(v = "10YHU-MAVIR----U", codingScheme = "A01"), parent = ScheduleTimeSeries)
  newXMLNode("MeteringPointIdentification", attrs = c(v = "HU000120B11-U-NAPE-SZIGVA-0400-32", codingScheme = "A01"), parent = ScheduleTimeSeries)
  newXMLNode("InParty", attrs = c(v = "15X-K-KILOGA---O", codingScheme = "A01"), parent = ScheduleTimeSeries)
  newXMLNode("MeasurementUnit", attrs = c(v = "KWT"), parent = ScheduleTimeSeries)
  Period <- newXMLNode("Period", parent = ScheduleTimeSeries)  
  newXMLNode("TimeInterval", attrs = c(v = forecast_interval), parent = Period)
  newXMLNode("Resolution", attrs = c(v = "PT15M"), parent = Period)
  
  for (j in 1:nrow(datau_w_prediction)){
    IntervalNode = newXMLNode("Interval", parent=Period)  
    newXMLNode("Pos", attrs = c(v = datau_w_prediction_forXML$Pos[j]), parent=IntervalNode)  
    newXMLNode("Qty", attrs = c(v = datau_w_prediction_forXML$Qty[j]), parent=IntervalNode)  
  }  
  
  saveXML(doc, file = paste0(dir_upl, "Feltoltesre_Benedek/masverziok/15X-K-KILOGA---O_", 
                             gsub("-", "", forecast_date), "_", version,
                             ".xml"), encoding = "UTF-8", prefix = '<?xml version="1.0" encoding="UTF-8"?>\n')
  
} 
forecast_date <- "2019-03-24"
forecast_gamma("2019-04-14")
rm(forecast_date, forecast_date_char)
 