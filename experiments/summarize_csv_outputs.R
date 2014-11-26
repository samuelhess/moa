data_sets <- c("abalone_v2","airlines","breast-w","car_v2","cmc","cmc_v2",
               "colic","cov_v2","credit-a","diabetes","dow","elecNormNew",
               "german","haberman","hepatitis","hyper","image","magic",
               "noaa","poker","sea","sick","spam","splice_v2")
#data_sets <- c("haberman") #"haberman", "poker"
base_clfr <- "hoeff"

for (data_set in data_sets) {
  # load the files that moa outputs for each experiment. to make life a bit easier, 
  # we are going to save new csv files that can easily loaded for plotting
  base_df <- read.table(paste("../outputs/results-", data_set, "-", base_clfr, ".csv", sep=""), 
                        header=TRUE, sep=",", colClasses="numeric", na.strings="?")
  bag_df <- read.table(paste("../outputs/results-", data_set, "-bagging-", base_clfr, ".csv", sep=""), 
                       header=TRUE, sep=",", colClasses="numeric", na.strings="?")
  baga_df <- read.table(paste("../outputs/results-", data_set, "-baggingAdwin-", base_clfr, ".csv", sep=""), 
                        header=TRUE, sep=",", colClasses="numeric", na.strings="?")
  boo_df <- read.table(paste("../outputs/results-", data_set, "-boosting-", base_clfr, ".csv", sep=""), 
                       header=TRUE, sep=",", colClasses="numeric", na.strings="?")
  booa_df <- read.table(paste("../outputs/results-", data_set, "-boostingAdwin-", base_clfr, ".csv", sep=""), 
                        header=TRUE, sep=",", colClasses="numeric", na.strings="?")
  
  p1_df <- read.table(paste("../outputs/results-", data_set, "-pame1-", base_clfr, ".csv", sep=""), 
                      header=TRUE, sep=",", colClasses="numeric", na.strings="?")
  p1a_df <- read.table(paste("../outputs/results-", data_set, "-pame1adwin-", base_clfr, ".csv", sep=""),
                       header=TRUE, sep=",", colClasses="numeric", na.strings="?")
  p2_df <- read.table(paste("../outputs/results-", data_set, "-pame2-", base_clfr, ".csv", sep=""), 
                      header=TRUE, sep=",", colClasses="numeric", na.strings="?")
  p2a_df <- read.table(paste("../outputs/results-", data_set, "-pame2adwin-", base_clfr, ".csv", sep=""), 
                       header=TRUE, sep=",", colClasses="numeric", na.strings="?")
  p3_df <- read.table(paste("../outputs/results-", data_set, "-pame3-", base_clfr, ".csv", sep=""), 
                      header=TRUE, sep=",", colClasses="numeric", na.strings="?")
  p3a_df <- read.table(paste("../outputs/results-", data_set, "-pame3adwin-", base_clfr, ".csv", sep=""), 
                       header=TRUE, sep=",", colClasses="numeric", na.strings="?")
  
  # save the accuracies in a data frame
  acc_df <- data.frame(time=base_df$learning.evaluation.instances, 
                       bagging=bag_df$classifications.correct..percent.,
                       boosting=boo_df$classifications.correct..percent.,
                       bagging_adwin=baga_df$classifications.correct..percent.,
                       boosting_adwin=booa_df$classifications.correct..percent.,
                       pame1=p1_df$classifications.correct..percent.,
                       pame2=p2_df$classifications.correct..percent.,
                       pame3=p3_df$classifications.correct..percent.,
                       pame1adwin=p1a_df$classifications.correct..percent.,
                       pame2adwin=p2a_df$classifications.correct..percent.,
                       pame3adwin=p3a_df$classifications.correct..percent.)
  write.csv(acc_df, file=paste("../outputs/clean-accuracy-",data_set, "-", base_clfr, ".csv", sep=""), row.names=F)
  
  kappa_df <- data.frame(time=base_df$learning.evaluation.instances, 
                       bagging=bag_df$Kappa.Statistic..percent.,
                       boosting=boo_df$Kappa.Statistic..percent.,
                       bagging_adwin=baga_df$Kappa.Statistic..percent.,
                       boosting_adwin=booa_df$Kappa.Statistic..percent.,
                       pame1=p1_df$Kappa.Statistic..percent.,
                       pame2=p2_df$Kappa.Statistic..percent.,
                       pame3=p3_df$Kappa.Statistic..percent.,
                       pame1adwin=p1a_df$Kappa.Statistic..percent.,
                       pame2adwin=p2a_df$Kappa.Statistic..percent.,
                       pame3adwin=p3a_df$Kappa.Statistic..percent.)
  write.csv(kappa_df, file=paste("../outputs/clean-kappa-",data_set, "-", base_clfr, ".csv", sep=""), row.names=F)
  
  kappa_df <- data.frame(time=base_df$learning.evaluation.instances, 
                         bagging=bag_df$model.cost..RAM.Hours.,
                         boosting=boo_df$model.cost..RAM.Hours.,
                         bagging_adwin=baga_df$model.cost..RAM.Hours.,
                         boosting_adwin=booa_df$model.cost..RAM.Hours.,
                         pame1=p1_df$model.cost..RAM.Hours.,
                         pame2=p2_df$model.cost..RAM.Hours.,
                         pame3=p3_df$model.cost..RAM.Hours.,
                         pame1adwin=p1a_df$model.cost..RAM.Hours.,
                         pame2adwin=p2a_df$model.cost..RAM.Hours.,
                         pame3adwin=p3a_df$model.cost..RAM.Hours.)
  write.csv(kappa_df, file=paste("../outputs/clean-rmhrs-",data_set, "-", base_clfr, ".csv", sep=""), row.names=F)
}

acc_df4 <- melt(acc_df, id.vars="time", value.name="accuracy", id.vars=c("bagging","boosting","pame1"))
ggplot(data=acc_df3, aes(x=time, y=value, color=variable)) 
  + geom_line() 
  + theme(legend.justification=c(1,0), 
          legend.position=c(1,0),
          axis.text.x = element_text(size = rel(2)), 
          axis.text.y = element_text(size = rel(2)), 
          axis.title.x = element_text(size = rel(2)), 
          axis.title.y = element_text(size = rel(2)), 
          legend.text=element_text(size=rel(2)), 
          legend.title=element_text(size=rel(0))) 
  + ylim(60,101)
