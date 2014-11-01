data_sets <- c("abalone_v2","airlines","breast-w","car_v2","cmc","cmc_v2",
               "colic","cov_v2","credit-a","diabetes","dow","elecNormNew",
               "german","haberman","hepatitis","hyper","image","magic",
               "noaa","poker","sea","sick","spam","splice_v2")
data_sets <- c("abalone_v2")
base_clfr <- "bayes"

for (data_set in data_sets) {
  # load the files that moa outputs for each experiment. to make life a bit easier, 
  # we are going to save new csv files that can easily loaded for plotting
  base_df <- read.csv(paste("../outputs/results-", data_set, "-", base_clfr, ".csv", sep=""))
  bag_df <- read.csv(paste("../outputs/results-", data_set, "-bagging-", base_clfr, ".csv", sep=""))
  baga_df <- read.csv(paste("../outputs/results-", data_set, "-baggingAdwin-", base_clfr, ".csv", sep=""))
  boo_df <- read.csv(paste("../outputs/results-", data_set, "-boosting-", base_clfr, ".csv", sep=""))
  booa_df <- read.csv(paste("../outputs/results-", data_set, "-boostingAdwin-", base_clfr, ".csv", sep=""))
  
  p1_df <- read.csv(paste("../outputs/results-", data_set, "-pame1-", base_clfr, ".csv", sep=""))
  p1a_df <- read.csv(paste("../outputs/results-", data_set, "-pame1adwin-", base_clfr, ".csv", sep=""))
  p2_df <- read.csv(paste("../outputs/results-", data_set, "-pame2-", base_clfr, ".csv", sep=""))
  p2a_df <- read.csv(paste("../outputs/results-", data_set, "-pame2adwin-", base_clfr, ".csv", sep=""))
  p3_df <- read.csv(paste("../outputs/results-", data_set, "-pame3-", base_clfr, ".csv", sep=""))
  p3a_df <- read.csv(paste("../outputs/results-", data_set, "-pame3adwin-", base_clfr, ".csv", sep=""))
  
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
  write.csv(acc_df, file=paste("../outputs/clean-",data_set, "-", base_clfr, ".csv", sep=""), row.names=F)
}
