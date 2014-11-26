library("reshape2")
library("ggplot2")

data_sets <- c("abalone_v2","airlines","breast-w","car_v2","cmc","cmc_v2",
              "colic","cov_v2","credit-a","diabetes","dow","elecNormNew",
              "german","haberman","hepatitis","hyper","image","magic",
              "noaa","sea","sick","spam","splice_v2")
base_clfr <- "hoeff"
scale <- 1.5

for (data_set in data_sets) {
  df_fp <- paste("~/Git/MassiveOnlineAnalysis/experiments/outputs/clean-accuracy-",data_set, "-", base_clfr, ".csv", sep="")
  df <- read.csv(df_fp)
  df_ds <- melt(df, id.vars="time", 
                value.name="accuracy", 
                measure.vars=c("bagging","boosting","pame1","pame2","pame3"))
  df_cd <- melt(df, id.vars="time", 
                value.name="accuracy", 
                measure.vars=c("bagging_adwin","boosting_adwin","pame1adwin","pame2adwin","pame3adwin"))
  g1 <- ggplot(data=df_ds, aes(x=time, y=accuracy, color=variable)) + geom_smooth(size=1) + theme(legend.justification=c(1,0), legend.position=c(1,0), axis.text.x = element_text(size = rel(scale)), axis.text.y = element_text(size = rel(scale)), axis.title.x = element_text(size = rel(scale)), axis.title.y = element_text(size = rel(scale)), legend.text=element_text(size=rel(scale)), legend.title=element_text(size=rel(0))) #+ ylim(60,101)
  ggsave(g1, file=paste("~/Git/MassiveOnlineAnalysis/experiments/plots/clean-accuracy-stat-",data_set, "-", base_clfr, ".pdf", sep=""))
  g2 <- ggplot(data=df_cd, aes(x=time, y=accuracy, color=variable)) + geom_smooth(size=1) + theme(legend.justification=c(1,0), legend.position=c(1,0), axis.text.x = element_text(size = rel(scale)), axis.text.y = element_text(size = rel(scale)), axis.title.x = element_text(size = rel(scale)), axis.title.y = element_text(size = rel(scale)), legend.text=element_text(size=rel(scale)), legend.title=element_text(size=rel(0))) #+ ylim(60,101)
  ggsave(g2, file=paste("~/Git/MassiveOnlineAnalysis/experiments/plots/clean-accuracy-cd-",data_set, "-", base_clfr, ".pdf", sep=""))
  
  df_fp <- paste("~/Git/MassiveOnlineAnalysis/experiments/outputs/clean-kappa-",data_set, "-", base_clfr, ".csv", sep="")
  df <- read.csv(df_fp)
  df_ds <- melt(df, id.vars="time", 
                value.name="kappa", 
                measure.vars=c("bagging","boosting","pame1","pame2","pame3"))
  df_cd <- melt(df, id.vars="time", 
                value.name="kappa", 
                measure.vars=c("bagging_adwin","boosting_adwin","pame1adwin","pame2adwin","pame3adwin"))
  g1 <- ggplot(data=df_ds, aes(x=time, y=kappa, color=variable)) + geom_smooth(size=1) + theme(legend.justification=c(1,0), legend.position=c(1,0), axis.text.x = element_text(size = rel(scale)), axis.text.y = element_text(size = rel(scale)), axis.title.x = element_text(size = rel(scale)), axis.title.y = element_text(size = rel(scale)), legend.text=element_text(size=rel(scale)), legend.title=element_text(size=rel(0))) #+ ylim(60,101)
  ggsave(g1, file=paste("~/Git/MassiveOnlineAnalysis/experiments/plots/clean-kappa-stat-",data_set, "-", base_clfr, ".pdf", sep=""))
  g2 <- ggplot(data=df_cd, aes(x=time, y=kappa, color=variable)) + geom_smooth(size=1) + theme(legend.justification=c(1,0), legend.position=c(1,0), axis.text.x = element_text(size = rel(scale)), axis.text.y = element_text(size = rel(scale)), axis.title.x = element_text(size = rel(scale)), axis.title.y = element_text(size = rel(scale)), legend.text=element_text(size=rel(scale)), legend.title=element_text(size=rel(0))) #+ ylim(60,101)
  ggsave(g2, file=paste("~/Git/MassiveOnlineAnalysis/experiments/plots/clean-kappa-cd-",data_set, "-", base_clfr, ".pdf", sep=""))
  
  df_fp <- paste("~/Git/MassiveOnlineAnalysis/experiments/outputs/clean-rmhrs-",data_set, "-", base_clfr, ".csv", sep="")
  df <- read.csv(df_fp)
  df_ds <- melt(df, id.vars="time", 
                value.name="RAMHours", 
                measure.vars=c("bagging","boosting","pame1","pame2","pame3"))
  df_cd <- melt(df, id.vars="time", 
                value.name="RAMHours", 
                measure.vars=c("bagging_adwin","boosting_adwin","pame1adwin","pame2adwin","pame3adwin"))
  g1 <- ggplot(data=df_ds, aes(x=time, y=RAMHours, color=variable)) + geom_smooth(size=1) + theme(legend.justification=c(1,0), legend.position=c(1,0), axis.text.x = element_text(size = rel(scale)), axis.text.y = element_text(size = rel(scale)), axis.title.x = element_text(size = rel(scale)), axis.title.y = element_text(size = rel(scale)), legend.text=element_text(size=rel(scale)), legend.title=element_text(size=rel(0))) #+ ylim(60,101)
  ggsave(g1, file=paste("~/Git/MassiveOnlineAnalysis/experiments/plots/clean-rmhrs-stat-",data_set, "-", base_clfr, ".pdf", sep=""))
  g2 <- ggplot(data=df_cd, aes(x=time, y=RAMHours, color=variable)) + geom_smooth(size=1) + theme(legend.justification=c(1,0), legend.position=c(1,0), axis.text.x = element_text(size = rel(scale)), axis.text.y = element_text(size = rel(scale)), axis.title.x = element_text(size = rel(scale)), axis.title.y = element_text(size = rel(scale)), legend.text=element_text(size=rel(scale)), legend.title=element_text(size=rel(0))) #+ ylim(60,101)
  ggsave(g2, file=paste("~/Git/MassiveOnlineAnalysis/experiments/plots/clean-rmhrs-cd-",data_set, "-", base_clfr, ".pdf", sep=""))
}

