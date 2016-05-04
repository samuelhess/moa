#!/usr/bin/env bash 

## remove any old files that may exist in the path
#rm outputs/*.csv


#JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.7.0_21.jdk/Contents/Home
#PATH=$JAVA_HOME/bin:$PATH


## set up some path locations
# home_fp=/data/gditzler
# home_fp=/Users/gditzler
home_fp=/scratch/ditzler

data_fp=${home_fp}/Git/MassiveOnlineAnalysis/experiments/data
moa_fp=${home_fp}/Git/MassiveOnlineAnalysis/runtime
lib_fp=${home_fp}/Git/MassiveOnlineAnalysis/moa/lib
out_fp=${home_fp}/Git/MassiveOnlineAnalysis/experiments/outputs

## get the datasets & defaults
datasets=$(ls ${data_fp}/*.arff)
base_clfr=$(echo "bayes.NaiveBayes")
base_short=bayes
#base_clfr=$(echo "trees.HoeffdingTree -b")
#base_short=hoeff

for dataset in ${datasets[@]}; do 
  data_short=$(echo $dataset | sed -e "s/.*\/\(.*\)\.arff/\1/g")

  f=$(wc ${dataset} | awk '{print $1/100}' | sed -e "s/^\([0-9]*\).[0-9]*/\1/g")
  
  # base 
  java -cp ${moa_fp}/moac.jar \
    -javaagent:${lib_fp}/sizeofag-1.0.0.jar \
    moa.DoTask "EvaluatePrequential \
    -l (${base_clfr})\
    -s (ArffFileStream -f ${dataset}) \
    -f $f" \
    > ${out_fp}/results-${data_short}-${base_short}.csv
  
  # ozabag
  java -cp ${moa_fp}/moac.jar \
    -javaagent:${lib_fp}/sizeofag-1.0.0.jar \
    moa.DoTask "EvaluatePrequential \
    -l (meta.OzaBag -l ($base_clfr))\
    -s (ArffFileStream -f ${dataset}) \
    -f $f" \
    > ${out_fp}/results-${data_short}-bagging-${base_short}.csv

  # ozaboost
  java -cp ${moa_fp}/moac.jar \
    -javaagent:${lib_fp}/sizeofag-1.0.0.jar \
    moa.DoTask "EvaluatePrequential \
    -l (meta.OzaBoost -l ($base_clfr))\
    -s (ArffFileStream -f ${dataset}) \
    -f $f" \
    > ${out_fp}/results-${data_short}-boosting-${base_short}.csv

  # ozabag (adwin)
  java -cp ${moa_fp}/moac.jar \
    -javaagent:${lib_fp}/sizeofag-1.0.0.jar \
    moa.DoTask "EvaluatePrequential \
    -l (meta.OzaBagAdwin -l ($base_clfr))\
    -s (ArffFileStream -f ${dataset}) \
    -f $f" \
    > ${out_fp}/results-${data_short}-baggingAdwin-${base_short}.csv

  # ozaboost (adwin)
  java -cp ${moa_fp}/moac.jar \
    -javaagent:${lib_fp}/sizeofag-1.0.0.jar \
    moa.DoTask "EvaluatePrequential \
    -l (meta.OzaBoostAdwin -l ($base_clfr))\
    -s (ArffFileStream -f ${dataset}) \
    -f $f" \
    > ${out_fp}/results-${data_short}-boostingAdwin-${base_short}.csv

  # pame-1 bagging
  java -cp ${moa_fp}/moac.jar \
    -javaagent:${lib_fp}/sizeofag-1.0.0.jar \
    moa.DoTask "EvaluatePrequential \
    -l (meta.PAME -a Bagging -C 2.0 -l ($base_clfr))\
    -s (ArffFileStream -f ${dataset}) \
    -f $f" \
    > ${out_fp}/results-${data_short}-pame1-bag-${base_short}.csv
  
  # pame-2 bagging
  java -cp ${moa_fp}/moac.jar \
    -javaagent:${lib_fp}/sizeofag-1.0.0.jar \
    moa.DoTask "EvaluatePrequential \
    -l (meta.PAME -a Bagging -C 2.0 -l ($base_clfr) -u PAME-II)\
    -s (ArffFileStream -f ${dataset}) \
    -f $f" \
    > ${out_fp}/results-${data_short}-pame2-bag-${base_short}.csv
  
  # pame-3 - bagging
  java -cp ${moa_fp}/moac.jar \
    -javaagent:${lib_fp}/sizeofag-1.0.0.jar \
    moa.DoTask "EvaluatePrequential \
    -l (meta.PAME -a Bagging -C 2.0 -l ($base_clfr) -u PAME-III)\
    -s (ArffFileStream -f ${dataset}) \
    -f $f" \
    > ${out_fp}/results-${data_short}-pame3-bag-${base_short}.csv

  
  # pame-1 - boosting
  java -cp ${moa_fp}/moac.jar \
    -javaagent:${lib_fp}/sizeofag-1.0.0.jar \
    moa.DoTask "EvaluatePrequential \
    -l (meta.PAME -a Boosting -C 2.0 -l ($base_clfr))\
    -s (ArffFileStream -f ${dataset}) \
    -f $f" \
    > ${out_fp}/results-${data_short}-pame1-boo-${base_short}.csv
  
  # pame-2 - boosting
  java -cp ${moa_fp}/moac.jar \
    -javaagent:${lib_fp}/sizeofag-1.0.0.jar \
    moa.DoTask "EvaluatePrequential \
    -l (meta.PAME -a Boosting -C 2.0 -l ($base_clfr) -u PAME-II)\
    -s (ArffFileStream -f ${dataset}) \
    -f $f" \
    > ${out_fp}/results-${data_short}-pame2-boo-${base_short}.csv
  
  # pame-3 - boosting
  java -cp ${moa_fp}/moac.jar \
    -javaagent:${lib_fp}/sizeofag-1.0.0.jar \
    moa.DoTask "EvaluatePrequential \
    -l (meta.PAME -a Boosting -C 2.0 -l ($base_clfr) -u PAME-III)\
    -s (ArffFileStream -f ${dataset}) \
    -f $f" \
    > ${out_fp}/results-${data_short}-pame3-boo-${base_short}.csv



  # pame-1
  java -cp ${moa_fp}/moac.jar \
    -javaagent:${lib_fp}/sizeofag-1.0.0.jar \
    moa.DoTask "EvaluatePrequential \
    -l (meta.PAMEAdwin -l ($base_clfr))\
    -s (ArffFileStream -f ${dataset}) \
    -f $f" \
    > ${out_fp}/results-${data_short}-pame1adwin-${base_short}.csv
  
  # pame-2
  java -cp ${moa_fp}/moac.jar \
    -javaagent:${lib_fp}/sizeofag-1.0.0.jar \
    moa.DoTask "EvaluatePrequential \
    -l (meta.PAMEAdwin -l ($base_clfr) -u PAME-II)\
    -s (ArffFileStream -f ${dataset}) \
    -f $f" \
    > ${out_fp}/results-${data_short}-pame2adwin-${base_short}.csv
  
  # pame-3
  java -cp ${moa_fp}/moac.jar \
    -javaagent:${lib_fp}/sizeofag-1.0.0.jar \
    moa.DoTask "EvaluatePrequential \
    -l (meta.PAMEAdwin -l ($base_clfr) -u PAME-III)\
    -s (ArffFileStream -f ${dataset}) \
    -f $f" \
    > ${out_fp}/results-${data_short}-pame3adwin-${base_short}.csv

done

