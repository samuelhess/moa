#!/usr/bin/env bash 

#
#
#
fname=$1 
rare=$2
ratio=$3

if [ ! -d "tmp/"]; then
  mkdir tmp
fi 

sed -n "1,/\@data$/p" $fname > tmp/header.txt
sed "1,/\@data$/d" $fname \
  | sed -e "s/,/\, /g" \
  | awk -v ratio=$ratio -v class=$rare \
    '{if ($NF==class) {if(rand()<ratio){print $0} } else {print $0}}' \
    > tmp/data.txt

new_name=`echo "$fname" | sed -e "s/.arff//g" -e "s/$/-imbal-$ratio.arff/g"`
paste tmp/header.txt  tmp/data.txt > $new_name
