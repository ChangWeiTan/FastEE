#!/bin/bash 
usage() { echo "Usage: $0 [-p <Dataset Name>] [-c <EE|LbEE|FastEE|ApproxEE>] [-d <Dataset Directory>] [-o <Output Directory>] [-s <Number of Samples>] [-r <Number of Runs>]" 1>&2; exit; }

PROJECTDIR=$PWD/
OUTPUTDIR=$PWD"/output/individual/"
DATASETDIR="/home/ubuntu/workspace/Dataset/TSC_Problems/"
if [ ! -d "$DATASETDIR" ]; then
      DATASETDIR="/mnt/c/Users/cwtan/workspace/Dataset/TSC_Problems/"
fi
if [ ! -d "$DATASETDIR" ]; then
      DATASETDIR="/mnt/lustre/projects/ud82/changt/workspace/Dataset/TSC_Problems/"
fi

DISTANCES=("Euclidean" "DTW_Rn" "DTW_R1" "WDTW" "WDDTW" "DDTW_Rn" "DDTW_R1" "LCSS" "MSM" "TWE" "ERP")
PROBLEM="ArrowHead"
CLASSIFIER="FastEE"
NSAMPLES=2
NRUNS=1
while getopts p:c:d:o:s:r:h: option; do
      case "${option}" in
            p)    PROBLEM=${OPTARG};;
            c)    CLASSIFIER=${OPTARG};;
            d)    DATASETDIR=${OPTARG};;
            o)    OUTPUTDIR=${OPTARG};;
            s)    NSAMPLES=${OPTARG};;
            r)    NRUNS=${OPTARG};;
            h)    usage;;
      esac
done
shift $((OPTIND-1))

if [ ! -d bin ]; then
      mkdir bin
fi

javac -sourcepath src -d bin -cp $PWD/lib/*: src/**/*.java

cd bin 
echo Current Directory: $PWD
echo Dataset Directory: $DATASETDIR
echo Output Directory:  $OUTPUTDIR
echo Problem:           $PROBLEM
echo Classifier:        $CLASSIFIER
echo nSamples:          $NSAMPLES
echo nRuns:             $NRUNS
echo 


for distance in "${DISTANCES[@]}"; do
      if [ $CLASSIFIER = "EE" ]; then
            java -Xmx14g -Xms14g -cp $PROJECTDIR/lib/*: experiments.IndividualClassifierEE $OUTPUTDIR $DATASETDIR $PROBLEM $distance
      elif [ $CLASSIFIER = "LbEE" ]; then   
            java -Xmx14g -Xms14g -cp $PROJECTDIR/lib/*: experiments.IndividualClassifierLbEE $OUTPUTDIR $DATASETDIR $PROBLEM $distance
      elif [ $CLASSIFIER = "FastEE" ]; then   
            java -Xmx14g -Xms14g -cp $PROJECTDIR/lib/*: experiments.IndividualClassifierFastEE $OUTPUTDIR $DATASETDIR $PROBLEM $distance
      elif [ $CLASSIFIER = "ApproxEE" ]; then   
            java -Xmx14g -Xms14g -cp $PROJECTDIR/lib/*: experiments.IndividualClassifierApproxEE $OUTPUTDIR $DATASETDIR $PROBLEM $distance $NSAMPLES $NRUNS
      else
            echo $CLASSIFIER Invalid, try EE, LbEE, FastEE, or ApproxEE
      fi
done