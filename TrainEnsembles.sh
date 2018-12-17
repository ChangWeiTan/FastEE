#!/bin/bash 
usage() { echo "Usage: $0 [-p <Dataset Name>] [-c <EE|LbEE|FastEE|ApproxEE>] [-d <Dataset Directory>] [-o <Output Directory>] [-s <Number of Samples>]" 1>&2; exit; }

PROJECTDIR=$PWD/
OUTPUTDIR=$PWD"/output/Ensemble/"
DATASETDIR="/home/ubuntu/workspace/Dataset/TSC_Problems/"
if [ ! -d "$DATASETDIR" ]; then
      DATASETDIR="/mnt/c/Users/cwtan/workspace/Dataset/TSC_Problems/"
fi
if [ ! -d "$DATASETDIR" ]; then
      DATASETDIR="/mnt/lustre/projects/ud82/changt/workspace/Dataset/TSC_Problems/"
fi

PROBLEM="ArrowHead"
CLASSIFIER="FastEE"
NSAMPLES=2
while getopts p:c:d:o:s:h: option; do
      case "${option}" in
            p)    PROBLEM=${OPTARG};;
            c)    CLASSIFIER=${OPTARG};;
            d)    DATASETDIR=${OPTARG};;
            o)    OUTPUTDIR=${OPTARG};;
            s)    NSAMPLES=${OPTARG};;
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


java -Xmx14g -Xms14g -cp $PROJECTDIR/lib/*: experiments.TrainElasticEnsembles $OUTPUTDIR $DATASETDIR $PROBLEM $CLASSIFIER $NSAMPLES