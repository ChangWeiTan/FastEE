#!/bin/bash
#SBATCH --job-name=DTW
#SBATCH --account=nc23
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16000
#SBATCH --cpus-per-task=32

module load jdk/14
cd ..
javac -sourcepath src -cp "lib/*" -d bin src/experiments/TrainingTimeBenchmark.java

cd bin
java -Xmx14g -Xms14g -cp "../lib/*": experiments.TrainingTimeBenchmark -machine="m3" -problem="all" -classifier="DTW1NN" -paramId=-1 -cpu=-1 -verbose=0 -iter=0 -trainOpts=2


