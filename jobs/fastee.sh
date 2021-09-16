#!/bin/bash
#SBATCH --job-name=FastEE
#SBATCH --account=nc23
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16000
#SBATCH --cpus-per-task=32

module load jdk/14
cd ..
javac -sourcepath src -cp "lib/*" -d bin src/experiments/TrainingTimeBenchmark.java

cd bin
java -Xmx14g -Xms14g -cp "../lib/*": experiments.TrainingTimeBenchmark -machine="m3" -problem="all" -classifier="FastEE" -paramId=-1 -cpu=-1 -verbose=0 -iter=0 -trainOpts=2


