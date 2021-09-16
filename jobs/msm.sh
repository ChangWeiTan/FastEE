#!/bin/bash
#SBATCH --job-name=MSM
#SBATCH --account=nc23
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=m3i

module load jdk/14
cd ..
javac -sourcepath src -cp "lib/*" -d bin src/experiments/TrainingTimeBenchmark.java

cd bin
java -Xmx14g -Xms14g -cp "../lib/*": experiments.TrainingTimeBenchmark -machine="m3" -problem="all" -classifier="MSM1NN" -paramId=-1 -cpu=-1 -verbose=0 -iter=0 -trainOpts=2


