# FastEE
FastEE: Fast Ensembles of Elastic Distances
This is the source code for FastEE - Faster version of the Ensembles of Elastic Distances (EE).
In particular, FastEE tackles the long training time of EE.
This code only focus on training EE. 

## Running the code:
Running from terminal
1. Training the individual classifers
* java -Xmx14g -Xms14g -cp $LIBDIR: experiments.IndividualClassifierEE $OUTPUTDIR $DATASETDIR $PROBLEM $DISTANCE
* java -Xmx14g -Xms14g -cp $LIBDIR: experiments.IndividualClassifierLbEE $OUTPUTDIR $DATASETDIR $PROBLEM $DISTANCE
* java -Xmx14g -Xms14g -cp $LIBDIR: experiments.IndividualClassifierFastEE $OUTPUTDIR $DATASETDIR $PROBLEM $DISTANCE
* java -Xmx14g -Xms14g -cp $LIBDIR: experiments.IndividualClassifierApproxEE $OUTPUTDIR $DATASETDIR $PROBLEM $DISTANCE $NSAMPLES $NRUNS

2. Training the whole ensemble
* java -Xmx14g -Xms14g -cp $LIBDIR: experiments.TrainElasticEnsembles $OUTPUTDIR $DATASETDIR $PROBLEM $CLASSIFIER $NSAMPLES

Running from Bash Script
1. bash TrainIndividualClasssifiers.sh [-p <Dataset_Name>] [-c <EE|LbEE|FastEE|ApproxEE>] [-d <Dataset_Directory>] [-o <Output_Directory>] [-s <Number_of_Samples>] [-r <Number_of_Runs>]
2. bash TrainEnsembles.sh [-p <Dataset_Name>] [-c <EE|LbEE|FastEE|ApproxEE>] [-d <Dataset_Directory>] [-o <Output_Directory>] [-s <Number_of_Samples>]
