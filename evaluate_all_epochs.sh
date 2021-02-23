#!/bin/bash

: '
This code evaluates the performance of the model for all epochs.
Then it runs the code that find the best validation epoch and uses it to calculate the performance of the model.

To run the code for interaction prediction on the reddit dataset: 
$ ./evaluate_all_epochs.sh reddit interaction

To run the code for state change prediction on the reddit dataset: 
$ ./evaluate_all_epochs.sh reddit state
'

network=$1
type=$2
gpu=$3
idx=${4-0}
interaction="interaction"

while [ $idx -le 9 ]
do
    echo $idx
    if [ $type == "$interaction" ]; then
	python2.7 evaluate_interaction_prediction.py --network $network --model jodie-mult --epoch ${idx} --device cpu
    else
	python2.7 evaluate_state_change_prediction.py --network $network --model jodie --epoch ${idx} --gpu ${gpu}
    fi
    (( idx+=1 ))
done 


if [ $type == "$interaction" ]; then
    python get_final_performance_numbers.py results/interaction_prediction_${network}.txt
else
    python get_final_performance_numbers.py results/state_change_prediction_${network}.txt
fi
