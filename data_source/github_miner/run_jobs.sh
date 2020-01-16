#!/bin/bash
 total=100
 step=$(($total/8))
 for a in $(seq 0 $step $total)
 do
    command="python runner.py $a  $step"
     # ($command &)
    # echo $command
    python runner.py $a  $step &
    sleep 1
 done