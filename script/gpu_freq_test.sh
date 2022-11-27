#!/bin/bash


# while getopts c:a:f: flag
while getopts c:a:f: flag
do
    case "${flag}" in
        c) cmd=${OPTARG};;
    esac
done

for (( i = 0; i <= 4000; i += 500 )); do
    echo -n GPUMemoryTransferRateOffset
    echo -$i
    sudo nvidia-settings -c :0 -a GPUMemoryTransferRateOffset[4]=-$i #1>>/dev/null
    for (( j = 0; j <= 1000; j += 5000)); do
        # echo $i $j 1>&2
        echo -n GPUGraphicsClockOffset
        echo -$j
        sudo nvidia-settings -c :0 -a GPUGraphicsClockOffset[4]=-$j #1>>/dev/null
        sleep 5
        $cmd

    done
done


