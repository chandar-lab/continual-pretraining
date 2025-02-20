#!/bin/bash

"Starting script with PID $$" >> nohup.out


i=0

while [ $i -lt 10 ]; do
	echo "Iteration $i:"
	salloc -A bif151 -N 32  -t 2:00:00 -q debug <<EOF >> 510M_joint_${i}.log 2>&1
	source joint_training_script.sh
EOF
	i=$((i + 1))
        sleep 20000
done

