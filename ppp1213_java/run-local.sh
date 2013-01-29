#!/bin/bash

POOLSIZE=8
POOLNAME=pool-$(date +%Y%m%d%H%M%S)-$(date +%N)

bin/java-run -agentlib:jdwp=transport=dt_socket,address=8000,server=y,suspend=n -ea -Dibis.pool.name=$POOLNAME -Dibis.pool.size=$POOLSIZE -Dibis.server.address=localhost:4321 rubiks.ipl.Rubiks &
sleep 0.5
for (( i=0; i < $((POOLSIZE - 1)); i++ )); do
	bin/java-run -ea -Dibis.pool.name=$POOLNAME -Dibis.pool.size=$POOLSIZE -Dibis.server.address=localhost:4321 rubiks.ipl.Rubiks &
done

wait
