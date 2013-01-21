#!/bin/bash

POOLSIZE=2

for (( i=0; i < $POOLSIZE; i++ )); do
	bin/java-run -Dibis.pool.name=test -Dibis.pool.size=$POOLSIZE -Dibis.server.address=localhost:4321 rubiks.ipl.Rubiks &
done

wait
