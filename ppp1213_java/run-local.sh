#!/bin/bash

POOLSIZE=8

bin/java-run -agentlib:jdwp=transport=dt_socket,address=8000,server=y,suspend=n -ea -Dibis.pool.name=test -Dibis.pool.size=$POOLSIZE -Dibis.server.address=localhost:4321 rubiks.ipl.Rubiks &
sleep 0.5
for (( i=0; i < $((POOLSIZE - 1)); i++ )); do
	bin/java-run -ea -Dibis.pool.name=test -Dibis.pool.size=$POOLSIZE -Dibis.server.address=localhost:4321 rubiks.ipl.Rubiks &
done

wait
