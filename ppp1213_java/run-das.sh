#!/bin/bash

POOLSIZE=32

prun -v -1 -np $POOLSIZE bin/java-run -Dibis.pool.name=test -Dibis.pool.size=2 -Dibis.server.address=fs0:4321 rubiks.ipl.Rubiks
