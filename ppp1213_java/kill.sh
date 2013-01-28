#!/bin/bash
for pid in `ps | grep Rubiks | cut -f1 -d" "`
do
	kill $pid
done
