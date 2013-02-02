#!/bin/bash
results=results_random
for size in 3000; do
	echo random $size >> $results
	echo seq >> $results
	for i in {1..3}; do
		./run_asp_seq -random $size 0 2>&1 | grep "ASP took" | awk '{print $3}' >> $results
	done
	for i in 1 2 4 8 16; do
		echo par $i >> $results
		for j in {1..3}; do
			./run_asp_par -random $size 0 $i 2>&1 | grep "ASP took" | awk '{print $3}' >> $results
		done
	done
done
