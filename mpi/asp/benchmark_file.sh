#!/bin/bash
results=results_file
for file in ../sample_graphs/test1.gr.new ../sample_graphs/test2.gr.new ../sample_graphs/test3.gr.new ../sample_graphs/test4.gr.new; do
	echo $file >> $results
	echo seq >> $results
	for i in {1..3}; do
		./run_asp_seq -file $file 2>&1 | grep "ASP took" | cut -f8 -d" " >> $results
	done
	for i in 1 2 4 8 16; do
		echo par $i >> $results
		for j in {1..3}; do
			./run_asp_par -file $file $i 2>&1 | grep "ASP took" | cut -f8 -d" " >> $results
		done
	done
done
