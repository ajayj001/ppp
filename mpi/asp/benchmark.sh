#!/bin/bash
for file in ../sample_graphs/test1.gr.new ../sample_graphs/test2.gr.new ../sample_graphs/test3.gr.new ../sample_graphs/test4.gr.new; do
	echo $file >> results
	echo seq >> results
	for i in {1..3}; do
		./run-asp-seq $file 2>&1 | grep "ASP took" | cut -f8 -d" " >> results
	done
	for i in 1 2 4 8 12 16; do
		echo par $i >> results
		for j in {1..3}; do
			./run-asp-par -file $file $i 2>&1 | grep "ASP took" | cut -f8 -d" " >> results
		done
	done
done
