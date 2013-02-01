#!/bin/bash
tar czf jlt230_JanvanderLugt_mpi_2.tar.gz mpi/asp/
tar czf jlt230_JanvanderLugt_gpu_2.tar.gz --exclude=gpu/images/* gpu/
cd java
	ant dist
	mv jlt230_Jan\ van\ der\ Lugt_2.zip ../jlt230_JanvanderLugt_java_2.zip
cd ..
