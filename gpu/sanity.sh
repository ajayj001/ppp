#!/bin/bash
for image in images/*
do
	./check.sh ${image}
done
