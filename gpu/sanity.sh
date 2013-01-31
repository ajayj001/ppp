#!/bin/bash
for image in images/*
do
	./check.sh ${image} | grep threshold
done
