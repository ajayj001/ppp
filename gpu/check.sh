#!/bin/bash
image=$1
prun -np 1 -native '-l gpu=GTX480' bin/cuda ${image}
mv smooth.bmp smooth_cuda.bmp
prun -np 1 bin/sequential ${image}
prun -np 1 bin/compare smooth.bmp smooth_cuda.bmp
