
debug		:= 0
include		../Makefile.inc


all: $(BIN)/cuda
 
$(BIN)/cuda: filters.o main.cpp defines.h
	$(LINKER) -o $(BIN)/cuda main.cpp filters.o ../src/Timer.cpp $(INCLUDES) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS) 

filters.o: filters.cu defines.h
	$(NVCC) -c filters.cu $(NVCCFLAGS) $(INCLUDES)

clean:
	rm -f ./filters.o
	rm -f $(BIN)/cuda

