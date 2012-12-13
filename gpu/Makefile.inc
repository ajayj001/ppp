
PROJ_BASE	:= /home/jlt230/ppp/gpu
BIN		:= $(PROJ_BASE)/bin

INCLUDES	:= -I"$(PROJ_BASE)/include"
LIBS		:= 

CUDA_LIBS	:= $(LIBS) -L"$(CUDA_INSTALL_PATH)/lib64"

NVCCFLAGS	:= -arch compute_20 -code sm_20 --ptxas-options=-v
ifneq ($(debug), 1)
	CFLAGS		:= -O3 -g0 -w
	NVCCFLAGS	+= -O3
else
	CFLAGS		:= -O0 -g3 -Wall
	NVCCFLAGS	+= -O0
endif

LDFLAGS         := -lm -lX11 -lpthread
CUDA_LDFLAGS	:= $(LDFLAGS) -lrt -lcudart

CC		:= g++
NVCC		:= nvcc
LINKER		:= g++

