
debug		:= 0
include		../Makefile.inc


all: $(BIN)/sequential $(BIN)/compare
 
$(BIN)/sequential: main.cpp filters.cpp
	$(LINKER) -o $(BIN)/sequential main.cpp filters.cpp ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

$(BIN)/compare: compare.cpp
	$(LINKER) -o $(BIN)/compare compare.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

clean:
	rm -f $(BIN)/sequential
	rm -f $(BIN)/compare

