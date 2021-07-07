CXX = g++

all: cl cl.o clean

LIBS += -L/usr/local/cuda/lib64 -lOpenCL
INCLUDES += -I/usr/local/cuda/include

cl: cl.o
	${CXX} -o cl cl.o ${LIBS}

cl.o: cl.cpp
	${CXX} -o cl.o -c cl.cpp ${INCLUDES}

clean:
	rm -rf *.o