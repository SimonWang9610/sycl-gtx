CXX = g++

all: cl cl.o clean

LIBS += -L/usr/lib/x86_64-linux-gnu -lOpenCL
INCLUDES += -I/usr/include

cl: cl.o
	${CXX} -o cl cl.o ${LIBS}

cl.o: cl.cpp
	${CXX} -o cl.o -c cl.cpp ${INCLUDES}

clean:
	rm -rf *.o