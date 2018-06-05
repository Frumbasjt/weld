CC=clang++
LDFLAGS=-L../../target/debug/
LIBS=-lweld

.PHONY: all clean

all:
	${CC} ${LDFLAGS} ${LIBS} filter_map_reorder.cpp -o run

clean:
	rm -rf run