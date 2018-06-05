CC=clang++
LDFLAGS=-L../../target/debug/
LIBS=-lweld

.PHONY: all clean

all:
	${CC} ${LDFLAGS} ${LIBS} pred_vs_no_pred.cpp -o run

clean:
	rm -rf run