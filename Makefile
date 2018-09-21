FLAGS= -g -Wall -std=c++11 -o
CC=g++

ann:main.cpp ann.cpp  ann.h
	$(CC) $(FLAGS) $@ $^

run:
	./ann
clean:
	rm -f ann
