CC=g++
CFLAGS=-O3 -Wall -g -march=native -mavx2 -I eigen-3.4.0/ -I json/include -I "./GTnn"

BIN=gtnn_max gtnn_sum gtnn_stream

all: $(BIN)

gtnn_max: example_GTmax.cpp
	$(CC) $(CFLAGS) $^ -o $@

gtnn_sum: example_GTsum.cpp
	$(CC) $(CFLAGS) $^ -o $@

gtnn_stream: example_GTstreaming.cpp
	$(CC) $(CFLAGS) $^ -o $@

clean: 
	rm $(BIN)