# tutorial referred on website: https://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
# Compiler and Flags
CC = gcc
CFLAGS = -I. -Wall -Wextra -g -march=haswell -std=c99


DEPS = dataloaders/idx-file-parser.h NN-funcs/NeuralNetwork.h Conv/Convolution2D.h
OBJ = dataloaders/idx-file-parser.o NN-funcs/NeuralNetwork.o Conv/Convolution2D.o main.o

# Pattern rule for object files
%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) 

# Target for the executable
main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

# Clean rule to remove generated files
.PHONY: clean
clean:
	rm -f *.o main