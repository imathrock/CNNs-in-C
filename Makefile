# tutorial referred on website: https://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
# Compiler and Flags
CC = gcc
CFLAGS = -I. -Wall -Wextra -g

# Dependencies
DEPS = idx-file-parser.h NeuralNetwork.h Convolution2D.h
OBJ = idx-file-parser.o NeuralNetwork.o Convolution2D.o main.o

# Pattern rule for object files
%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) -lpthread

# Target for the executable
main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

# Clean rule to remove generated files
.PHONY: clean
clean:
	rm -f *.o main
