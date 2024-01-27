CC = g++
CFLAGS = -std=c++17 -c
LIB = libfad.a
SRCS = Auto_grad.cpp fad.cpp
OBJS = $(SRCS:.cpp=.o)
$(LIB) : $(OBJS)
	ar rc $(LIB) $(OBJS)

.c.o:
	$(CC) $(CFLAGS) $(SRCS)

clean:
	rm -f $(OBJS) $(LIB)
