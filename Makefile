CC = gcc
CFLAGS = -fPIC -Wall -Wextra -O3 -g -ffast-math 
LDFLAGS = -shared
RM = rm -f
TARGET_LIB = libpecann.so

SRCS = src/matrix.c src/network.c
OBJS = $(SRCS:.c=.o)

.PHONY: all
all: ${TARGET_LIB}
test: libpecann.so
	$(CC) -L. -Wl,-rpath=. test/mnist.c -lpecann -lm -o test/mnist 

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -o $@ $^

$(SRCS:.c=.d):%.d:%.c
	$(CC) $(CFLAGS) -MM $< >$@

include $(SRCS:.c=.d)

.PHONY: clean
clean:
	-${RM} ${TARGET_LIB} ${OBJS} $(SRCS:.c=.d) test/mnist test/libpecann.so
