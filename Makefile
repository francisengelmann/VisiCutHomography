objects = main.o

all : clean $(objects)
	g++ `pkg-config opencv --cflags --libs` main.cpp
	./a.out  

main.o : main.cpp
	#g++ `pkg-config opencv --cflags --libs` main.c -o main.o

clean :
	rm -f $(objects)
