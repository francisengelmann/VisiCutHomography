objects = main.o

all : clean $(objects)
	g++ `pkg-config opencv --cflags --libs` main.cpp
	./a.out  

main.o : main.cpp
	#g++ `pkg-config opencv --cflags --libs` main.c -o main.o

debug : 
	g++ `pkg-config opencv --cflags --libs` main.cpp -DEBUG=1
	./a.out

clean :
	rm -f $(objects)
