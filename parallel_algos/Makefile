main: main.o helper.o omp_algos.o c11_algos.o
	@g++ -o main *.o -g -std=c++20 -m64 -Wall -pthread -fopenmp

helper.o: helper.cpp
	@g++ -c -g -std=c++20 helper.cpp -pthread -fopenmp

omp_algos.o: omp_algos.cpp
	@g++ -c -g -std=c++20 omp_algos.cpp -pthread -fopenmp

c11_algos.o: c11_algos.cpp
	@g++ -c -g -std=c++20 c11_algos.cpp -pthread -fopenmp

main.o: main.cpp
	@g++ -c -g -std=c++20 main.cpp -pthread -fopenmp
