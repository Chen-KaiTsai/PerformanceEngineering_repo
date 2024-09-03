main: main.o malloc2D.o gemm.o
	@g++ -o main *.o -O3 -m64 -Wall -march=native -fopenmp -I/home/erebus/VCL/version2-2.02.01

malloc2D.o:
	@g++ -c -O3 -m64 malloc2D.cpp -march=native -fopenmp -I/home/erebus/VCL/version2-2.02.01

gemm.o:
	@g++ -c -O3 -m64 gemm.cpp -march=native -fopenmp -I/home/erebus/VCL/version2-2.02.01

main.o:
	@g++ -c -O3 -m64 main.cpp -march=native -fopenmp -I/home/erebus/VCL/version2-2.02.01
