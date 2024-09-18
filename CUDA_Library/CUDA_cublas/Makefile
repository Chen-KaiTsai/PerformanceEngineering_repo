main: main.o
	@nvcc -o main *.o -g -m64 -I/usr/local/cuda/samples/common/ -gencode arch=compute_61,code=sm_61 -lcublas -lcudnn --compiler-options -Wall

main.o:
	@nvcc -c -g main.cpp -I/usr/local/cuda/samples/common/
