#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N (1024 * 512)

void merge(int *input_data, int i1, int j1, int i2, int j2)
{
    int i, j, k;
    i = i1;
    j = i2;
    k = 0;

    int temp_size = j2 - i1 + 1;
    int *temp = (int *)malloc(temp_size * sizeof(int));

    while (i <= j1 && j <= j2)
    {
        if (input_data[i] < input_data[j])
            temp[k++] = input_data[i++];
        else
            temp[k++] = input_data[j++];
    }

    while (i <= j1)
        temp[k++] = input_data[i++];

    while (j <= j2)
        temp[k++] = input_data[j++];

    for (i = i1, j = 0; i <= j2; i++, j++)
        input_data[i] = temp[j];

    free(temp);
}

void serial_mergesort(int *input_data, int i, int j)
{
    int mid;

    if (i < j)
    {
        mid = (i + j) / 2;

        serial_mergesort(input_data, i, mid);
        serial_mergesort(input_data, mid + 1, j);

        merge(input_data, i, mid, mid + 1, j);
    }
}

int main(int argc, char **argv)
{
    int rank, nprocs;
    int *input_data;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int partial_data_size = N / nprocs;

    int *partial_data = (int *)malloc(partial_data_size * sizeof(int));

    if (rank == 0)
    {
        input_data = (int *)malloc(N * sizeof(int));
        int i;

        for (i = 0; i < N; i++)
            input_data[i] = rand();
    }

    double time_x = MPI_Wtime();

    MPI_Scatter(input_data, partial_data_size, MPI_INT, partial_data, partial_data_size, MPI_INT, 0, MPI_COMM_WORLD);

    serial_mergesort(partial_data, 0, partial_data_size - 1);

    int i, jump = 1;

    for (i = nprocs; i != 1; i /= 2, jump *= 2)
    {
        if ((rank + 1) % (2 * jump) != 0 && (rank + 1) % jump == 0)
        {
            //printf("%d Send to %d size = %d\n", rank, (rank + (1 * jump)), (partial_data_size * jump));
            MPI_Send(partial_data, (partial_data_size * jump), MPI_INT, (rank + (1 * jump)), rank, MPI_COMM_WORLD);
        }
        else if((rank + 1) % (2 * jump) == 0 && (rank + 1) % jump == 0)
        {
            //printf("%d Recv from %d size = %d\n", rank, (rank - (1 * jump)), (partial_data_size * jump));

            MPI_Status status;
            int *recv_partial_data = (int *)malloc(partial_data_size * jump * sizeof(int));
            MPI_Recv(recv_partial_data, (partial_data_size * jump), MPI_INT, (rank - (1 * jump)), (rank - (1 * jump)), MPI_COMM_WORLD, &status);

            partial_data = (int *)realloc(partial_data, 2 * partial_data_size * jump * sizeof(int));

            int j, k = 0;
            for (j = partial_data_size * jump; j < 2 * partial_data_size * jump; j++)
                partial_data[j] = recv_partial_data[k++];

            free(recv_partial_data);

            /*
            for (j = 0; j < 2 * partial_data_size * jump; j++)
                printf("%d ", partial_data[j]);
            printf("\n\n");
            */
            merge(partial_data, 0, (partial_data_size * jump) - 1, partial_data_size * jump, (2 * partial_data_size * jump) - 1);
            /*
            for (j = 0; j < 2 * partial_data_size * jump; j++)
                printf("%d ", partial_data[j]);
            printf("\n");
            */
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == nprocs - 1)
    {
        MPI_Send(partial_data, N, MPI_INT, 0, nprocs - 1, MPI_COMM_WORLD);
    }
    else if(rank == 0)
    {
        MPI_Status status;
        MPI_Recv(input_data, N, MPI_INT, nprocs - 1, nprocs - 1, MPI_COMM_WORLD, &status);

        double time_y = MPI_Wtime();

        /*
        for (i = 0; i < N; i++)
            printf("%d ", input_data[i]);
        printf("\n");
        */

        printf("execute time = %lf\n", time_y - time_x);

        free(input_data);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    free(partial_data);

    MPI_Finalize();
}