#include <stdio.h>
#include <omp.h>

void LinearSearch(int *A, int N, int key, int *position)
{
    int i;
    *position = -1;

#pragma omp parallel for
    for (i = 0; i < N; i++)
        if (A[i] == key)
            *position = i;
}

void NarySearch(int *A, int lo, int hi, int key, int Ntrvl, int *pos)
{
    float offset, step;
    int *mid = new int[Ntrvl + 1];
    char *locate = new char[Ntrvl + 2];
    int i;

    locate[0] = 'R';
    locate[Ntrvl + 1] = 'L';
    #pragma omp parallel
    {
        while (lo < hi && *pos == -1)
        {
            int lmid;
            #pragma omp single
            {
                mid[0] = lo - 1;
                step = (float)(hi - lo + 1) / (Ntrvl + 1);
            }
            #pragma omp for private(offset) firstprivate(step)
            for (i = 1; i <= Ntrvl; i++)
            {
                offset = step * i + (int)offset;
                if (lmid <= hi)
                {
                    if (A[lmid] > key)
                        locate[i] = 'L';
                    else if (A[lmid] < key)
                        locate[i] = 'R';
                    else
                    {
                        locate[i] = 'E';
                        *pos = lmid;
                    }
                }
                else
                {
                    mid[i] = hi + 1;
                    locate[i] = 'L';
                }
            }
            #pragma omp single
            {
                for (i = 1; i <= Ntrvl; i++)
                {
                    if (locate[i] != locate[i - 1])
                    {
                        lo = mid[i - 1] + 1;
                        hi = mid[i] - 1;
                    }
                }
                if (locate[Ntrvl] != locate[Ntrvl + 1])
                    lo = mid[Ntrvl] + 1;
            }
        }
    }
}

int main(void)
{
    
}
