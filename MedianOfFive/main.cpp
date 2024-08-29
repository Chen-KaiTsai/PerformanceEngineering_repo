#include <iostream>
#include <cstdio>
#include <cstring>

/*
 * Reference : https://stackoverflow.com/questions/480960/how-do-i-calculate-the-median-of-five-in-c
 * If the array is bigger than 5, only the first 5 will be take into consideration
*/
int MedianOfFive(const int* data) {
    // a, b, c, d, e
    // 0, 1, 2, 3, 4
    int* temp = new int[5];
    int result = 0;

    memcpy(temp, data, sizeof(int) * 5);

    // make a < b
    if (temp[1] < temp[0]) {
        std::swap(temp[1], temp[0]);
    }

    // make c < d
    if (temp[3] < temp[2]) {
        std::swap(temp[3], temp[2]);
    }

    // eliminate the lowest
    // notice that some elements will be eleminated
    // therefore, not all the original data will be kept in final temp 
    if (temp[2] < temp[0]) {
        std::swap(temp[3], temp[1]);
        temp[2] = temp[0];
    }

    // take e into consideration
    // a = e
    temp[0] = temp[4];

    // make a < b
    if (temp[1] < temp[0]) {
        std::swap(temp[1], temp[0]);
    }

    // eliminate another lowest
    if (temp[0] < temp[2]) {
        std::swap(temp[1], temp[3]);
        temp[0] = temp[2];
    }

    if (temp[3] < temp[0]) {
        result = temp[3];
    }
    else {
        result = temp[0];
    }

    delete[] temp;

    return result;
}

int main(int argc, char** argv) {
    int data[] = {5, 4, 3, 2, 1};
    int result = MedianOfFive(data);

    printf("%d\n\n", result);

    return 0;
}
