#include <stdio.h>

#include <time.h>
#include "opencl.h"

#include "ex01.h"
#include "ex02.h"
#include "ex03.h"
#include "ex04.h"
#include "ex05.h"
#include "ex06.h"

int main() {
    printf("Hello GPU Compute World! ;-)\n\n");

    int tests = 4;
    int gpuix = 0; // 0 => CPU, 1,2,3... => GPU

    printf("EX01: SUM INT\n");
    ex01(10000000, tests, gpuix);
    printf("\n");

    printf("EX02: SUM\n");
    ex02(10000000, tests, gpuix);
    printf("\n");

    printf("EX03: SUB\n");
    ex03(10000000, tests, gpuix);
    printf("\n");

    printf("EX04: MUL\n");
    ex04(50, tests, gpuix);
    printf("\n");

    printf("EX05: DIV\n");
    ex05(25, tests, gpuix);
    printf("\n");

    printf("EX06: ANN\n");
    ex06(2000, 20, tests, gpuix);
    printf("\n");

    return 0;
}