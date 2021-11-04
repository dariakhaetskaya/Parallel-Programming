//
// Created by rey on 2/25/21.
//
#include <iostream>
#include <cmath>
#include <omp.h>

const long int N = 2400;
const double epsilon = pow(10, -7);
const float parameter =  0.001;

double EuclideanNorm(const double* u){
    double norm = 0;
    #pragma omp parallel for reduction (+:norm)
    for (int i = 0; i < N; i++){
        norm += u[i]*u[i];
    }
    return sqrt(norm);
}

void sub(double* a, double* b, double* c){
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        c[i] = a[i] - b[i];
    }
}

void mul(double* A, double* b, double* result, int n) {
    unsigned int i, j;
    #pragma omp parallel for private (j)
    for(i = 0; i < n; i++) {
        result[i] = 0;
        for(j = 0; j < n; j++) {
            result[i] += A[i * n + j] * b[j];
        }
    }
}


void scalMul(double* A, double tau){
    int i;
    #pragma omp parallel for
    for (i = 0; i < N; ++i) {
        A[i] = A[i] * tau;
    }
}

void printMatrix(double* A){
    printf("\n");

    for(unsigned int i = 0; i < N; i++) {
        for(unsigned int j = 0; j < N; j++) {
            printf("%lf ", A[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printVec(double* A, int n){
    printf("\n");
    printf("\n");
    for (int i = 0; i < n; ++i){
        printf("%lf ", A[i]);
    }
    printf("\n");
    printf("\n");
}

double drand(double low, double high) {
    double f = (double)rand() / RAND_MAX;
    return low + f * (high - low);
}

double rand_double(){
    return (double)rand()/RAND_MAX*4.0 - 2.0;
}

void generate_matrix(double* matrix) {
    double rand_value = 0;

    for(int i = 0; i < N; i++){
        for(int j = 0; j < i; j++){
            matrix[i*N + j] = matrix[j*N + i];
        }
        for(int j = i; j < N; j++){
            matrix[i * N + j] = rand_double();
            if(i == j){
                matrix[i*N + j] = fabs(matrix[i*N + j]) + 124.0;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    srand(100);
    omp_set_num_threads(2);
    double* prevX = (double*)malloc(N * sizeof(double));
    double* Ax = (double*)malloc(N * sizeof(double));
    double tau = parameter;

    double normAxb = 0; // ||A*xn - b||
    double normb = 0;
    double saveRes = 0;
    double res = 0;
    double lastres = 0;

    double start, end;

    double* b = (double*)malloc(N * sizeof(double));
    double* nextX = (double*)malloc(N * sizeof(double));
    double* A = (double*)malloc(N * N * sizeof(double));

    for (long i = 0; i < N; i++) {
        prevX[i] = rand_double();
        b[i] = rand_double();

    }

    generate_matrix(A);

    mul(A, prevX, Ax, N); // A*xn

    sub(Ax, b, Ax); // A*xn - b
    normAxb = EuclideanNorm(Ax); // ||A*xn - b||
    normb = EuclideanNorm(b);
    scalMul(Ax, tau); // TAU*(A*xn - b)
    sub(prevX, Ax, nextX); // xn - TAU * (A*xn - b)
    saveRes = normAxb / normb;
    res = normAxb / normb;
    lastres = res;

    int countIt = 1;

    start = omp_get_wtime();

    while (res > epsilon) {
        for (long i = 0; i < N; i++) {
            prevX[i] = nextX[i];
        }

        mul(A, prevX, Ax, N); //A*xn
        sub(Ax, b, Ax); //A*xn - b
        normAxb = EuclideanNorm(Ax); // ||A*xn - b||
        scalMul(Ax, tau); // TAU*(A*xn - b)
        sub(prevX, Ax, nextX); // xn - TAU * (A*xn - b)
        res = normAxb / normb;
        countIt++;

        if ((countIt > 100000 && lastres > res) || res == INFINITY) {
            if (tau < 0) {
                printf("Does not converge\n");
                res = 0;
            } else {
                tau = -0.01;
                countIt = 0;
                res = saveRes;
            }
        }
        lastres = res;
        printf("init res = %lf, epsil = %lf\n", res, epsilon);
    }

    end = omp_get_wtime();
    printf("time: %lf\n", end - start);

    printf("res = %lf, iterations: %d\n", res, countIt);
    free(Ax);
    free(nextX);
    free(prevX);
    free(b);
    free(A);
    return 0;
}
