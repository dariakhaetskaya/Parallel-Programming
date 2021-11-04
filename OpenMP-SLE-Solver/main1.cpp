#include <iostream>
#include <cmath>
#include <omp.h>
#include <cstdlib>
#include <cstddef>
#include <stdio.h>

const long int N = 2400;
const double epsilon = pow(10, -7);
const float parameter =  0.001;

double EuclideanNorm(const double* u){
    double norm = 0;
    for (int i = 0; i < N; i++){
        norm += u[i]*u[i];
    }
    return sqrt(norm);
}

void sub(double* a, double* b, double* c){
    for (int i = 0; i < N; i++) {
        c[i] = a[i] - b[i];
    }
}

void mul(double* A, double* b, double* result, int n) {
    unsigned int i, j;
    for(i = 0; i < n; i++) {
        result[i] = 0;
        for(j = 0; j < n; j++) {
            result[i] += A[i * n + j] * b[j];
        }
    }
}


void scalMul(double* A, double tau){
    int i;
    for (i = 0; i < N; ++i) {
        A[i] = A[i] * tau;
    }
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
    omp_set_num_threads(8);
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
    unsigned int i, j;
    double norm = 0;

#pragma omg parallel private(i, j)
    {
        while (res > epsilon) {
            #pragma omp single
            {
                for (long i = 0; i < N; i++) {
                    prevX[i] = nextX[i];
                }
            }

            #pragma omp parallel for private (j) // works fine
            for(i = 0; i < N; i++) {
                Ax[i] = 0;
                for(j = 0; j < N; j++) {
                    Ax[i] += A[i * N + j] * prevX[j];
                }
            }

            #pragma omp parallel for // works fine
            for (i = 0; i < N; i++) {
                Ax[i] = Ax[i] - b[i];
            }

            #pragma omp atomic write
            norm = 0;

            #pragma omp parallel for reduction (+:norm)
            for (i = 0; i < N; i++){
                norm += Ax[i]*Ax[i];
            }

            #pragma omp atomic write
            norm = sqrt(norm);

            #pragma omp parallel for
            for (i = 0; i < N; ++i) {
                Ax[i] = Ax[i] * tau;
            }

            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                nextX[i] = prevX[i] - Ax[i];
            }

            #pragma omp critical
            {
                res = norm / normb;
                countIt++;

                if ((countIt > 100000 && lastres > res) || res == INFINITY) {
                    if (tau < 0) {
                        printf("Does not converge\n");
                        res = 0;
                    } else {
                        tau = (-1)*parameter;
                        countIt = 0;
                        res = saveRes;
                    }
                }
                lastres = res;
            }
        }
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