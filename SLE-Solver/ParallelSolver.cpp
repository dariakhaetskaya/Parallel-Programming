//
// Created by rey on 2/25/21.
//
#include <iostream>
#include <cmath>
#include <mpi.h>

const long int N = 6400;
const double epsilon = pow(10, -7);
const double parameter =  0.0001;

double EuclideanNorm(const double* u){
    double norm = 0;
    for (int i = 0; i < N; i++){
        norm += u[i]*u[i];
    }
    return sqrt(norm);
}

void sub(double* a, double* b, double* c, int n){
    for (int i = 0; i < n; i++) {
        c[i] = a[i] - b[i];
    }
}

void mul(double* A, double* b, double* result, int n) {
    for(unsigned int i = 0; i < n; i++){
        result[i] = 0;
        for(unsigned int j = 0; j < N; j++){
            result[i] += A[i * N + j] * b[j];
        }
    }
}

void scalMul(double* A, double tau, int n){
    for (int i = 0; i < n; ++i) {
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

double drand(double low, double high) {
    double f = (double)rand() / RAND_MAX;
    return low + f * (high - low);
}

double rand_double(){
    return (double)rand()/RAND_MAX*4.0 - 2.0;
}

void generate_matrix(double* matrix) {

    for(int i = 0; i < N; i++){
        for(int j = 0; j < i; j++){
            matrix[i*N + j] = matrix[j*N + i];
        }
        for(int j = i; j < N; j++){
            matrix[i*N + j] = rand_double();
            if(i == j){
                matrix[i*N + j] = fabs(matrix[i*N + j]) + 410.0;
            }
        }
    }
}

void printVector(const double* B, const char* name, int procRank, int procNum, int n){
    for (int numProc = 0; numProc < procNum; ++numProc) {
        if (procRank == numProc) {
            printf("%s in rank %d:\n", name, procRank);
            for (int i = 0; i < n; ++i) {
                printf("%lf\n", B[i]);
            }
            printf("\n");
        }
    }
}

int main(int argc, char** argv) {
    srand(time(0));

    int process_Rank, size_Of_Cluster;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);

    // process shared buffers
    double* ABuf  = (double*)malloc(N*N / size_Of_Cluster * sizeof(double)); // free
    double* AxBuf = (double*)malloc(N / size_Of_Cluster * sizeof(double)); // free

    double* prevX = nullptr;
    double* Ax = nullptr;
    double* nextX = nullptr;
    double* A = nullptr;
    double* b = nullptr;
    double tau = parameter;
    double startTime = 0, endTime = 0, currentTime = 0;

    bool timeOut = false;
    int timeLimit = 120;

    double normAxb = 0; // ||A*xn - b||
    double normb = 0;
    double saveRes = 1;
    double res = 1;
    double lastres = 1;
    b = (double*)malloc(N * sizeof(double));
    prevX = (double*)malloc(N * sizeof(double));
    Ax = (double*)malloc(N * sizeof(double));
    nextX = (double*)malloc(N * sizeof(double));

    for (long i = 0; i < N; i++) {
        prevX[i] = drand(1,5);
        nextX[i] = drand(1,5);
        b[i] = drand(1,N);
    }

    if (process_Rank == 0){
        A = (double*)malloc(N * N * sizeof(double));
        generate_matrix(A);
        mul(A, prevX, Ax, N);
        normb = EuclideanNorm(b);
    }

    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Ax,    N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&res, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&normb, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&saveRes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(A, N * N / size_Of_Cluster, MPI_DOUBLE, ABuf,
                N * N / size_Of_Cluster, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int countIt = 1;

    startTime = MPI_Wtime();

    while (res > epsilon){

        if(process_Rank == 0) {
            for (long i = 0; i < N; i++) {
                prevX[i] = nextX[i];
            }
        }

        MPI_Bcast(prevX, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(Ax, N / size_Of_Cluster, MPI_DOUBLE, AxBuf,
                    N / size_Of_Cluster, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        mul(ABuf, prevX, AxBuf, N / size_Of_Cluster); // A*xn

        sub(AxBuf, b, AxBuf, N / size_Of_Cluster); // A*xn - b
        scalMul(AxBuf, tau, N / size_Of_Cluster); // TAU*(A*xn - b)

        MPI_Allgather(AxBuf, N / size_Of_Cluster, MPI_DOUBLE, Ax,
                      N / size_Of_Cluster, MPI_DOUBLE, MPI_COMM_WORLD);

        if(process_Rank == 0){
            normAxb = EuclideanNorm(Ax); // ||A*xn - b||
            sub(prevX, Ax, nextX, N); // xn - TAU * (A*xn - b)
            res = normAxb / normb;
            countIt++;
            if ((countIt > 100000 && lastres > res) || res == INFINITY) {
                if (tau < 0) {
                    printf("Does not converge\n");
                    saveRes = res;
                    res = 0;
                }
                else {
                    tau = (-1)*parameter;
                    countIt = 0;
                }
            }
            lastres = res;
        }

        currentTime = MPI_Wtime();



        if ((currentTime - startTime) > timeLimit){
            timeOut = true;
        }

        MPI_Bcast(nextX, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(Ax,    N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&countIt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&res, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&lastres, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&tau, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&saveRes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    endTime = MPI_Wtime();

    if (process_Rank == 0){
        if (res == 0){
            res = saveRes;
        }
        if (!timeOut){
            printf("%ld*%ld matrix error coefficient is %lf, iterations: %d\n",N, N, res, countIt);
            printf("That took %lf seconds\n",endTime-startTime);
        } else {
            printf("it took more than %d seconds so I killed the process,"
                   "error coefficient was %lf\n", timeLimit, res);
        }
        free(A);
    }
    free(ABuf);
    free(Ax);
    free(nextX);
    free(prevX);
    free(b);
    free(AxBuf);
    MPI_Finalize();
    return 0;
}

