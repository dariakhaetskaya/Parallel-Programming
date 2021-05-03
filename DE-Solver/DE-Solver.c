#include <iostream>
#include <mpi.h>
#include <cmath>
#include <cfloat>
#include <cstring>

using namespace std;

#define N 320
#define a 10e5
#define epsilon 10e-8
#define idx(i,j,k) N*N*(i) + N*(j) + k

#define D_X 2
#define D_Y 2
#define D_Z 2

#define X_0 -1
#define Y_0 -1
#define Z_0 -1

int ProcNum = 0;
int ProcRank = 0;

double H_X = D_X/(double)(N - 1);
double H_Y = D_Y/(double)(N - 1);
double H_Z = D_Z/(double)(N - 1);

double H_X2 = H_X * H_X;
double H_Y2 = H_Y * H_Y;
double H_Z2 = H_Z * H_Z;

double phi(double x, double y, double z){
    return x*x + y*y + z*z;
}

double ro(double x, double y, double z){
    return 6 - a * phi(x, y, z);
}

//calculating coords;
double X(int i) {
    return (X_0 + i * H_X);
}

double Y(int j) {
    return (Y_0 + j * H_Y);
}

double Z(int k) {
    return (Z_0 + k * H_Z);
}

void initializePhi(int LayerHeight, double* currentLayer){
    for (int i = 0; i < LayerHeight + 2; i++){
        int RelativeZ = i + ((ProcRank * LayerHeight) - 1);
        double z = Z(RelativeZ);

        for(int j = 0; j < N; j++){
            double x = X(j);

            for(int k = 0; k < N; k++){
                double y = Y(k);

                if ( k != 0 && k != N-1 &&
                     j != 0 && j != N-1 &&
                     z != Z_0 && z != Z_0 + D_Z ){
                    currentLayer[idx(i,j,k)] = 0;
                } else {
                    currentLayer[idx(i,j,k)] = phi(x, y, z);
                }

            }
        }
    }
}

void printCube(double* A){
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                printf(" %7.4f", A[idx(i,j,k)]);
            }
            cout << ";";
        }
        cout << endl;
    }
}

double CalculateDelta(double* Omega){
    auto deltaMax = DBL_MIN;
    double x, y, z;
    for(int i = 0; i < N; i++){
        x = X(i);
        for (int j = 0; j < N; j++){
            y = Y(j);
            for(int k = 0; k < N; k++){
                z = Z(k);
                deltaMax = max(deltaMax, abs(Omega[idx(i,j,k)] - phi(x, y, z)));
            }
        }
    }

    return deltaMax;
}

double UpdateLayer(int RelativeZCoordinate, int LayerIdx, double* CurrentLayer, double* CurrentLayerBuf){
    int AbsoluteZCoordinate = RelativeZCoordinate + LayerIdx; // во всей омега
    auto deltaMax = DBL_MIN;
    double x, y, z;

    if (AbsoluteZCoordinate == 0 || AbsoluteZCoordinate == N - 1){
        memcpy(CurrentLayerBuf + LayerIdx*N*N, CurrentLayer + LayerIdx*N*N, N*N*sizeof(double));
        deltaMax = 0;
    } else {
        z = Z(AbsoluteZCoordinate);

        for(int i = 0; i < N; i++){
            x = X(i);
            for (int j = 0; j < N; j++){
                y = Y(j);

                if(i == 0 || i == N-1 || j == 0 || j == N - 1){
                    CurrentLayerBuf[idx(LayerIdx, i, j)] = CurrentLayer[idx(LayerIdx, i, j)];
                } else {
                    CurrentLayerBuf[idx(LayerIdx, i, j)] =
                            ((CurrentLayer[idx(LayerIdx+1, i, j)] + CurrentLayer[idx(LayerIdx-1, i, j)]) / H_Z2 +
                             (CurrentLayer[idx(LayerIdx, i+1, j)] + CurrentLayer[idx(LayerIdx, i-1, j)]) / H_X2 +
                             (CurrentLayer[idx(LayerIdx, i, j+1)] + CurrentLayer[idx(LayerIdx, i, j-1)]) / H_Y2 -
                             ro(x, y, z)) / (2/H_X2 + 2/H_Y2 + 2/H_Z2 + a);

                    if (abs(CurrentLayerBuf[idx(LayerIdx, i, j)] - CurrentLayer[idx(LayerIdx, i, j)]) > deltaMax){
                        deltaMax = CurrentLayerBuf[idx(LayerIdx, i, j)] - CurrentLayer[idx(LayerIdx, i, j)];
                    }

                }
            }
        }
    }

    return deltaMax;
}

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Request req[4];

    if (N % ProcNum && ProcRank == 0) {
        cout << "Grid size " << N << " should be divisible by the ProcNum " << endl;
        return 0;
    }

    double* Omega;

    auto GlobalMaxDelta = DBL_MAX;

    int LayerSize = N / ProcNum;
    int LayerZCoordinate = ProcRank * LayerSize - 1;

    int ExtendedLayerSize = (LayerSize + 2) * N * N;
    auto* CurrentLayer = new double[ExtendedLayerSize];
    auto* CurrentLayerBuf = new double[ExtendedLayerSize];

    initializePhi(LayerSize, CurrentLayer);

    double Start = MPI_Wtime();
    do {
        auto ProcMaxDelta = DBL_MIN;
        double tmpMaxDelta;

        if(ProcRank != 0){
            MPI_Isend(CurrentLayerBuf + N*N, N*N, MPI_DOUBLE,
                      ProcRank - 1, 888, MPI_COMM_WORLD, &req[1]);

            MPI_Irecv(CurrentLayerBuf, N*N, MPI_DOUBLE,
                    ProcRank - 1, 888, MPI_COMM_WORLD, &req[0]);
        }

        if(ProcRank != ProcNum  - 1){
            MPI_Isend(CurrentLayerBuf + N*N*LayerSize, N*N, MPI_DOUBLE,
                      ProcRank + 1, 888, MPI_COMM_WORLD, &req[3]);

            MPI_Irecv(CurrentLayerBuf + N*N*(LayerSize + 1), N*N, MPI_DOUBLE,
                      ProcRank + 1, 888, MPI_COMM_WORLD, &req[2]);
        }

        for (int LayerIdx = 2; LayerIdx < LayerSize; LayerIdx++){
            tmpMaxDelta = UpdateLayer(LayerZCoordinate, LayerIdx, CurrentLayer, CurrentLayerBuf);
            ProcMaxDelta = max(ProcMaxDelta, tmpMaxDelta);
        }

        if(ProcRank != ProcNum  - 1){
            MPI_Wait(&req[2], MPI_STATUS_IGNORE);
            MPI_Wait(&req[3], MPI_STATUS_IGNORE);
        }

        if(ProcRank != 0){
            MPI_Wait(&req[0], MPI_STATUS_IGNORE);
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
        }

        tmpMaxDelta = UpdateLayer(LayerZCoordinate, 1, CurrentLayer, CurrentLayerBuf);
        ProcMaxDelta = max(ProcMaxDelta, tmpMaxDelta);

        tmpMaxDelta = UpdateLayer(LayerZCoordinate, LayerSize, CurrentLayer, CurrentLayerBuf);
        ProcMaxDelta = max(ProcMaxDelta, tmpMaxDelta);

        memcpy(CurrentLayer, CurrentLayerBuf, ExtendedLayerSize * sizeof(double));

        MPI_Allreduce(&ProcMaxDelta, &GlobalMaxDelta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    } while (GlobalMaxDelta > epsilon);

    delete [] CurrentLayerBuf;

    double End = MPI_Wtime();

    if (ProcRank == 0){
        Omega = new double [N * N * N];
    }

    MPI_Gather(CurrentLayer + N*N, LayerSize * N*N, MPI_DOUBLE, Omega,
               LayerSize * N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (ProcRank == 0){
        cout << "Time taken: " << End - Start << endl;
        cout << "Delta: " << CalculateDelta(Omega) << endl;
        delete [] Omega;
    }

    delete [] CurrentLayer;

    MPI_Finalize();
    return 0;
}
