#include <cmath>
#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <mpi.h>
using namespace std;

vector<double> sequentialSpMV(
    const vector<double>& values,
    const vector<int>& col_indices,
    const vector<int>& row_ptr,
    const vector<double>& x
) {
    int rows = row_ptr.size() - 1;
    vector<double> result(rows, 0.0);
    for (int i = 0; i < rows; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            result[i] += values[j] * x[col_indices[j]];
        }
    }
    return result;
}

vector<double> parallelSpMV(
    const vector<double>& values,
    const vector<int>& col_indices,
    const vector<int>& row_ptr,
    const vector<double>& x
) {
    int rows = row_ptr.size() - 1;
    vector<double> result(rows, 0.0);
#pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            result[i] += values[j] * x[col_indices[j]];
        }
    }
    return result;
}

void generateSparseMatrix(
    int rows,
    int nonZerosPerRow,
    vector<double>& values,
    vector<int>& col_indices,
    vector<int>& row_ptr
) {
    row_ptr.push_back(0);
    for (int i = 0; i < rows; i++) {
        if (i % 10 == 0) {
            row_ptr.push_back(static_cast<int>(values.size()));
            continue;
        }
        for (int j = 0; j < nonZerosPerRow; j++) {
            values.push_back(1.0 + (i + j) % 10);
            col_indices.push_back((i + j) % rows);
        }
        row_ptr.push_back(values.size());
    }
}

void runTests() {
    vector<double> values = { 1, 2, 3, 4, 5, 6, 7 };
    vector<int> cols = { 0, 3, 0, 1, 1, 2, 3 };
    vector<int> ptr = { 0, 2, 2, 4, 7 };
    vector<double> x = { 1, 2, 3, 4 };
    vector<double> expected = { 9, 0, 11, 56 };
    auto seq = sequentialSpMV(values, cols, ptr, x);
    auto par = parallelSpMV(values, cols, ptr, x);
    cout << "Sequential test: "
        << (seq == expected ? "PASS" : "FAIL") << endl;
    cout << "OpenMP test: "
        << (par == expected ? "PASS" : "FAIL") << endl;
    vector<double> emptyValues;
    vector<int> emptyCols;
    vector<int> emptyPtr = { 0, 0, 0, 0 };
    vector<double> emptyX = { 1, 2, 3 };
    vector<double> expectedZeros = { 0, 0, 0 };
    auto seqZero = sequentialSpMV(
        emptyValues, emptyCols, emptyPtr, emptyX
    );
    auto ompZero = parallelSpMV(
        emptyValues, emptyCols, emptyPtr, emptyX
    );

    cout << "Zero matrix test: "
        << (seqZero == expectedZeros &&
            ompZero == expectedZeros ? "PASS" : "FAIL")
        << endl;
}

int main() {
    MPI_Init(nullptr, nullptr);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int N = 100000;
    const int nnzPerRow = 100;
    vector<double> values;
    vector<int> col_indices;
    vector<int> row_ptr;
    vector<double> x(N, 1.0);
    if (rank == 0) {
        generateSparseMatrix(
            N, nnzPerRow, values, col_indices, row_ptr
        );
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();
    MPI_Bcast(x.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int startRow = rank * N / size;
    int endRow = (rank + 1) * N / size;
    int localRows = endRow - startRow;
    vector<double> localValues;
    vector<int> localCols;
    vector<int> localRowPtr(localRows + 1);
    MPI_Barrier(MPI_COMM_WORLD);
    double distributionStart = MPI_Wtime();
    vector<int> nnzCounts(size, 0);
    vector<int> nnzDispls(size, 0);
    vector<int> ptrCounts(size, 0);
    vector<int> ptrDispls(size, 0);
    if (rank == 0) {
        for (int p = 0; p < size; p++) {
            int begin = p * N / size;
            int end = (p + 1) * N / size;
            nnzCounts[p] = row_ptr[end] - row_ptr[begin];
            nnzDispls[p] = row_ptr[begin];
            ptrCounts[p] = end - begin + 1;
            ptrDispls[p] = begin;
        }
    }
    int localNnz = 0;
    MPI_Scatter(
        rank == 0 ? nnzCounts.data() : nullptr,
        1, MPI_INT,
        &localNnz, 1, MPI_INT,
        0, MPI_COMM_WORLD
    );
    localValues.resize(localNnz);
    localCols.resize(localNnz);
    MPI_Scatterv(
        rank == 0 ? values.data() : nullptr,
        nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
        localValues.data(), localNnz, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );
    MPI_Scatterv(
        rank == 0 ? col_indices.data() : nullptr,
        nnzCounts.data(), nnzDispls.data(), MPI_INT,
        localCols.data(), localNnz, MPI_INT,
        0, MPI_COMM_WORLD
    );
    MPI_Scatterv(
        rank == 0 ? row_ptr.data() : nullptr,
        ptrCounts.data(), ptrDispls.data(), MPI_INT,
        localRowPtr.data(), localRows + 1, MPI_INT,
        0, MPI_COMM_WORLD
    );
    int offset = localRowPtr[0];
    for (int& ptr : localRowPtr) {
        ptr -= offset;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double distributionTime =
        MPI_Wtime() - distributionStart;
    double computeStart = MPI_Wtime();
    vector<double> localResult =
        sequentialSpMV(localValues, localCols, localRowPtr, x);
    double computeTime = MPI_Wtime() - computeStart;
    double maxDistribution = 0.0;
    double maxCompute = 0.0;
    MPI_Reduce(&distributionTime, &maxDistribution,
        1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&computeTime, &maxCompute,
        1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    vector<int> counts(size), displs(size);
    for (int p = 0; p < size; p++) {
        int begin = p * N / size;
        int end = (p + 1) * N / size;
        counts[p] = end - begin;
        displs[p] = begin;
    }
    vector<double> globalResult;
    if (rank == 0) globalResult.resize(N);
    MPI_Gatherv(
        localResult.data(), localRows, MPI_DOUBLE,
        rank == 0 ? globalResult.data() : nullptr,
        counts.data(), displs.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );
    double localTime = MPI_Wtime() - startTime;
    double mpiTime = 0.0;
    MPI_Reduce(
        &localTime,
        &mpiTime,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );
    if (rank == 0) {
        runTests();
        auto seq = sequentialSpMV(values, col_indices, row_ptr, x);
        omp_set_num_threads(4);
        auto ompResult = parallelSpMV(values, col_indices, row_ptr, x);
        bool mpiCorrect = (seq == globalResult);
        bool ompCorrect = (seq == ompResult);
        cout << "Processes: " << size << endl;
        cout << "MPI time: " << mpiTime << " s" << endl;
        cout << "Distribution time: "
            << maxDistribution << " s\n";
        cout << "Computation time: "
            << maxCompute << " s\n";
        cout << "Sequential vs OpenMP: "
            << (ompCorrect ? "YES" : "NO") << endl;
        vector<int> threadCounts = { 1, 2, 4, 8 };
        const int repeats = 5;
        omp_set_dynamic(0);
        omp_set_num_threads(1);
        for (int k = 0; k < 5; k++) {
            auto warmSeq = sequentialSpMV(values, col_indices, row_ptr, x);
            auto warmOmp = parallelSpMV(values, col_indices, row_ptr, x);
            if (warmSeq != warmOmp) {
                cout << "Warm-up correctness: FAIL\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        vector<double> seqTimes;
        for (int r = 0; r < repeats; r++) {
            double start = omp_get_wtime();
            auto result = sequentialSpMV(values, col_indices, row_ptr, x);
            seqTimes.push_back(omp_get_wtime() - start);
            if (result != seq) return 1;
        }
        sort(seqTimes.begin(), seqTimes.end());
        double seqMedian = seqTimes[repeats / 2];
        ofstream csv("openmp_final.csv");
        csv << "threads,time,speedup,efficiency,correct\n";
        cout << "\nSequential median: " << seqMedian << endl;
        omp_set_dynamic(0);
        double ompBaseline = 0.0;
        for (int threads : threadCounts) {
            omp_set_num_threads(threads);
            vector<double> times;
            bool correct = true;
            auto warmup = parallelSpMV(values, col_indices, row_ptr, x);
            for (int r = 0; r < repeats; r++) {
                double start = omp_get_wtime();
                auto result = parallelSpMV(values, col_indices, row_ptr, x);
                times.push_back(omp_get_wtime() - start);
                if (result != seq) correct = false;
            }
            sort(times.begin(), times.end());
            double median = times[repeats / 2];
            if (threads == 1) {
                ompBaseline = median;
            }
            double speedup = ompBaseline / median;
            double efficiency = speedup / threads * 100.0;
            cout << "Threads: " << threads
                << " | Time: " << median
                << " | Speedup: " << speedup
                << " | Correct: " << (correct ? "YES" : "NO")
                << endl;
            csv << threads << "," << median << ","
                << speedup << "," << efficiency << ","
                << (correct ? "YES" : "NO") << "\n";
        }
        csv.close();
        cout << "OpenMP benchmark saved!\n";
        cout << "Sequential vs MPI: "
            << (mpiCorrect ? "YES" : "NO") << endl;
    }
    MPI_Finalize();
    return 0;
}



