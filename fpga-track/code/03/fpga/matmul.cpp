#include <iostream>
using namespace std;

#define SIZE 4

void matmul(int A[SIZE][SIZE], int B[SIZE][SIZE], int C[SIZE][SIZE]) {

#pragma HLS PIPELINE
#pragma HLS ARRAY_PARTITION variable = A complete dim = 2
#pragma HLS ARRAY_PARTITION variable = B complete dim = 1

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      C[i][j] = 0;
      for (int k = 0; k < SIZE; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
int main() {
  int A[SIZE][SIZE] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {1, 1, 1, 1}, {2, 2, 2, 2}};

  int B[SIZE][SIZE] = {{1, 0, 0, 1},
                       {0, 1, 1, 0},
                       {
                           1,
                           1,
                           0,
                           0,
                       },
                       {0, 0, 1, 1}};
  int C[SIZE][SIZE];

  matmul(A, B, C);
  cout << "Result Matrix:" << endl;
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      cout << C[i][j] << " ";
    }
    cout << endl;
  }
  return 0;
}
