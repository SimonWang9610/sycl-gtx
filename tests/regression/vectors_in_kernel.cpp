#include "../common.h"

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <stdexcept>
#include <string>

using namespace cl::sycl;
class Matrix;

float* generate(int row, int col, bool tag) {

	float* arr = (float*)malloc(row * col * sizeof(float));

	int total = row * col;
	srand(time(NULL));

	for (int i = 0; i < total; i++) {

		if (tag) {
			arr[i] = 0;
		} else {
			arr[i] = (float) rand() / RAND_MAX * 0.01;
		}
	}
	return arr;
}

void print(float* arr, int row, int col) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      std::cout << arr[i * col + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "----------------" << std::endl;
}

void multiplication(queue& Q, float* left, float* right, float* result, int N) {
	buffer<float, 1> buf_left(left, N * N);
	buffer<float, 1> buf_right(right, N * N);
	buffer<float, 2> buf_result(result, range<2>(N, N));

	Q.submit([&] (handler& cgh) {
		auto acc_left = buf_left.get_access<access::mode::read>(cgh);
		auto acc_right = buf_right.get_access<access::mode::read>(cgh);
		auto acc_result = buf_result.get_access<access::mode::discard_write>(cgh);

		cgh.parallel_for<class Matrix>(range<2>(N, N), [=] (id<2> index) {
      int1 row = index[0];
      int1 col = index[1];
      float1 temp = 0.;
      SYCL_FOR(int1 i = 0, i < N, i++) {
        temp += acc_left[row * N + i] * acc_right[i * N + col];
      }
      SYCL_END;
      acc_result[index] = temp;
		});
	});

  Q.wait();
  Q.stats();
}

void addition(queue& Q, float* left, float* right, float* result, int N) {
	buffer<float, 1> buf_left(left, N);
	buffer<float, 1> buf_right(right, N);
	buffer<float, 1> buf_result(result, N);

  Q.submit([&](handler& cgh) {
    auto acc_left = buf_left.get_access<access::mode::read>(cgh);
    auto acc_right = buf_right.get_access<access::mode::read>(cgh);
    auto acc_result = buf_result.get_access<access::mode::discard_write>(cgh);

    cgh.parallel_for<class Matrix>(range<1>(N), [=](id<1> index) {
      acc_result[index] = acc_left[index] + acc_right[index];
    });
  });

  Q.wait();
  Q.stats();
}

void validate_matrix(float* left, float* right, float* result, int N);
void validate_addition(float* left, float* right, float* result, int N);

int main(int argc, char* argv[]) {

  std::string first = argv[1];
  std::string second = argv[2];
  size_t first_pos, second_pos;

  int M = std::stoi(first, &first_pos);
  int epoch = std::stoi(second, &second_pos);

  int N = 10000;

  float* left = generate(M, N, false);
  float* right = generate(M, N, false);
  float* result = generate(M, N, true);

  std::cout << "generated data" << std::endl;

  clock_t total = 0;
  clock_t warmup = 0;
  clock_t calculation = 0;

  for (int i = 0; i < epoch; i++) {
    auto start = clock();
    queue Q;
    auto end_warmup = clock();
    addition(Q, left, right, result, M*N);
    auto end_calculation = clock();
    std::cout << "*******[" << i << "]**********" << std::endl;

    warmup += end_warmup - start;
    calculation += end_calculation - end_warmup;
    total += end_calculation - start;
  }
  
  std::cout << "time on warmup: " << warmup / epoch << std::endl;
  std::cout << "time on calculation: " << calculation / epoch << std::endl;
  std::cout << "total on execution: " << total / epoch << std::endl;

  validate_addition(left, right, result, M*N);

  return 0;
}

void validate_addition(float* left, float* right, float* result, int N) {
		std::cout << "validating..." << std::endl;
	int count = 0;

	for (int i = 0; i < N; i++) {
		float temp = left[i] + right[i];
		
		float diff = abs(temp - result[i]);

		if (diff > 2e-7) {
			count += 1;
			std::cout << "error" << std::endl;
		}
	}

	if (count == 0) {
		std::cout << "successfully" <<std::endl;
	} else {
		std::cout << "someing wrong: " << count << std::endl;
	}

}

void validate_matrix(float* left, float* right, float* result, int N) {
  std::cout << "validating..." << std::endl;
  int count = 0;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float temp = 0;

			for (int k = 0; k < N; k++) {
				temp += left[i * N + k] * right[k * N + j];
			}
      // std::cout << temp << " ";
			float diff = abs(temp - result[j * N + i]);
			if (diff > 2e-7) {
        count += 1;
				std::cout << diff << std::endl;
			}
		}
    // std::cout << std::endl;
	}

  if (count == 0) {
    std::cout << "successfully" << std::endl;
  } else {
    std::cout << "something wrong: " << count << std::endl;
  }
}