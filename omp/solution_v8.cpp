#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <vector>

const double pi = 3.14159265358979323846;

const double L = 1;
const int N = 256;
const double T = 0.0001;
const int K = 20;
const double theta = T / K;

// 0..N
const int mesh_size = N + 1;
const double L_x{L}, L_y{L}, L_z{L};
const double h_x{L_x / N}, h_y{L_y / N}, h_z{L_z / N};
const double a_t =
    pi * std::sqrt(4 / (L_x * L_x) + 16 / (L_y * L_y) + 36 / (L_z * L_z));
const double a_2 = 1;

double analytical(const double x, const double y, const double z,
                  const double t) {
  return std::sin(2 * pi * x / L_x) * std::sin(4 * pi * y / L_y) *
         std::sin(6 * pi * z / L_z) * std::cos(a_t * t);
}

template <class T> using matrix = std::vector<std::vector<T>>;

template <class T> using tensor3 = std::vector<matrix<T>>;

using double_tensor3 = tensor3<double>;

double_tensor3 get_double_tensor(const int size) {
  double_tensor3 tnsr;
  for (int i = 0; i < size; i++) {
    matrix<double> m;
    for (int j = 0; j < size; j++) {
      m.push_back(std::vector<double>(size));
    }
    tnsr.push_back(m);
  }
  return tnsr;
}

double_tensor3 get_analytical_tensor(const int size, const double t) {
  auto tnsr = get_double_tensor(size);
  for (int i = 0; i < mesh_size; i++) {
    for (int j = 0; j < mesh_size; j++) {
      for (int k = 0; k < mesh_size; k++) {
        tnsr[i][j][k] = analytical(i * h_x, j * h_y, k * h_z, t);
      }
    }
  }

  return tnsr;
}

double delta_h(const double_tensor3 &u, const int i, const int j, const int k) {
  double res{0};
  res += (u[i - 1][j][k] - 2 * u[i][j][k] + u[i + 1][j][k]) / (h_x * h_x);
  res += (u[i][j - 1][k] - 2 * u[i][j][k] + u[i][j + 1][k]) / (h_y * h_y);
  res += (u[i][j][k - 1] - 2 * u[i][j][k] + u[i][j][k + 1]) / (h_z * h_z);
  return res;
}

void get_neightbours(int &idx, int &left_idx, int &right_idx) {
  if (idx == 0 || idx == N) {
    idx = N;
    left_idx = N - 1;
    right_idx = 1;
    return;
  }
  left_idx = idx - 1;
  right_idx = idx + 1;
}

double delta_h_with_boundaries(const double_tensor3 &u, int i, int j, int k) {
  double res{0};
  int left_i, right_i;
  int left_j, right_j;
  int left_k, right_k;
  get_neightbours(i, left_i, right_i);
  get_neightbours(j, left_j, right_j);
  get_neightbours(k, left_k, right_k);

  res += (u[left_i][j][k] - 2 * u[i][j][k] + u[right_i][j][k]) / (h_x * h_x);
  res += (u[i][left_j][k] - 2 * u[i][j][k] + u[i][right_j][k]) / (h_y * h_y);
  res += (u[i][j][left_k] - 2 * u[i][j][k] + u[i][j][right_k]) / (h_z * h_z);
  return res;
}

double_tensor3 get_u1(const double_tensor3 &u0) {
  auto tnsr = get_double_tensor(mesh_size);
  for (int i = 0; i < mesh_size; i++) {
    for (int j = 0; j < mesh_size; j++) {
      for (int k = 0; k < mesh_size; k++) {
        if (i > 0 && i < (N + 1) && j > 0 && j < (N + 1) && k > 0 &&
            k < (N + 1)) {
          tnsr[i][j][k] =
              u0[i][j][k] +
              a_2 * (theta * theta / 2) * delta_h_with_boundaries(u0, i, j, k);
          continue;
        }
        tnsr[i][j][k] = analytical(i * h_x, j * h_y, k * h_z, theta);
      }
    }
  }

  return tnsr;
}

double get_u_n_plus_1_element(const int i, const int j, const int k,
                              const double_tensor3 &u_n,
                              const double_tensor3 &u_n_minus_1) {
  return a_2 * delta_h_with_boundaries(u_n, i, j, k) * theta * theta +
         2 * u_n[i][j][k] - u_n_minus_1[i][j][k];
}

void fill_inner_region(double_tensor3 &u_n_plus_1, const double_tensor3 &u_n,
                       const double_tensor3 &u_n_minus_1) {
#pragma omp parallel for collapse(3)
  for (int i = 0; i < mesh_size; i++) {
    for (int j = 0; j < mesh_size; j++) {
      for (int k = 0; k < mesh_size; k++) {
        u_n_plus_1[i][j][k] = get_u_n_plus_1_element(i, j, k, u_n, u_n_minus_1);
      }
    }
  }
}

double total_error(const double_tensor3 &analytical,
                   const double_tensor3 &calculated) {
  double err{0};
  for (int i = 1; i < mesh_size - 1; i++) {
    for (int j = 1; j < mesh_size - 1; j++) {
      for (int k = 1; k < mesh_size - 1; k++) {
        err += std::abs(analytical[i][j][k] - calculated[i][j][k]);
      }
    }
  }
  return err;
}

// for plots
void print_tensors(const double_tensor3 &analytical,
                   const double_tensor3 &calculated,
                   const bool skip_every_second = false) {
  std::cout << "{ \"analytical\": ";
  std::cout << std::setprecision(16) << std::scientific << "[";
  for (int i = 1; i < mesh_size - 1; i++) {
    if (skip_every_second && (i % 2) == 0)
      continue;
    std::cout << "[";
    for (int j = 1; j < mesh_size - 1; j++) {
      if (skip_every_second && (j % 2) == 0)
        continue;
      std::cout << "[";
      for (int k = 1; k < mesh_size - 1; k++) {
        if (skip_every_second && (k % 2) == 0)
          continue;
        std::cout << analytical[i][j][k] << ", ";
      }
      std::cout << "], ";
    }
    std::cout << "], ";
  }

  std::cout << "], \"calculated\": [";
  for (int i = 1; i < mesh_size - 1; i++) {
    if (skip_every_second && (i % 2) == 0)
      continue;
    std::cout << "[";
    for (int j = 1; j < mesh_size - 1; j++) {
      if (skip_every_second && (j % 2) == 0)
        continue;
      std::cout << "[";
      for (int k = 1; k < mesh_size - 1; k++) {
        if (skip_every_second && (k % 2) == 0)
          continue;
        std::cout << calculated[i][j][k] << ", ";
      }
      std::cout << "], ";
    }
    std::cout << "], ";
  }
  std::cout << "] }" << std::endl;
}

int main() {
  auto t_start = std::chrono::high_resolution_clock::now();
  auto u_n_minus_1 = get_analytical_tensor(mesh_size, 0);
  auto u_n = get_u1(u_n_minus_1);
  auto u_n_plus_1 = get_double_tensor(mesh_size);

  double_tensor3 current_analytical;
  for (int n = 2; n <= K; n++) {

    fill_inner_region(u_n_plus_1, u_n, u_n_minus_1);

    // current_analytical = get_analytical_tensor(mesh_size, theta * n);
    // std::cout << total_error(current_analytical, u_n_plus_1) << ", ";
    // print_tensors(current_analytical, u_n_plus_1);

    u_n_minus_1 = u_n;
    u_n = u_n_plus_1;
  }
  const auto t_stop = std::chrono::high_resolution_clock::now();
  std::cout << "threads: " << omp_get_max_threads() << std::endl;
  std::cout << "N: " << N << std::endl;
  std::cout << "L: " << L << std::endl;

  current_analytical = get_analytical_tensor(mesh_size, theta * 20);
  std::cout << "error: " << total_error(current_analytical, u_n_plus_1)
            << std::endl;
  std::cout
      << "run time: "
      << std::chrono::duration<double, std::milli>(t_stop - t_start).count()
      << std::endl;
  return 0;
}
