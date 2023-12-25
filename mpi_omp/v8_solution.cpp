#include "mpi.h"
#ifdef USE_OMP
#include <omp.h>
#endif
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

const double K = 20;
const double T = 0.00001;
const double tau = T / K;
const double L_x{L}, L_y{L}, L_z{L};
const double h_x{L_x / N}, h_y{L_y / N}, h_z{L_z / N};
double h = h_x;

const double a_t =
    M_PI * std::sqrt(4 / (L_x * L_x) + 16 / (L_y * L_y) + 32 / (L_z * L_z));
const double a_2 = 1;

using int3 = std::array<int, 3>;

double u_analytical(size_t ix, size_t iy, size_t iz, double time) {
  return std::sin(2 * M_PI * ix / (N - 1)) * std::sin(4 * M_PI * iy / (N - 1)) *
         std::sin(6 * M_PI * iz / (N - 1)) * std::cos(a_t * time);
}

void fill_u_analytical(std::vector<double> &u, size_t grid_size,
                       const int3 &block_size, const int3 &block_position) {
  for (size_t z = 0; z < block_size[2]; z++) {
    size_t iz = z + block_size[2] * block_position[2];
    size_t off_z = block_size[0] * block_size[1] * z;
    for (size_t y = 0; y < block_size[1]; y++) {
      size_t iy = y + block_size[1] * block_position[1];
      size_t off_y = block_size[0] * y;
      for (size_t x = 0; x < block_size[0]; x++) {
        size_t ix = x + block_size[0] * block_position[0];
        size_t off_x = x;
        u[off_z + off_y + off_x] = u_analytical(ix, iy, iz, 0.0);
      }
    }
  }
}

void solve(int grid_size, int size, int rank) {

  int3 grid = {0, 0, 0};

  switch (size) {
  case 1:
    grid[0] = 1;
    grid[1] = 1;
    grid[2] = 1;
    break;
  case 2:
    grid[0] = 2;
    grid[1] = 1;
    grid[2] = 1;
    break;
  case 4:
    grid[0] = 2;
    grid[1] = 2;
    grid[2] = 1;
    break;
  case 8:
    grid[0] = 2;
    grid[1] = 2;
    grid[2] = 2;
    break;
  case 16:
    grid[0] = 2;
    grid[1] = 2;
    grid[2] = 4;
    break;
  case 32:
    grid[0] = 2;
    grid[1] = 4;
    grid[2] = 4;
    break;
  default:
    if (rank == 0) {
      std::cout << "ERROR: Count of process" << std::endl;
    }
    return;
  }

  int3 block_size = {grid_size / grid[0], grid_size / grid[1],
                     grid_size / grid[2]};
  int total_block_size = block_size[0] * block_size[1] * block_size[2];

  int3 periods = {0, 0, 0};
  int3 block_position;

  int block_up, block_down, block_left, block_right, block_close, block_far;

  MPI_Comm CART_COMM;
  MPI_Cart_create(MPI_COMM_WORLD, 3, grid.data(), periods.data(), 0,
                  &CART_COMM);
  MPI_Cart_coords(CART_COMM, rank, 3, block_position.data());
  MPI_Cart_shift(CART_COMM, 0, 1, &block_left, &block_right);
  MPI_Cart_shift(CART_COMM, 1, 1, &block_close, &block_far);
  MPI_Cart_shift(CART_COMM, 2, -1, &block_up, &block_down);

  std::vector<double> u_n_plus_1(total_block_size);
  std::vector<double> u_n(total_block_size);
  std::vector<double> u_n_minus_1(total_block_size);

  fill_u_analytical(u_n, grid_size, block_size, block_position);

  MPI_Datatype type_xOy_surface, type_yOz_surface, type_xOz_surface;

  int size_xOy = block_size[0] * block_size[1];
  MPI_Type_vector(block_size[0] * block_size[1], 1, 1, MPI_DOUBLE,
                  &type_xOy_surface);
  MPI_Type_commit(&type_xOy_surface);

  int size_yOz = block_size[1] * block_size[2];
  MPI_Type_vector(block_size[2] * block_size[1], 1, block_size[0], MPI_DOUBLE,
                  &type_yOz_surface);
  MPI_Type_commit(&type_yOz_surface);

  int size_xOz = block_size[0] * block_size[2];
  MPI_Type_vector(block_size[2], block_size[0], size_xOy, MPI_DOUBLE,
                  &type_xOz_surface);
  MPI_Type_commit(&type_xOz_surface);

  std::vector<double> down_buffer(size_xOy);
  std::vector<double> up_buffer(size_xOy);

  std::vector<double> left_buffer(size_yOz);
  std::vector<double> right_buffer(size_yOz);

  std::vector<double> close_buffer(size_xOz);
  std::vector<double> far_buffer(size_xOz);

  MPI_Barrier(MPI_COMM_WORLD);

  double max_diff_all = 0.0;
  for (size_t step = 0; step < K; step++) {

    MPI_Request send_up, recv_up, send_down, recv_down, send_left, recv_left,
        send_right, recv_right, send_close, recv_close, send_far, recv_far;

    if (grid[0] == 1) {
      for (int z = 0; z < grid_size; z++) {
        for (int y = 0; y < grid_size; y++) {
          right_buffer[z * block_size[1] + y] =
              u_n[z * size_xOy + y * block_size[0] + 1];
          u_n[z * size_xOy + y * block_size[0]] =
              u_n[z * size_xOy + (y + 1) * block_size[0] - 1];
        }
      }
    } else {
      if (block_left == MPI_PROC_NULL) {
        MPI_Isend(&(u_n[1]), 1, type_yOz_surface, block_right, 3,
                  MPI_COMM_WORLD, &send_left);
        MPI_Irecv(&(u_n[0]), 1, type_yOz_surface, block_right, 2,
                  MPI_COMM_WORLD, &recv_left);
      } else {
        MPI_Isend(&(u_n[0]), 1, type_yOz_surface, block_left, 3, MPI_COMM_WORLD,
                  &send_left);
        MPI_Irecv(left_buffer.data(), size_yOz, MPI_DOUBLE, block_left, 2,
                  MPI_COMM_WORLD, &recv_left);
      }

      if (block_right == MPI_PROC_NULL) {
        MPI_Isend(&(u_n[block_size[0] - 1]), 1, type_yOz_surface, block_left, 2,
                  MPI_COMM_WORLD, &send_right);
        MPI_Irecv(right_buffer.data(), size_yOz, MPI_DOUBLE, block_left, 3,
                  MPI_COMM_WORLD, &recv_right);
      } else {
        MPI_Isend(&(u_n[block_size[0] - 1]), 1, type_yOz_surface, block_right,
                  2, MPI_COMM_WORLD, &send_right);
        MPI_Irecv(right_buffer.data(), size_yOz, MPI_DOUBLE, block_right, 3,
                  MPI_COMM_WORLD, &recv_right);
      }
      MPI_Wait(&send_left, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_left, MPI_STATUS_IGNORE);
      MPI_Wait(&send_right, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_right, MPI_STATUS_IGNORE);
    }

    if (grid[1] == 1) {
      for (int z = 0; z < block_size[2]; z++) {
        for (int x = 0; x < block_size[0]; x++) {
          far_buffer[z * block_size[0] + x] =
              u_n[z * size_xOy + x + block_size[0]];
          u_n[z * size_xOy + x] = u_n[(z + 1) * size_xOy + x - block_size[0]];
        }
      }
    } else {
      if (block_close == MPI_PROC_NULL) {
        MPI_Isend(&(u_n[block_size[0]]), 1, type_xOz_surface, block_far, 5,
                  MPI_COMM_WORLD, &send_close);
        MPI_Irecv(&(u_n[0]), 1, type_xOz_surface, block_far, 4, MPI_COMM_WORLD,
                  &recv_close);
      } else {
        MPI_Isend(&(u_n[0]), 1, type_xOz_surface, block_close, 5,
                  MPI_COMM_WORLD, &send_close);
        MPI_Irecv(close_buffer.data(), size_xOz, MPI_DOUBLE, block_close, 4,
                  MPI_COMM_WORLD, &recv_close);
      }

      if (block_far == MPI_PROC_NULL) {
        MPI_Isend(&(u_n[size_xOy - block_size[0]]), 1, type_xOz_surface,
                  block_close, 4, MPI_COMM_WORLD, &send_far);
        MPI_Irecv(far_buffer.data(), size_xOz, MPI_DOUBLE, block_close, 5,
                  MPI_COMM_WORLD, &recv_far);
      } else {
        MPI_Isend(&(u_n[size_xOy - block_size[0]]), 1, type_xOz_surface,
                  block_far, 4, MPI_COMM_WORLD, &send_far);
        MPI_Irecv(far_buffer.data(), size_xOz, MPI_DOUBLE, block_far, 5,
                  MPI_COMM_WORLD, &recv_far);
      }

      MPI_Wait(&send_close, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_close, MPI_STATUS_IGNORE);
      MPI_Wait(&send_far, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_far, MPI_STATUS_IGNORE);
    }

    if (grid[2] == 1) {
      for (int y = 0; y < block_size[1]; y++) {
        for (int x = 0; x < block_size[0]; x++) {
          up_buffer[y * block_size[0] + x] =
              u_n[y * block_size[0] + x + size_xOy];
          u_n[y * block_size[0] + x] =
              u_n[size + y * block_size[0] + x - size_xOy];
        }
      }
    } else {
      if (block_down == MPI_PROC_NULL) {
        MPI_Isend(&(u_n[size_xOy]), 1, type_xOy_surface, block_up, 1,
                  MPI_COMM_WORLD, &send_down);
        MPI_Irecv(&(u_n[0]), 1, type_xOy_surface, block_down, 0, MPI_COMM_WORLD,
                  &recv_down);
      } else {
        MPI_Isend(&(u_n[0]), 1, type_xOy_surface, block_down, 1, MPI_COMM_WORLD,
                  &send_down);
        MPI_Irecv(down_buffer.data(), size_xOy, MPI_DOUBLE, block_down, 0,
                  MPI_COMM_WORLD, &recv_down);
      }

      if (block_up == MPI_PROC_NULL) {
        MPI_Isend(&(u_n[total_block_size - size_xOy]), 1, type_xOy_surface,
                  block_up, 0, MPI_COMM_WORLD, &send_up);
        MPI_Irecv(up_buffer.data(), size_xOy, MPI_DOUBLE, block_down, 1,
                  MPI_COMM_WORLD, &recv_up);
      } else {
        MPI_Isend(&(u_n[total_block_size - size_xOy]), 1, type_xOy_surface,
                  block_up, 0, MPI_COMM_WORLD, &send_up);
        MPI_Irecv(up_buffer.data(), size_xOy, MPI_DOUBLE, block_up, 1,
                  MPI_COMM_WORLD, &recv_up);
      }

      MPI_Wait(&send_down, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_down, MPI_STATUS_IGNORE);
      MPI_Wait(&send_up, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_up, MPI_STATUS_IGNORE);
    }

    double coeff_tau = (step == 0) ? 0.5 : 1;
    double coeff_u_n = (step == 0) ? 1 : 2;

    int z_begin = (block_position[2] == 0 ? 1 : 0);
    int z_end = block_size[2];

    int y_begin = (block_position[1] == 0 ? 1 : 0);
    int y_end = block_size[1];

    int x_begin = (block_position[0] == 0 ? 1 : 0);
    int x_end = block_size[0];

#pragma omp parallel for collapse(3)
    for (int z = z_begin; z < z_end; z++) {
      for (int y = y_begin; y < y_end; y++) {
        for (int x = x_begin; x < x_end; x++) {
          int idx = z * size_xOy + y * block_size[0] + x;

          double far_term =
              ((y == (block_size[1] - 1)) ? far_buffer[z * block_size[0] + x]
                                          : u_n[idx + block_size[0]]);
          double close_term = ((y == 0) ? close_buffer[z * block_size[0] + x]
                                        : u_n[idx - block_size[0]]);

          double left_term =
              ((x == 0) ? left_buffer[z * block_size[1] + y] : u_n[idx - 1]);
          double right_term =
              ((x == (block_size[0] - 1)) ? right_buffer[z * block_size[1] + y]
                                          : u_n[idx + 1]);

          double up_term =
              ((z == (block_size[2] - 1)) ? up_buffer[y * block_size[0] + x]
                                          : u_n[idx + size_xOy]);
          double down_term = ((z == 0) ? down_buffer[y * block_size[0] + x]
                                       : u_n[idx - size_xOy]);

          u_n_plus_1[idx] = coeff_tau * a_2 * pow((tau / h), 2) *
                                (up_term + down_term + close_term + far_term +
                                 left_term + right_term - 6 * u_n[idx]) +
                            coeff_u_n * u_n[idx] - u_n_minus_1[idx];
        }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (step == K - 1) {
      double local_diff = 0.0;
      double global_diff = 0.0;

      for (size_t z = 0; z < block_size[2]; z++) {
        size_t iz = z + block_size[2] * block_position[2];
        size_t off_z = size_xOy * z;
        for (size_t y = 0; y < block_size[1]; y++) {
          size_t iy = y + block_size[1] * block_position[1];
          size_t off_y = block_size[0] * y;
          for (size_t x = 0; x < block_size[0]; x++) {
            size_t ix = x + block_size[0] * block_position[0];
            size_t off_x = x;
            double time = tau * (step + 1);
            double disc = std::abs(u_n_plus_1[off_z + off_y + off_x] -
                                   u_analytical(ix, iy, iz, time));
            local_diff += disc;
          }
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Reduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, 0,
                 MPI_COMM_WORLD);

      if (rank == 0) {
        std::cout << "error: " << global_diff << std::endl;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::swap(u_n_minus_1, u_n);
    std::swap(u_n, u_n_plus_1);

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

int main(int argc, char **argv) {
  size_t block_size;
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  block_size = N;

  auto start = MPI_Wtime();
  solve(block_size, size, rank);
  MPI_Barrier(MPI_COMM_WORLD);
  auto end = MPI_Wtime();

  if (rank == 0) {
    std::cout << "run time: " << (end - start) * 1000 << std::endl;
#ifdef USE_OMP
    std::cout << "threads: " << omp_get_max_threads() << std::endl;
#endif
    std::cout << "processes: " << size << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "L: " << L << std::endl;
  }

  MPI_Finalize();
  return 0;
}
