#include "common.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>

struct Cell {
    std::vector<int> particles; // Indices of particles in each cell
};

// Global variables
std::vector<Cell> grid; // Grid of cells
int grid_size;          // Number of cells along one side of the grid
int grid_size_sq;       // grid_size * grid_size
omp_lock_t* cell_locks; // Locks for each cell in the grid

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    grid_size = ceil(size / cutoff);
    grid_size_sq = grid_size * grid_size;
    grid.resize(grid_size_sq);
    cell_locks = new omp_lock_t[grid_size_sq];
    for (int i = 0; i < grid_size_sq; ++i) {
        omp_init_lock(&cell_locks[i]);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    int tid = omp_get_thread_num();
    int max_tid = omp_get_num_threads();
#pragma omp for
    for (int i = tid; i < num_parts;
         i += max_tid) { // Assign each particle to a cell based on its position
        int cell_x = parts[i].x / (size / grid_size);
        int cell_y = parts[i].y / (size / grid_size);

        omp_set_lock(&cell_locks[cell_y * grid_size + cell_x]);
        grid[cell_y * grid_size + cell_x].particles.push_back(i);
        omp_unset_lock(&cell_locks[cell_y * grid_size + cell_x]);
    }
#pragma omp barrier
#pragma omp for
    for (int cell_id = tid; cell_id < grid.size(); cell_id += max_tid) { // Loop over every cell
        int cell_row = cell_id / grid_size;
        int cell_col = cell_id % grid_size;

        for (int p_id : grid[cell_id].particles) { // Loop over every particle in that cell
            parts[p_id].ax = parts[p_id].ay = 0;

            for (int row_change = -1; row_change <= 1;
                 ++row_change) { // Iterate over all 8 neighboring cells and current cell
                for (int col_change = -1; col_change <= 1; ++col_change) {
                    int neighbor_row = cell_row + row_change;
                    int neighbor_col = cell_col + col_change;

                    if (neighbor_row >= 0 && neighbor_row < grid_size && neighbor_col >= 0 &&
                        neighbor_col < grid_size) { // Make sure within grid bounds (border case)
                        int neighbor_cell_id = neighbor_row * grid_size + neighbor_col;

                        for (int neighbor_p_id :
                             grid[neighbor_cell_id]
                                 .particles) { // Loop over all particles in current cell and
                                               // neighboring cells
                            apply_force(parts[p_id], parts[neighbor_p_id]); // Compute forces
                        }
                    }
                }
            }
        }
    }

// Move particles
#pragma omp barrier
#pragma omp for
    for (int i = tid; i < num_parts; i += max_tid) {
        move(parts[i], size);
    }
}
