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
std::vector<std::pair<int, int>> changes = {
    {-1, 0}, {-1, -1}, {0, -1}, {1, -1}, {0, 0}}; // Changes in row and column

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;

    double dx_coef = coef * dx;
    double dy_coef = coef * dy;

// Use atomic operations for thread-safe updates
#pragma omp atomic
    particle.ax += dx_coef;
#pragma omp atomic
    particle.ay += dy_coef;
#pragma omp atomic
    neighbor.ax -= dx_coef; // Apply equal and opposite force
#pragma omp atomic
    neighbor.ay -= dy_coef;
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

#pragma omp for
    for (int i = 0; i < grid_size_sq; i += 1) {
        grid[i].particles.clear();
    }

#pragma omp for
    for (int i = 0; i < num_parts; i += 1) { // Assign each particle to a cell based on its position
        int cell_x = parts[i].x / (size / grid_size);
        int cell_y = parts[i].y / (size / grid_size);

        omp_set_lock(&cell_locks[cell_y * grid_size + cell_x]);
        grid[cell_y * grid_size + cell_x].particles.push_back(i);
        omp_unset_lock(&cell_locks[cell_y * grid_size + cell_x]);
    }

#pragma omp for
    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = 0;
        parts[i].ay = 0;
    }

#pragma omp for
    for (int cell_id = 0; cell_id < grid.size(); cell_id += 1) {
        int cell_row = cell_id / grid_size;
        int cell_col = cell_id % grid_size;

        for (int p_id : grid[cell_id].particles) {

            for (auto change : changes) {
                int neighbor_row = cell_row + change.first;
                int neighbor_col = cell_col + change.second;

                if (neighbor_row >= 0 && neighbor_row < grid_size && neighbor_col >= 0 &&
                    neighbor_col < grid_size) {
                    int neighbor_cell_id = neighbor_row * grid_size + neighbor_col;

                    for (int neighbor_p_id : grid[neighbor_cell_id].particles) {
                        if ((cell_id != neighbor_cell_id) ||
                            (p_id < neighbor_p_id)) // Avoid double counting
                            apply_force(parts[p_id], parts[neighbor_p_id]);
                    }
                }
            }
        }
    }

// Move particles
#pragma omp for
    for (int i = 0; i < num_parts; i += 1) {
        move(parts[i], size);
    }
}
