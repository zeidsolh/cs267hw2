#include "common.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>

struct Cell {
    std::vector<int> particles; // Indices of particles in each cell
};

void apply_force(particle_t& particle, particle_t& neighbor) {
    double col_change = neighbor.x - particle.x;
    double row_change = neighbor.y - particle.y;
    double r2 = col_change * col_change + row_change * row_change;
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * col_change;
    particle.ay += coef * row_change;
}

void move(particle_t& p, double size) {
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
    // Initialization logic if needed
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    int grid_size = ceil(size / cutoff);
    std::vector<Cell> grid(grid_size * grid_size);

#pragma omp parallel for
    for (int i = 0; i < num_parts; ++i) {
        int cell_x = parts[i].x / (size / grid_size);
        int cell_y = parts[i].y / (size / grid_size);
#pragma omp critical
        { grid[cell_y * grid_size + cell_x].particles.push_back(i); }
    }

    std::vector<particle_t> sorted_particles(num_parts);
    std::vector<int> map_original_particles_to_sorted(num_parts);
    int index = 0;
    for (int i = 0; i < grid_size * grid_size; ++i) {
        for (int p_id : grid[i].particles) {
            sorted_particles[index] = parts[p_id];
            map_original_particles_to_sorted[p_id] = index;
            index++;
        }
    }

    for (auto& cell : grid) {
        cell.particles.clear();
    }

#pragma omp parallel for
    for (int i = 0; i < num_parts; ++i) {
        int cell_x = sorted_particles[i].x / (size / grid_size);
        int cell_y = sorted_particles[i].y / (size / grid_size);
#pragma omp critical
        { grid[cell_y * grid_size + cell_x].particles.push_back(i); }
    }

#pragma omp parallel for
    for (int cell_id = 0; cell_id < grid.size(); ++cell_id) {
        int cell_row = cell_id / grid_size;
        int cell_col = cell_id % grid_size;

        for (int p_id : grid[cell_id].particles) {
            sorted_particles[p_id].ax = sorted_particles[p_id].ay = 0;

            for (int row_change = -1; row_change <= 1; ++row_change) {
                for (int col_change = -1; col_change <= 1; ++col_change) {
                    int neighbor_row = cell_row + row_change;
                    int neighbor_col = cell_col + col_change;

                    if (neighbor_row >= 0 && neighbor_row < grid_size && neighbor_col >= 0 &&
                        neighbor_col < grid_size) {
                        int neighbor_cell_id = neighbor_row * grid_size + neighbor_col;

                        for (int neighbor_p_id : grid[neighbor_cell_id].particles) {
                            apply_force(sorted_particles[p_id], sorted_particles[neighbor_p_id]);
                        }
                    }
                }
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < num_parts; ++i) {
        move(sorted_particles[i], size);
    }

    std::copy(sorted_particles.begin(), sorted_particles.end(), parts);
}