/*

zeid@nid200003:~/hw2-1/build> ./serial
Simulation Time = 0.112458 seconds for 1000 particles.
zeid@nid200003:~/hw2-1/build> ./serial -n 10000
Simulation Time = 1.17675 seconds for 10000 particles.
zeid@nid200003:~/hw2-1/build> ./serial -n 100000
Simulation Time = 11.8413 seconds for 100000 particles.
zeid@nid200003:~/hw2-1/build> ./serial -n 200000
Simulation Time = 25.7562 seconds for 200000 particles.
zeid@nid200003:~/hw2-1/build> ./serial -n 300000
Simulation Time = 41.2297 seconds for 300000 particles.
zeid@nid200003:~/hw2-1/build> ./serial -n 400000
Simulation Time = 56.5778 seconds for 400000 particles.
zeid@nid200003:~/hw2-1/build> ./serial -n 500000
Simulation Time = 71.1062 seconds for 500000 particles.
zeid@nid200003:~/hw2-1/build> ./serial -n 600000
Simulation Time = 84.3825 seconds for 600000 particles.
zeid@nid200003:~/hw2-1/build> ./serial -n 700000
Simulation Time = 100.631 seconds for 700000 particles.
zeid@nid200003:~/hw2-1/build> ./serial -n 800000
Simulation Time = 115.068 seconds for 800000 particles.
zeid@nid200003:~/hw2-1/build> ./serial -n 900000
Simulation Time = 130.764 seconds for 900000 particles.
zeid@nid200003:~/hw2-1/build> ./serial -n 1000000
Simulation Time = 143.545 seconds for 1000000 particles.

*/

#include "common.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <omp.h>

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

    int tid = omp_get_thread_num();
    int max_tid = omp_get_num_threads();

    int grid_size = ceil(size / cutoff);           // Number of cells along one side of the grid
    std::vector<Cell> grid(grid_size * grid_size); // Instantiate grid to be a square

    std::vector<particle_t> sorted_particles(num_parts); // Sorted particles vector
    std::vector<int> map_original_particles_to_sorted(
        num_parts); // Map original particles to sorted vector

    #pragma omp master 
    { // Start of master thread region


    for (int i = 0; i < num_parts; ++i) { // Assign each particle to a cell based on its position
        int cell_x = parts[i].x / (size / grid_size);
        int cell_y = parts[i].y / (size / grid_size);
        grid[cell_y * grid_size + cell_x].particles.push_back(i);
    }

    int index = 0;
    for (int i = 0; i < grid_size * grid_size; ++i) {
        for (int p_id : grid[i].particles) {
            sorted_particles[index] =
                parts[p_id]; // Fill sorted particles vector to improve cache locality
            map_original_particles_to_sorted[p_id] = index;
            index++;
        }
    }

    for (auto& cell : grid) {
        cell.particles.clear(); // Clear cells so we can re-populate them using sorted particles
    }

    for (int i = 0; i < num_parts;
         ++i) { // Re-assign each particle to a cell based on its position in sorted vector
        int cell_x = sorted_particles[i].x / (size / grid_size);
        int cell_y = sorted_particles[i].y / (size / grid_size);
        grid[cell_y * grid_size + cell_x].particles.push_back(i);
    }

    for (int cell_id = 0; cell_id < grid.size(); ++cell_id) { // Loop over every cell
        int cell_row = cell_id / grid_size;
        int cell_col = cell_id % grid_size;

        for (int p_id : grid[cell_id].particles) { // Loop over every particle in that cell
            sorted_particles[p_id].ax = sorted_particles[p_id].ay = 0;
		}
	}

    } // End of master thread region

    #pragma omp barrier

    for (int cell_id = tid; cell_id < grid.size(); cell_id += max_tid) { // 0 < cell_id < grid.size()
        int cell_row = cell_id / grid_size;
        int cell_col = cell_id % grid_size;

        for (int p_id : grid[cell_id].particles) { // Loop over every particle in that cell

            for (int row_change = -1; row_change <= 1;
                 ++row_change) { // Iterate over all 8 neighboring cells and current cell
                for (int col_change = -1; col_change <= 1; ++col_change) {
                    int neighbor_row = cell_row + row_change;
                    int neighbor_col = cell_col + col_change;

                    if (neighbor_row >= 0 && neighbor_row < grid_size && neighbor_col >= 0 &&
                        neighbor_col < grid_size) { // Make sure within grid bounds (border case)
                        int neighbor_cell_id = neighbor_row * grid_size + neighbor_col;

                        for (int neighbor_p_id :
                             grid[neighbor_cell_id].particles) { // Loop over all particles in
                                                                 // current cell and current cell
                            apply_force(sorted_particles[p_id],
                                        sorted_particles[neighbor_p_id]); // Compute forces
                        }
                    }
                }
            }
        }
    }


    // Move particles
    for (int i = tid; i < num_parts; i += max_tid) { // 0 < i < num_parts
        move(sorted_particles[i], size);
    }
    #pragma omp master
    {
    std::copy(sorted_particles.begin(), sorted_particles.end(),
              parts); // copies updated particles back to parts
    }
    #pragma omp barrier

}
