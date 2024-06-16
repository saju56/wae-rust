extern crate rand;
extern crate plotters;

use plotters::prelude::*;
use plotters::style::{Color, RED};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::f64;

const MU: usize = 50; // Number of best solutions to keep
const H: usize = 5; // History size
const CC: f64 = 0.05; // Learning rate for covariance matrix
const CD: f64 = 0.05; // Learning rate for differential weights
const EPSILON: f64 = 0.1; // Small constant

static mut DIMENSIONS: usize = 3; // Dimensionality of the problem
static mut LAMBDA: usize = 7; // Population size (will be updated)
static mut MAX_GENERATIONS: usize = 30000; // Max number of generations (will be updated)

fn main() {
    unsafe {
        DIMENSIONS = 3; // Set your desired dimensions
        LAMBDA = 4 + (3.0 * (DIMENSIONS as f64).ln()).floor() as usize;
        // MAX_GENERATIONS = 10000 * DIMENSIONS;
        MAX_GENERATIONS = 1400;
    }

    println!("LAMBDA: {}", unsafe { LAMBDA });
    println!("MAX_GENERATIONS: {}", unsafe { MAX_GENERATIONS });

    let mut des = DES::new();
    des.run(|x| x.iter().map(|&xi| xi * xi).sum(), "sum_of_squares.png"); // Optimize the sum of squares function
}

struct DES {
    population: Vec<Vec<f64>>,
    centroid: Vec<f64>,
    rng: rand::rngs::ThreadRng,
    uniform: Uniform<f64>,
    generation: usize,
}

impl DES {
    fn new() -> Self {
        let dimensions = unsafe { DIMENSIONS };
        let lambda = unsafe { LAMBDA };
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(0.0, 1.0);

        let population: Vec<Vec<f64>> = (0..lambda)
            .map(|_| (0..dimensions).map(|_| uniform.expect("REASON").sample(&mut rng)).collect())
            .collect();

        let centroid: Vec<f64> = (0..dimensions)
            .map(|d| population.iter().map(|xi| xi[d]).sum::<f64>() / lambda as f64)
            .collect();

        Self {
            population,
            centroid,
            rng,
            uniform: uniform.unwrap(),
            generation: 1,
        }
    }

    fn evaluate<F>(&self, fitness_fn: F) -> Vec<f64>
    where
        F: Fn(&Vec<f64>) -> f64,
    {
        self.population
            .iter()
            .map(|individual| fitness_fn(individual))
            .collect()
    }

    fn stop_condition(&self) -> bool {
        self.generation >= unsafe { MAX_GENERATIONS }
    }

    fn run<F>(&mut self, fitness_fn: F, plot_file: &str)
    where
        F: Fn(&Vec<f64>) -> f64,
    {
        let mut fitness_values = self.evaluate(&fitness_fn);
        let mut best_fitness_history = vec![];

        while !self.stop_condition() {
            let best_fitness = fitness_values.iter().cloned().fold(f64::INFINITY, f64::min);
            best_fitness_history.push(best_fitness);
            println!("Generation {}: Best fitness = {}", self.generation, best_fitness);

            let old_centroid = self.centroid.clone();

            self.centroid = (0..unsafe { DIMENSIONS })
                .map(|d| self.population.iter().take(MU).map(|xi| xi[d]).sum::<f64>() / MU as f64)
                .collect();

            let delta: Vec<f64> = self
                .centroid
                .iter()
                .zip(&old_centroid)
                .map(|(mt, mt_old)| mt - mt_old)
                .collect();

            let mut p: Vec<f64>;
            if self.generation == 1 {
                p = delta.clone();
            } else {
                let mut old_p = vec![0.0; unsafe { DIMENSIONS }];
                p = (0..unsafe { DIMENSIONS })
                    .map(|d| {
                        (1.0 - CC) * old_p[d]
                            + (CC * (2.0 - CC) * MU as f64).sqrt() * delta[d]
                    })
                    .collect();
            }

            let sqrt_cd_half = (CD / 2.0).sqrt();
            let sqrt_cd = CD.sqrt();
            let sqrt_cd_times_cc = (CC * CD).sqrt();

            for i in 0..unsafe { LAMBDA } {
                let tau_indices = (0..H)
                    .map(|_| self.rng.gen_range(0..unsafe { LAMBDA }))
                    .collect::<Vec<_>>();

                let mut d_i = vec![0.0; unsafe { DIMENSIONS }];
                for d in 0..unsafe { DIMENSIONS } {
                    let diff = sqrt_cd_half
                        * (self.population[tau_indices[0] % unsafe { LAMBDA }][d]
                        - self.population[tau_indices[1] % unsafe { LAMBDA }][d])
                        + sqrt_cd * delta[tau_indices[2] % delta.len()]  // Ensure the index is within bounds
                        * self.uniform.sample(&mut self.rng)
                        + sqrt_cd_times_cc * p[tau_indices[2] % p.len()]  // Ensure the index is within bounds
                        * self.uniform.sample(&mut self.rng)
                        + EPSILON
                        * (1.0 - CC).powi(self.generation as i32 / 2)
                        * self.uniform.sample(&mut self.rng);
                    d_i[d] = self.centroid[d] + diff;
                }

                self.population[i] = d_i;
            }

            fitness_values = self.evaluate(&fitness_fn);
            self.generation += 1;
        }

        // Plot the fitness values
        self.plot_fitness(best_fitness_history, plot_file);
    }

    fn plot_fitness(&self, fitness_values: Vec<f64>, plot_file: &str) {
        // Determine the maximum fitness value for setting the y-axis range
        let max_fitness = *fitness_values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        // Initialize the plot
        let root_area = BitMapBackend::new(plot_file, (1024, 768)).into_drawing_area();
        root_area.fill(&WHITE).unwrap();

        // Create the chart with logarithmic scale on the y-axis
        let mut chart = ChartBuilder::on(&root_area)
            .caption("Fitness over Generations", ("sans-serif", 50).into_font())
            .margin(10)
            .x_label_area_size(50)
            .y_label_area_size(80)
            .build_cartesian_2d(
                0..fitness_values.len(), // X-axis range from 0 to number of fitness values
                (1e-10..1e6).log_scale(), // Y-axis range with logarithmic scale // max_fitness instead of 1e6
            )
            .unwrap();

        // Configure the mesh and axis labels
        chart
            .configure_mesh()
            .y_desc("Fitness (log scale)")
            .x_desc("Generation")
            .y_label_formatter(&|y| format!("{:.0e}", y))
            .draw()
            .unwrap();

        // Prepare data points for plotting
        let data: Vec<(i32, f64)> = fitness_values
            .iter()
            .enumerate()
            .map(|(x, &y)| (x as i32, y))
            .collect();

        // Plot the fitness values as a line series
        chart
            .draw_series(LineSeries::new(
                data.iter().map(|&(x, y)| (x as usize, y)),
                &RED, // Use RED color for the line series
            ))
            .unwrap()
            .label("Fitness")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    &RED.mix(0.5), // Adjust transparency if needed
                )
            });

        // Configure series labels and draw the legend
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap();
    }
}
