extern crate rand;
extern crate plotters;

use plotters::prelude::*;
use plotters::style::{Color, RED};
use rand::prelude::*;
use rand_distr::{Normal, Distribution as OtherDistribution};
use std::f64;

static mut MU: usize = 50; // Number of the best solutions to keep
static mut H: usize = 5; // History size
static mut CC: f64 = 0.1; // Learning rate for covariance matrix
static mut CD: f64 = 0.1; // Learning rate for differential weights
static mut CE: f64 = 0.1;
static mut EPSILON: f64 = 1e-6; // Small constant
static mut DIMENSIONS: usize = 3; // Dimensionality of the problem
static mut LAMBDA: usize = 7; // Population size (will be updated)
static mut MAX_GENERATIONS: usize = 30000; // Max number of generations (will be updated)

fn main() {
    // Parameter initialization as in the Appendix
    unsafe {
        DIMENSIONS = 30; // Set your desired dimensions
        LAMBDA = 4 + (3.0 * (DIMENSIONS as f64).ln()).floor() as usize;
        MAX_GENERATIONS = 10000 * DIMENSIONS;
        MU = (LAMBDA as f64 / 2.0).floor() as usize;
        H = 6 + (3.0 * (DIMENSIONS as f64).sqrt()).floor() as usize;
        CC = 1.0 / (DIMENSIONS as f64).sqrt();
        CD = MU as f64 / ((MU + 2) as f64);
        CE = 2.0 / (DIMENSIONS * DIMENSIONS) as f64;

        println!("LAMBDA (Population size): {}", LAMBDA);
        println!("MAX_GENERATIONS: {}", MAX_GENERATIONS);
        println!("MU (No best solutions kept): {}", MU);
        println!("H (History size): {}", H);
        println!("CC (LR for covariance matrix): {}", CC);
        println!("CD (LR for differential weights): {}", CD);
        println!("CE (idk xD): {}", CE);

        // Fitness function 1
        let mut des1 = DES::new();
        des1.run(
            |x| x.iter().map(|&xi| xi * xi).sum(),
            "sum_of_squares.png",
        );
        // Fitness function 2
        let mut des2 = DES::new();
        des2.run(
            |x| x[0] * x[0] + 1e6 * x.iter().skip(1).map(|&xi| xi * xi).sum::<f64>(),
            "cigar.png",
        );
        // Fitness function 3
        let mut des3 = DES::new();
        des3.run(
            |x| 1e6 * x[0] * x[0] + x.iter().skip(1).map(|&xi| xi * xi).sum::<f64>(),
            "discus.png",
        );
        // Fitness function 4
        let mut des4 = DES::new();
        des4.run(
            |x| x.iter().enumerate().map(|(i, &xi)| 10_f64.powf(6.0 * (i as f64) / (x.len() as f64 - 1.0)) * xi * xi).sum::<f64>(),
            "ellipsoid.png",
        );
        // Fitness function 5
        let mut des5 = DES::new();
        des5.run(
            |x| x.iter().enumerate().map(|(i, &xi)| xi.abs().powf(2.0 * (1.0 + 5.0 * (i as f64) / (x.len() as f64 - 1.0)))).sum::<f64>(),
            "different_powers.png",
        );
        // Ridge functions:
        // Fitness function 6
        let mut des6 = DES::new();
        des6.run(
            |x| (x[0] + 100.0 * x[1..].iter().map(|&xi| xi * xi).sum::<f64>()).abs(),
            "sharp_ridge.png",
        );
        // Fitness function 7
        let mut des7 = DES::new();
        des7.run(
            |x| (x[0] + 100.0 * x[1..].iter().map(|&xi| xi * xi).sum::<f64>().sqrt()).abs(),
            "parabolic_ridge.png",
        );
        // Fitness function 8
        let mut des8 = DES::new();
        des8.run(
            |x| (0..x.len() - 1).map(|i| 100.0 * (x[i] * x[i] - x[i + 1] * x[i + 1]) + (x[i] - 1.0).powi(2)).sum::<f64>().abs(),
            "Rosenbrock.png",
        );
    }
}

struct DES {
    population_history: Vec<Vec<Vec<f64>>>, // dim: t x LAMBDA x DIMENSIONS // Initial population has values from [-5.0, 5.0]
    delta_history: Vec<Vec<f64>>, // dim: t x DIMENSIONS // one value per LAMBDA population
    p_history: Vec<Vec<f64>>, // dim: t x DIMENSIONS // one value per LAMBDA population
    m: Vec<f64>, // Just the previous m // one value per LAMBDA population
    rng: ThreadRng,
    generation: usize,
}

impl DES {
    pub fn new() -> Self {
        let rng = thread_rng();
        let generation = 1;

        DES {
            population_history: Vec::new(),
            delta_history: Vec::new(),
            p_history: Vec::new(),
            m: vec![0.0; unsafe { DIMENSIONS }],
            rng,
            generation,
        }
    }

    fn stop_condition(&self) -> bool {
        self.generation >= unsafe { MAX_GENERATIONS }
    }

    unsafe fn initialize<F>(&mut self, fitness_fn: F)
    where
        F: Fn(&Vec<f64>) -> f64,
    {
        let mut rng = thread_rng();
        let mut initial_population: Vec<Vec<f64>> = (0..LAMBDA)
            .map(|_| (0..DIMENSIONS).map(|_| rng.gen_range(-5.0..5.0)).collect())
            .collect();

        // Evaluate initial population and sort by fitness
        let fitness_values: Vec<f64> = initial_population.iter().map(|ind| fitness_fn(ind)).collect();
        let mut sorted_indices: Vec<usize> = (0..LAMBDA).collect();
        sorted_indices.sort_by(|&a, &b| fitness_values[a].partial_cmp(&fitness_values[b]).unwrap());
        initial_population = sorted_indices.into_iter().map(|i| initial_population[i].clone()).collect();

        self.population_history.push(initial_population);

        // Calculate initial m
        for d in 0..DIMENSIONS {
            self.m[d] = (0..LAMBDA)
                .map(|i| self.population_history[0][i][d])
                .sum::<f64>() / MU as f64;
        }
    }

    unsafe fn run<F>(&mut self, fitness_fn: F, plot_file: &str)
    where
        F: Fn(&Vec<f64>) -> f64,
    {
        // Initialize: stage 1 - stage 4 in the article
        self.initialize(&fitness_fn);

        let mut best_fitness_history = vec![];

        while !self.stop_condition() {
            let best_fitness = fitness_fn(self.population_history.last().unwrap().first().unwrap());
            best_fitness_history.push(best_fitness);
            // println!("Generation {}: Best fitness = {}", self.generation, *best_fitness);

            let t_idx = self.generation - 1; // Index from t
            let old_m = self.m.clone();

            // TODO: take min(MU, LAMBDA) in case MU is bigger than population size
            for d in 0..DIMENSIONS {
                self.m[d] = (0..MU)
                    .map(|i| self.population_history[t_idx % MAX_GENERATIONS][i][d])
                    .sum::<f64>() / MU as f64;
            }

            let delta: Vec<f64> = self.m.iter().zip(&old_m).map(|(m_t, m_t_old)| m_t - m_t_old).collect();
            self.delta_history.push(delta.clone());

            let mut p = vec![0.0; DIMENSIONS];
            if self.generation == 1 {
                p = delta.clone();
            } else {
                for d in 0..DIMENSIONS {
                    p[d] = (1.0 - CC) * self.p_history.last().unwrap()[d] + (CC * (2.0 - CC) * MU as f64).sqrt() * delta[d];
                }
            }
            self.p_history.push(p.clone());

            // Loop: stage 13 - stage 21 in the article
            let sqrt_cd_half = (CD / 2.0).sqrt();
            let sqrt_cd = CD.sqrt();
            let sqrt_one_minus_cd = (1.0 - CD).sqrt();

            let mut new_population = vec![vec![0.0; DIMENSIONS]; LAMBDA];
            for i in 0..LAMBDA {
                let normal = Normal::new(0.0, 1.0).unwrap();
                let normal_i = Normal::new(0.0, (i as f64).sqrt()).unwrap(); // For N(0,i)

                // Because the last pushed X atp is from generation t-1 we start from 0 here
                let tau_1 = self.rng.gen_range(0..H);
                let tau_2 = self.rng.gen_range(0..H);
                let tau_3 = self.rng.gen_range(0..H);
                let j = self.rng.gen_range(0..MU);
                let k = self.rng.gen_range(0..MU);

                // Random history index or 0, whichever is greater
                let history_idx_1 = t_idx.saturating_sub(tau_1);
                let history_idx_2 = t_idx.saturating_sub(tau_2);
                let history_idx_3 = t_idx.saturating_sub(tau_3);

                let mut d_i = vec![0.0; DIMENSIONS];
                for d in 0..DIMENSIONS {
                    let diff = sqrt_cd_half
                        * (self.population_history[history_idx_1][j][d]
                        - self.population_history[history_idx_1][k][d])
                        + sqrt_cd * self.delta_history[history_idx_2][d] * normal.sample(&mut thread_rng())
                        + sqrt_one_minus_cd * self.p_history[history_idx_3][d] * normal.sample(&mut thread_rng())
                        + EPSILON * (1.0 - CE).powi(self.generation as i32 / 2) * normal_i.sample(&mut thread_rng());
                    d_i[d] = self.m[d] + diff;
                }

                new_population[i] = d_i;
            }

            // Evaluate new population and sort by fitness
            let fitness_values: Vec<f64> = new_population.iter().map(|ind| fitness_fn(ind)).collect();
            let mut sorted_indices: Vec<usize> = (0..LAMBDA).collect();
            sorted_indices.sort_by(|&a, &b| fitness_values[a].partial_cmp(&fitness_values[b]).unwrap());
            new_population = sorted_indices.into_iter().map(|i| new_population[i].clone()).collect();

            self.population_history.push(new_population);
            self.generation += 1;
        }

        // Plot the fitness values
        self.plot_fitness(best_fitness_history.clone(), plot_file);


        // Print the last population (the best member) and fitness value
        println!("Function {}", &plot_file[..plot_file.rfind('.').unwrap_or(plot_file.len())]);
        println!("Best fitness = {}", best_fitness_history.last().unwrap());
        let last_x: &Vec<f64> = self.population_history.last().unwrap().first().unwrap();
        for value in last_x.iter() {
            println!("{:.4}", value); // Printing each value with 4 decimal places for clarity
        }
        println!();
    }

    fn plot_fitness(&self, fitness_values: Vec<f64>, plot_file: &str) {
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
