# Load required packages
library(microbenchmark)
library(Rcpp)
library(ggplot2)
library(cecs)

# Set global parameters
MU <- 50
H <- 5
CC <- 0.1
CD <- 0.1
CE <- 0.1
EPSILON <- 1e-6
DIMENSIONS <- 10 # Must be dimensions from [10, 30, 50, 100] for cec to work
LAMBDA <- 4 + floor(3.0 * log(DIMENSIONS))
MAX_GENERATIONS <- 100 * DIMENSIONS
MU <- floor(LAMBDA / 2)
H <- 6 + floor(3.0 * sqrt(DIMENSIONS))
CC <- 1.0 / sqrt(DIMENSIONS)
CD <- MU / (MU + 2)
CE <- 2.0 / (DIMENSIONS * DIMENSIONS)

# Define fitness functions
fitness_functions <- list(
  sum_of_squares = function(x) sum(x^2),
  cigar = function(x) x[1]^2 + 1e6 * sum(x[-1]^2),
  discus = function(x) 1e6 * x[1]^2 + sum(x[-1]^2),
  ellipsoid = function(x) sum(10^6 * (1:length(x) - 1) / (length(x) - 1) * x^2),
  different_powers = function(x) sum(abs(x)^(2 + 5 * (0:(length(x) - 1)) / (length(x) - 1))),
  sharp_ridge = function(x) abs(x[1] + 100 * sum(x[-1]^2)),
  parabolic_ridge = function(x) abs(x[1] + 100 * sqrt(sum(x[-1]^2))),
  rosenbrock = function(x) sum(100 * (x[-length(x)]^2 - x[-1])^2 + (x[-length(x)] - 1)^2),
  cec2017_1 = function(x) cec2017(1, x),
  cec2017_2 = function(x) cec2017(2, x),
  cec2017_3 = function(x) cec2017(3, x),
  cec2017_4 = function(x) cec2017(4, x),
  cec2017_5 = function(x) cec2017(5, x),
  cec2017_6 = function(x) cec2017(6, x),
  cec2017_7 = function(x) cec2017(7, x),
  cec2017_8 = function(x) cec2017(8, x),
  cec2017_9 = function(x) cec2017(9, x),
  cec2017_10 = function(x) cec2017(10, x),
  cec2017_11 = function(x) cec2017(11, x),
  cec2017_12 = function(x) cec2017(12, x),
  cec2017_13 = function(x) cec2017(13, x),
  cec2017_14 = function(x) cec2017(14, x),
  cec2017_15 = function(x) cec2017(15, x),
  cec2017_16 = function(x) cec2017(16, x),
  cec2017_17 = function(x) cec2017(17, x),
  cec2017_18 = function(x) cec2017(18, x),
  cec2017_19 = function(x) cec2017(19, x),
  cec2017_20 = function(x) cec2017(20, x),
  cec2017_21 = function(x) cec2017(21, x),
  cec2017_22 = function(x) cec2017(22, x),
  cec2017_23 = function(x) cec2017(23, x),
  cec2017_24 = function(x) cec2017(24, x),
  cec2017_25 = function(x) cec2017(25, x),
  cec2017_26 = function(x) cec2017(26, x),
  cec2017_27 = function(x) cec2017(27, x),
  cec2017_28 = function(x) cec2017(28, x),
  cec2017_29 = function(x) cec2017(29, x),
  cec2017_30 = function(x) cec2017(30, x)
)


# Define Differential Evolution Strategy (DES) in R
DES <- R6::R6Class(
  "DES",
  public = list(
    population_history = NULL,
    delta_history = NULL,
    p_history = NULL,
    m = NULL,
    generation = NULL,
    
    initialize = function() {
      self$population_history <- list()
      self$delta_history <- list()
      self$p_history <- list()
      self$m <- rep(0, DIMENSIONS)
      self$generation <- 1
    },
    
    stop_condition = function() {
      self$generation >= MAX_GENERATIONS
    },
    
    initialize_population = function(fitness_fn) {
      initial_population <- replicate(LAMBDA, runif(DIMENSIONS, -5, 5), simplify = FALSE)
      fitness_values <- sapply(initial_population, fitness_fn)
      sorted_indices <- order(fitness_values)
      initial_population <- initial_population[sorted_indices]
      self$population_history <- list(initial_population)
      for (d in 1:DIMENSIONS) {
        self$m[d] <- mean(sapply(initial_population[1:MU], function(ind) ind[d]))
      }
    },
    
    run = function(fitness_fn, plot_file) {
      total_time <- 0
      self$initialize_population(fitness_fn)
      best_fitness_history <- numeric(MAX_GENERATIONS)
      
      while (!self$stop_condition()) {
        start_time <- Sys.time()
        
        best_fitness <- fitness_fn(self$population_history[[length(self$population_history)]][[1]])
        best_fitness_history[self$generation] <- best_fitness
        
        t_idx <- self$generation
        old_m <- self$m
        for (d in 1:DIMENSIONS) {
          self$m[d] <- mean(sapply(self$population_history[[t_idx]][1:MU], function(ind) ind[d]))
        }
        
        delta <- self$m - old_m
        self$delta_history[[self$generation]] <- delta
        
        if (self$generation == 1) {
          p <- delta
        } else {
          p <- (1 - CC) * self$p_history[[length(self$p_history)]] + sqrt(CC * (2 - CC) * MU) * delta
        }
        self$p_history[[self$generation]] <- p
        
        sqrt_cd_half <- sqrt(CD / 2)
        sqrt_cd <- sqrt(CD)
        sqrt_one_minus_cd <- sqrt(1 - CD)
        
        new_population <- list()
        for (i in 1:LAMBDA) {
          normal <- rnorm(DIMENSIONS)
          normal_i <- rnorm(DIMENSIONS, sd = sqrt(i))
          tau_1 <- sample(1:H, 1)
          tau_2 <- sample(1:H, 1)
          tau_3 <- sample(1:H, 1)
          j <- sample(1:MU, 1)
          k <- sample(1:MU, 1)
          
          history_idx_1 <- max(1, t_idx - tau_1)
          history_idx_2 <- max(1, t_idx - tau_2)
          history_idx_3 <- max(1, t_idx - tau_3)
          
          d_i <- numeric(DIMENSIONS)
          for (d in 1:DIMENSIONS) {
            diff <- sqrt_cd_half * (self$population_history[[history_idx_1]][[j]][d] - self$population_history[[history_idx_1]][[k]][d]) +
              sqrt_cd * self$delta_history[[history_idx_2]][d] * normal[d] +
              sqrt_one_minus_cd * self$p_history[[history_idx_3]][d] * normal[d] +
              EPSILON * (1 - CE)^(self$generation / 2) * normal_i[d]
            d_i[d] <- self$m[d] + diff
          }
          new_population[[i]] <- d_i
        }
        
        pre_fitness_time <- Sys.time()
        fitness_values <- sapply(new_population, fitness_fn)
        post_fitness_time <- Sys.time()
        
        sorted_indices <- order(fitness_values)
        new_population <- new_population[sorted_indices]
        self$population_history[[self$generation + 1]] <- new_population
        self$generation <- self$generation + 1
        end_time <- Sys.time()
        total_time <- total_time + (as.numeric(pre_fitness_time - start_time) + as.numeric(end_time - post_fitness_time))
      }
      
      self$plot_fitness(best_fitness_history, plot_file)
      
      cat(sprintf("Function %s\n", plot_file))
      cat(sprintf("Best fitness = %f\n", best_fitness_history[self$generation - 1]))
      best_member <- self$population_history[[self$generation]][[1]]
      cat(paste(sprintf("%.4f", best_member), collapse = " "), "\n")
      
      cat(sprintf("Total time excluding fitness evaluations: %f seconds\n\n", total_time))
    },
    
    plot_fitness = function(fitness_values, plot_file) {
      # Filter out non-positive values to avoid issues with log scale
      fitness_df <- data.frame(Generation = 1:length(fitness_values), Fitness = fitness_values)
      fitness_df <- fitness_df[fitness_df$Fitness > 0, ]
      
      # Create the plot
      p <- ggplot(fitness_df, aes(x = Generation, y = Fitness)) +
        geom_line(color = "red") +
        scale_y_log10() +
        labs(title = "Fitness over Generations", x = "Generation", y = "Fitness (log scale)") +
        theme_minimal()
      
      # Save the plot
      if (nrow(fitness_df) > 0) { # Check to ensure there's data to plot
        ggsave(plot_file, plot = p, width = 7, height = 7)
      } else {
        message("No positive fitness values to plot.")
      }
    }
  )
)

# Run the DES for each fitness function
for (name in names(fitness_functions)) {
  fitness_fn <- fitness_functions[[name]]
  plot_file <- paste0(name, ".png")
  des <- DES$new()
  des$run(fitness_fn, plot_file)
}
