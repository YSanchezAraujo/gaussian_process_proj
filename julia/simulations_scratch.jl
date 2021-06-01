base_dir = "/Users/yoelsanchezaraujo/Desktop/gaussian_process_proj/julia/";
include(joinpath(base_dir, "generate_simulated_data.jl"));
include(joinpath(base_dir, "gp_inference.jl"));

nsteps, x_bias, y_bias, noise_level = 1000, 0.2, -0.7, 1.
sim_info = sim_data(nsteps, x_bias, y_bias, noise_level)
