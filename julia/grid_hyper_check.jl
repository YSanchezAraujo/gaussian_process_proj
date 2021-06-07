base_dir = "/Users/yoelsanchezaraujo/Desktop/gaussian_process_proj/julia/";
include(joinpath(base_dir, "generate_simulated_data.jl"));
include(joinpath(base_dir, "gp_inference.jl"));

nsteps, x_bias, y_bias, noise_level = 2001, 0.2, -0.7, 1.
sim_info = sim_data(nsteps, x_bias, y_bias, noise_level)

# split into training and testing data
n_train = Int64(floor(0.75*nsteps))
X_train, y_train = sim_info.X[1:n_train, :], sim_info.y[1:n_train]
X_test, y_test = sim_info.X[n_train+1:end, :], sim_info.y[n_train+1:end]

alpha_list = collect(0.001:0.1:2)
rho_list = collect(0.001:0.1:1)

N, P = length(alpha_list), length(rho_list)
lml_vals = zeros(N, P)
f_vals = zeros(length(sim_info.y), N, P)

for n in 1:N, p in 1:P
    kernel = sqexpkernel(alpha_list[n], rho_list[p])
    K = kernelmatrix(kernel, sim_info.X')
    f, L, W, Wsqr, lml = estimate_laplace_gp(K, Float64.(sim_info.y))
    lml_vals[n, p] = lml
    f_vals[:, n, p] = f
end

best_lml = lml_vals[argmax(lml_vals)]
best_alpha = alpha_list[argmax(lml_vals)[1]]
best_rho = rho_list[argmax(lml_vals)[2]]

# need to figure out how to change the xticks and labels
using StatsPlots;

tstring = string("amplitude = ", best_alpha, ", length scale = ", best_rho)

hmap = heatmap(alpha_list, rho_list, lml_vals, 
               xlabel="amplitude", 
               ylabel="length scale", 
               title=tstring, dpi=300)
savefig(hmap, "/Users/yoelsanchezaraujo/Desktop/alpha_rho_map.png")

sf = scatter(sim_info.f, f_vals[:, argmax(lml_vals)], xlabel="true f", 
              ylabel="estimated f at argmax(lml)", legend=nothing, dpi=300)
savefig(sf, "/Users/yoelsanchezaraujo/Desktop/alpha_rho_sf_map.png")
