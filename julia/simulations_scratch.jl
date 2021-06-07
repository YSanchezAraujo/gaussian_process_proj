base_dir = "/Users/yoelsanchezaraujo/Desktop/gaussian_process_proj/julia/";
include(joinpath(base_dir, "generate_simulated_data.jl"));
include(joinpath(base_dir, "gp_inference.jl"));

nsteps, x_bias, y_bias, noise_level = 2001, 0.2, -0.7, 1.
sim_info = sim_data(nsteps, x_bias, y_bias, noise_level)

# split into training and testing data
n_train = Int64(floor(0.75*nsteps))
X_train, y_train = sim_info.X[1:n_train, :], sim_info.y[1:n_train]
X_test, y_test = sim_info.X[n_train+1:end, :], sim_info.y[n_train+1:end]


# just want to try and visualize the firing patterns in the field
x = -1:0.001:1
y = -1:0.001:1
meshgrid(x, y) = (x' .* ones(length(x)), ones(length(y))' .* y)
xx, yy = meshgrid(x, y)


# this looks like it works, I've testing and it recovers the parameters roughly
opts = estimate_gp_params(rand(2), X_train, Float64.(y_train), NelderMead)

using Distances;
using KernelFunctions;
using Interpolations;

# rho is length scale, alpha deals with expected variation
kernel = sqexpkernel(sim_info.alpha, sim_info.rho)
K_XX = kernelmatrix(kernel, X_train')
f, L, W, Wsqr, lml = estimate_laplace_gp(K_XX, Float64.(y_train))

add_dim(x::Array) = reshape(x, (size(x)...,1))

preds = zeros(size(X_test, 1))

# there's likely a much faster way to do this
# don't do this yet as we aren't doing the poisson integral atm
# for k in 1:size(X_test, 1)
#     k_star = kernelmatrix(kernel, X_train', add_dim(X_test[k, :]))
#     K_new = kernelmatrix(kernel, add_dim(X_test[k, :]), add_dim(X_test[k, :]))
#     preds[k] = predict_laplace_gp(f, Float64.(y_train), K_XX, K_new, k_star)[1, 1]
# end

preds2 = zeros(size(X_test, 1))
for k in 1:size(X_test, 1)
    k_star = kernelmatrix(kernel, X_train', add_dim(X_test[k, :]))
    preds2[k] = simple_predict(f, Float64.(y_train), k_star)[1, 1]
end

# ok so now how do I visualize the firing rate map
scatter(y_test, exp.(preds2), xlabel="true spikes", ylabel="predicted", 
         legend=nothing, dpi=300)

histogram(y_test, alpha=0.4, label="true spikes", dpi=300)
histogram!(exp.(preds2), alpha=0.4, label="predicted", dpi=300)