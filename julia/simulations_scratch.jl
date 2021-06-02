base_dir = "/Users/yoelsanchezaraujo/Desktop/gaussian_process_proj/julia/";
include(joinpath(base_dir, "generate_simulated_data.jl"));
include(joinpath(base_dir, "gp_inference.jl"));

nsteps, x_bias, y_bias, noise_level = 1000, 0.2, -0.7, 1.
sim_info = sim_data(nsteps, x_bias, y_bias, noise_level)

# split into training and testing data
n_train = Int64(floor(0.75*nsteps))
X_train, y_train = sim_info.X[1:n_train, :], sim_info.y[1:n_train]
X_test, y_test = sim_info.X[n_train+1:end, :], sim_info.y[n_train+1:end]

using Distances;
using KernelFunctions;

alpha, rho = sim_info.alpha, sim_info.rho
sqexpkernel(alpha::Real, rho::Real) = alpha^2 * transform(SqExponentialKernel(), 1/(rho*sqrt(2)))

kernel = sqexpkernel(alpha, rho)
K_XX = kernelmatrix(kernel, X_train')
K_XXS = kernelmatrix(kernel, X_train', X_test')
K_XSXS = kernelmatrix(kernel, X_test')

f, lml = estimate_laplace_gp(K_XX, Float64.(y_train))



