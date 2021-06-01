base_dir = "/Users/yoelsanchezaraujo/Desktop/gaussian_process_proj/julia/";
include(joinpath(base_dir, "preproc_data.jl"));
include(joinpath(base_dir, "gp_inference.jl"));

using Distributions
using Distances
using KernelFunctions


files_path = "/Users/yoelsanchezaraujo/Desktop/grid_data_pillow"
spike_path = joinpath(files_path, "Dorian_111013o1+o2+o3_T1C1.mat")
loc_path = joinpath(files_path, "Dorian_111013o1+o2+o3_pos.mat")
data = preprocess(spike_path, loc_path, 1.)
#spkinfo, spks = compute_spike_info(test_file_spk, "cellTS", 1.)

X, y = data.X, data.y

sqexpkernel(alpha::Real, rho::Real) = alpha^2 * transform(SqExponentialKernel(), 1/(rho*sqrt(2)))

alpha =rand(LogNormal(0.0, 0.1))
rho = rand(LogNormal(1.0, 1.0))
kernel = sqexpkernel(alpha, rho)
K = kernelmatrix(kernel, X')
m = size(K, 1)
sig2 = 1.
K = K + I(m) * sig2

# to quickly visualize it
plt.ion()
plt.imshow(K, aspect="auto")