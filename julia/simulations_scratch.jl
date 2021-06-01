base_dir = "/Users/yoelsanchezaraujo/Desktop/gaussian_process_proj/julia/";
include(joinpath(base_dir, "generate_simulated_data.jl"));
include(joinpath(base_dir, "gp_inference.jl"));


bx_left = -1
bx_right = 1
by_up = 1
by_down = -1
nsteps = 1000
lr_lr = 0.2
lr_ud = 0.2
x_mu, y_mu = 0.2, 0.7

# generate some random x-y positions on a 1 by 1 square
X = make_xy(0.6, -0.8,  x_mu, y_mu, bx_left, bx_right, by_down, by_up, lr_lr, lr_ud, nsteps)

# visualize it on that square
plt.scatter(X[:, 1], X[:, 2], alpha=0.45)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)


# now lets create its gram matrix
sqexpkernel(alpha::Real, rho::Real) = alpha^2 * transform(SqExponentialKernel(), 1/(rho*sqrt(2)))

alpha =rand(LogNormal(0.0, 0.1))
rho = rand(LogNormal(1.0, 1.0))
kernel = sqexpkernel(alpha, rho)
K = kernelmatrix(kernel, XY')
m = size(K, 1)
sig2 = 1.
K = K + I(m) * sig2

# and sample a function for the firing rate
f = MvNormal(zeros(m), K)
f_sampled = rand(f)

# now sample spikes from f_sampled
y = rand.(Poisson.(exp.(f_sampled)))

