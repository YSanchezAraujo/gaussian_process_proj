using Distributions;
using Distances;
using KernelFunctions;

function make_xy(x, y, x_mu, y_mu, sigma, bx_left, bx_right, by_down, by_up, lr_lr, lr_ud, nsteps)
    xy = zeros(nsteps, 2)
    for k in 1:nsteps
        x_sample = rand(Normal(x_mu, sigma))
        x = x + lr_lr * (x_sample - x)
        y_sample = rand(Normal(y_mu, sigma))
        y = y + lr_ud * (y_sample - y)
        if x <= bx_right && x >= bx_left
            xy[k, 1] = x
        else
            xy[k, 1] = xy[k-1, 1]
        end
        if y <= by_up && y >= by_down
            xy[k, 2] = y
        else 
            xy[k, 2] = xy[k-1, 2]
        end
    end
    return xy
end

sqexpkernel(alpha::Real, rho::Real) = alpha^2 * transform(SqExponentialKernel(), 1/(rho*sqrt(2)))

struct GPSIM
    X::Array{Float64, 2}
    y::Array{Int64, 1}
    f::Array{Float64, 1}
    K::Array{Float64, 2}
    rho::Float64
    alpha::Float64
end

function sim_data(nsteps, x_bias, y_bias, noise_level)
    bx_left = -1
    bx_right = 1
    by_up = 1
    by_down = -1
    nsteps = 1000
    lr_lr = 0.2
    lr_ud = 0.2
    # generate some random x-y positions on a 1 by 1 square
    X = make_xy(0.6, -0.8,  x_bias, y_bias, noise_level, 
                bx_left, bx_right, by_down, by_up, lr_lr, lr_ud, nsteps)
    # parameters for the kernel
    alpha =rand(LogNormal(0.0, 0.1))
    rho = rand(LogNormal(1.0, 1.0))
    kernel = sqexpkernel(alpha, rho)
    # generate kernel
    K = kernelmatrix(kernel, XY')
    m = size(K, 1)
    sig2 = 1.
    K = K + I(m) * sig2
    # and sample a function for the firing rate
    f = MvNormal(zeros(m), K)
    f_sampled = rand(f)
    # now sample spikes from f_sampled
    y = rand.(Poisson.(exp.(f_sampled)))
    return GPSIM(X, y, f_sampled, K, rho, alpha)
end