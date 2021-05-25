include("/Users/yoelsanchezaraujo/Desktop/preproc_data.jl");

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

# ok we have the kernel and we have the data, now to try and implement the general algorithm
# q.1: what is the likelihood function here? 

function drop_singleton(a)
    dropdims(a, dims = (findall(size(a) .== 1)...,))
end

# log[p(y|f)]
function log_poisson_pmf(f::Array{Float64, 1}, y::Array{Float64, 1})
    np_f, np_y = size(f), size(y)
    f = length(np_f) > 1 ? drop_singleton(f) : f
    y = length(np_y) > 1 ? drop_singleton(y) : y
    return y .* f .- exp.(f)
end

# first derivative of log[p(y|f)]
function ddf_log_poisson_pmf(f::Array{Float64, 1}, y::Array{Float64, 1})
    np_f, np_y = size(f), size(y)
    f = length(np_f) > 1 ? drop_singleton(f) : f
    y = length(np_y) > 1 ? drop_singleton(y) : y
    return y .- exp.(f)
end

# second derivative of log[p(y|f)]
function d2df_log_poisson_pmf(f::Array{Float64, 1})
    np_f, np_y = size(f)
    f = length(np_f) > 1 ? drop_singleton(f) : f
    return diagm(-exp.(f))
end

function estimate_laplace_gp(K, y, tol=1e-3)
    n, c = length(y), 1
    f, I_n = zeros(n), I(n)
    W = -d2df_log_poisson_pmf(f)
    Wsqr = sqrt(W)
    L = cholesky(Symmetric(I_n + Wsqr * K * Wsqr)).L
    La = Matrix(L)
    b = W * f + ddf_log_poisson_pmf(f, y)
    a = b - Wsqr * (La' \ (L \ (Wsqr * K * b)))
    obj_cmp = -0.5 * a'f + sum(log_poisson_pmf(f, y))
    f = K * a
    obj_new = -0.5 * a'f + sum(log_poisson_pmf(f, y))
    log_yxt = -0.5 * a'f + sum(log_poisson_pmf(f, y)) - sum(log.(diag(La)))
    while abs(obj_new - obj_cmp) > tol
        println(c, "\t", abs(obj_new - obj_cmp))
        W = -d2df_log_poisson_pmf(f)
        Wsqr = sqrt(W)
        L = cholesky(Symmetric(I_n + Wsqr * K  * Wsqr)).L
        La = Matrix(L)
        b = W * f + ddf_log_poisson_pmf(f, y)
        a = b - Wsqr * (La' \ (L \ (Wsqr * K * b)))
        obj_cmp = -0.5 * a'f + sum(log_poisson_pmf(f, y))
        f = K * a
        obj_new = -0.5 * a'f + sum(log_poisson_pmf(f, y))
        log_yxt = obj_new - sum(log.(diag(La)))
        c += 1
    end
    return f, log_yxt
end


