using Distributions;
using LinearAlgebra;
using Distances;
using KernelFunctions;
using Optim;

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
    np_f = size(f, 1)
    f = length(np_f) > 1 ? drop_singleton(f) : f
    return diagm(-exp.(f))
end

function estimate_laplace_gp(K, y, tol=1e-3)
    n, c = length(y), 1
    f, I_n = zeros(n), I(n)
    W = -d2df_log_poisson_pmf(f)
    Wsqr = sqrt(W)
    L = Matrix(cholesky(Symmetric(I_n + Wsqr * K * Wsqr)).L)
    b = W * f + ddf_log_poisson_pmf(f, y)
    a = b - Wsqr * (L' \ (L \ (Wsqr * K * b)))
    obj_cmp = -0.5 * a'f + sum(log_poisson_pmf(f, y))
    f = K * a
    obj_new = -0.5 * a'f + sum(log_poisson_pmf(f, y))
    log_yxt = obj_new - sum(log.(diag(L)))
    while abs(obj_new - obj_cmp) > tol
        println(c, "\t", abs(obj_new - obj_cmp))
        W = -d2df_log_poisson_pmf(f)
        Wsqr = sqrt(W)
        L = Matrix(cholesky(Symmetric(I_n + Wsqr * K  * Wsqr)).L)
        b = W * f + ddf_log_poisson_pmf(f, y)
        a = b - Wsqr * (L' \ (L \ (Wsqr * K * b)))
        obj_cmp = -0.5 * a'f + sum(log_poisson_pmf(f, y))
        f = K * a
        obj_new = -0.5 * a'f + sum(log_poisson_pmf(f, y))
        log_yxt = obj_new - sum(log.(diag(L)))
        c += 1
    end
    return f, log_yxt
end

sqexpkernel(alpha::Real, rho::Real) = alpha^2 * transform(SqExponentialKernel(), 1/(rho*sqrt(2)))

function estimate_laplace_gp_opt(kernel_params, X, y, tol=1e-3)
    K = kernelmatrix(sqexpkernel(kernel_params...), X')
    n, c = length(y), 1
    f, I_n = zeros(n), I(n)
    W = -d2df_log_poisson_pmf(f)
    Wsqr = sqrt(W)
    L = Matrix(cholesky(Symmetric(I_n + Wsqr * K * Wsqr)).L)
    b = W * f + ddf_log_poisson_pmf(f, y)
    a = b - Wsqr * (L' \ (L \ (Wsqr * K * b)))
    obj_cmp = -0.5 * a'f + sum(log_poisson_pmf(f, y))
    f = K * a
    obj_new = -0.5 * a'f + sum(log_poisson_pmf(f, y))
    log_yxt = obj_new - sum(log.(diag(L)))
    while abs(obj_new - obj_cmp) > tol
        W = -d2df_log_poisson_pmf(f)
        Wsqr = sqrt(W)
        L = Matrix(cholesky(Symmetric(I_n + Wsqr * K  * Wsqr)).L)
        b = W * f + ddf_log_poisson_pmf(f, y)
        a = b - Wsqr * (L' \ (L \ (Wsqr * K * b)))
        obj_cmp = -0.5 * a'f + sum(log_poisson_pmf(f, y))
        f = K * a
        obj_new = -0.5 * a'f + sum(log_poisson_pmf(f, y))
        log_yxt = obj_new - sum(log.(diag(L)))
    end
    return -log_yxt
end

estimate_gp_params(params, X, y)  = optimize(w -> estimate_laplace_gp_opt(w, X, y), params, BFGS())


# these two below work on a per test-input basis
# need to double check this
function approx_laplace_gp_integral(spikes, var_f_star, f_star)
    unique_spikes = sort(unique(spikes))
    pred_prob = (pdf.(Poisson(exp(f_star)), unique_spikes) .* 
                 pdf.(Normal(f_star, var_f_star), unique_spikes))
    return pred_prob ./ sum(pred_prob)
end

function predict_laplace_gp(f_hat, y, K, K_new, k_star)
    n = length(y)
    I_n = I(n)
    W = -d2df_log_poisson_pmf(f_hat)
    Wsqr = sqrt(W)
    L = Matrix(cholesky(Symmetric(I_n + Wsqr * K * Wsqr)).L)
    f_new = k_star'ddf_log_poisson_pmf(f_hat, y)
    v = L \ (Wsqr * k_star)
    var_f_new = K_new - v'v
    return approx_laplace_gp_integral(y, var_f_new, f_new)
end


