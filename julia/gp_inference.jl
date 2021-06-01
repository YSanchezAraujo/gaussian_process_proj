using Distributions;
using LinearAlgebra;

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
    L = cholesky(Symmetric(I_n + Wsqr * K * Wsqr)).L
    La = Matrix(L)
    b = W * f + ddf_log_poisson_pmf(f, y)
    a = b - Wsqr * (La' \ (L \ (Wsqr * K * b)))
    obj_cmp = -0.5 * a'f + sum(log_poisson_pmf(f, y))
    f = K * a
    obj_new = -0.5 * a'f + sum(log_poisson_pmf(f, y))
    log_yxt = obj_new - sum(log.(diag(La)))
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
    return f, log_yxt, La, W, a
end


