using LinearAlgebra
using Random
include("../src/graphutils.jl")

function one_hot(digit::Int64)
    one_hot_vector = zeros(Int, 10) 
    one_hot_vector[digit + 1] = 1
    return one_hot_vector
end

nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end]) 

function xavier_init(rng::AbstractRNG, dims::Integer...; gain::Real=1)
    scale = Float32(gain) * sqrt(24.0f0 / sum(nfan(dims...)))
    (rand(rng, Float32, dims...) .- 0.5f0) .* scale
end


function mse_grad(y_true, y_pred)
    return 2*(y_pred - y_true) / prod(size(y_pred)[1:end-1])
end

function cross_entropy_loss(y_pred, y_true)
    
    epsilon = 1e-12
    y_pred = clamp.(y_pred, epsilon, 1 - epsilon)
    return -sum(y_true .* log.(y_pred))
end

function cross_grad(y_true, y_pred) 
    return (y_pred - y_true) 
end

function relu(matrix::AbstractVector{T}, c = 5.0) where T<:Real
    m = matrix 
    return max.(m, 0.0)
end

function relu_derivative(matrix::AbstractVector{T}, c = 5.0) where T<:Real
    m = matrix 
    map(x -> x > 0 ? 1 : 0, m)
end

function tanhip(matrix::AbstractVector{T}, c=5.0) where T<:Real
    m = matrix 
    return tanh.(m)
end

function tanhip_derivative(matrix::AbstractVector{T}, c=5.0) where T<:Real
    m = matrix 
    return Vector{Float32}(map(x -> 1 - tanh(x)^2, m))
end

function softmax(matrix::AbstractMatrix{T}) where T<:Real
    v_max = maximum(matrix)
    exp_matrix = exp.(matrix .- v_max)
    sum_exp_matrix = sum(exp_matrix)
    return exp_matrix ./ sum_exp_matrix
end

function softmax_derivative(matrix::AbstractVector{T}) where T<:Real 
    n = length(matrix)
    
    jacobian_element(i, j) = i == j ? matrix[i] * (1 - matrix[i]) : -matrix[i] * matrix[j]
    return [jacobian_element(i, j) for i in 1:n, j in 1:n]

end

