using LinearAlgebra, MKL
using Convex, SCS
using Combinatorics
using QuantumInformation
using SparseArrays

function projector_s_as(d=1, N=1, atol=1e-3)
    """
        Usage: Take matrix X. Define |Y>> := projector_s_as(d,N) |X>>. Then, [Y,U^{⊗N} ⊗ U*] = 0 for all U.
    """
    
    P = zeros(d^(2N+2), d^(2N+2))
    idx = CartesianIndices(Tuple(fill(d, N+1)))

    for perm in permutations(1:N+1)
        x = zeros(fill(d, 2N+2)...)
        for k1 in idx
            k0 = CartesianIndex(Tuple(k1)[perm])
            x[k0, k1] = 1
        end
        x = vec(x)
        P += x * x'
    end

    F = eigen(P)
    Λ = Diagonal([x > atol ? 1 : 0 for x in F.values])
    P0 = F.vectors * Λ * F.vectors'
    asym_perm = Array(1:(2N+2))
    asym_perm[N+1], asym_perm[2N+2] = 2N + 2, N + 1
    sparse(permutesystems(P0, fill(d, 2N + 2), asym_perm))
end

function get_network_properties(Rs, d, N)
    S = [sum(Rs[i, j] for j=1:d) for i ∈ CartesianIndices(Tuple(fill(1:d, N)))]
    S = reshape(S, fill(d, N)...)
    constraints = [S[i] == partialtrace(S[i], N+1, fill(d, N+1)) ⊗ I(d)/d for i=1:d^N]
    S = [partialtrace(x, N+1, fill(d, N+1))/d for x ∈ S]
    S = reshape(S, fill(d, N)...)
    idx = CartesianIndices(Tuple(fill(1:d, N-1)))
    constraints += vec([S[1, i] == S[j, i] for j=2:d, i ∈ idx])
    S = reshape([S[1, i] for i ∈ idx], fill(d, N-1)...)

    for n=N-1:-1:1
        S = [partialtrace(x, 1, fill(d, n+1)) for x ∈ S]
        S = reshape(S, fill(d, n)...)
        idx = CartesianIndices(Tuple(fill(1:d, n-1)))
        constraints += vec([S[1, i] == S[j, i] for j=2:d, i ∈ idx])
        S = [S[1, i] for i ∈ idx]
        S = reshape(S, fill(d, n-1)...)
    end

    constraints += real(tr(S[1])) == 1
    constraints
end

function get_network_properties_parallel(Rs, d, N)
    S = [sum(Rs[i, j] for j=1:d) for i ∈ CartesianIndices(Tuple(fill(1:d, N)))]
    S = reshape(S, fill(d, N)...)
    constraints = [S[i] == partialtrace(S[i], N+1, fill(d, N+1)) ⊗ I(d)/d for i=1:d^N]
    S = [partialtrace(x, N+1, fill(d, N+1))/d for x ∈ S]
    constraints += [S[1] == S[j] for j=2:d^N]
    constraints += real(tr(S[1])) == 1
    constraints
end

function calc(d, N, P::SparseMatrixCSC{Float64, Int64})
    Rs = [ComplexVariable(d^(N+1), d^(N+1)) for _=1:d^(N+1)]
    Rs = reshape(Rs, fill(d, N+1)...)

    constraints = [R in :SDP for R in Rs]
    constraints += [P * vec(R) == vec(R) for R in Rs]
    constraints += get_network_properties(Rs, d, N)

    linear = LinearIndices(Rs)
    t = 1/d * real(
        sum(
            Rs[CartesianIndex(reverse(Tuple(a)))][linear[a], linear[a]] for a in CartesianIndices(Rs)
        )
        )
    problem = maximize(t, constraints)
    solve!(problem, Convex.MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-8))

    return (string(problem.status), problem.optval)
end

function calc_p(d, N, P::SparseMatrixCSC{Float64, Int64})
    Rs = [ComplexVariable(d^(N+1), d^(N+1)) for _=1:d^(N+1)]
    Rs = reshape(Rs, fill(d, N+1)...)

    constraints = [R in :SDP for R in Rs]
    constraints += [P * vec(R) == vec(R) for R in Rs]
    constraints += get_network_properties_parallel(Rs, d, N)

    linear = LinearIndices(Rs)
    t = 1/d * real(
        sum(
            Rs[CartesianIndex(reverse(Tuple(a)))][linear[a], linear[a]] for a in CartesianIndices(Rs)
        )
        )
    problem = maximize(t, constraints)
    solve!(problem, Convex.MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-8))

    return (string(problem.status), problem.optval)
end

function results()
    open("results.txt", "w") do file end
    for N = 1:10
        P = projector_s_as(2, N, 1e-8)
        open("results.txt", "a") do file
            ans = calc_p(2, N, P)
            if ans[1] == "OPTIMAL"
                write(file, "P, N $(N), $(ans[2]) \n")
            else
                exit(0)
            end
        end
        open("results.txt", "a") do file
            ans = calc(2, N, P)
            if ans[1] == "OPTIMAL"
                write(file, "A, N $(N), $(ans[2]) \n")
            else
                exit(0)
            end
        end
    end
end

results()
