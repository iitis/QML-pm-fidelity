using Convex, SCS
using LinearAlgebra
using Combinatorics
using QuantumInformation
using SparseArrays


function get_parallel_network_properties(L, N)
    sL = L[1] + L[2]
    constraints = [2^(N+1) * sL == partialtrace(sL, 2, [2^N, 2^(N+1)]) ⊗ I(2^(N+1))]
    constraints += real(tr(sL)) == 2^(N+1)
    constraints
end

function c_projector(N, atol=1e-3)
    """
        Take matrix X. Define |Y>> := c_projector(N) |X>>. 
        Then, [Y, \1_A ⊗ U^{⊗N} ⊗ U* ] = 0 for all U.
    """
    P = zeros(2^(2N+2), 2^(2N+2))
    idx = CartesianIndices(Tuple(fill(2, N+1)))        

    for perm in permutations(1:N+1)
        x = zeros(fill(2, 2N+2)...)
        for k1 in idx
            k0 = CartesianIndex(Tuple(k1)[perm])
            x[k0, k1] = 1
        end
        x = vec(x)
        P += x * x'
    end

    F = eigen(P)
    Λ = Diagonal([x > atol ? 1 : 0 for x in F.values])
    P = F.vectors * Λ * F.vectors'

    P = I(2^N) ⊗ P ⊗ I(2^N)
    P = permutesystems(P, [2^(2N+1), 2^N, 2, 2^N], [1, 4, 3, 2])
    P = permutesystems(P, [2^(2N), 2, 2^(2N), 2], [1, 4, 3, 2])

    sparse(P)
end

function calculate_fidelity(N=1)
    """
        L = ∑_i L_i ⊗ |i><i|, is a network.
        A ⊗ B ⊗ in ⊗ out, is the system order for L.
        dim(A) = dim(B) = 2^N, dim(in) = dim(out) = 2.

        The conditions for the network:
            a) L_i ≥ 0,
            b) ∑_i L_i = ρ_A ⊗ \1_B ⊗ \1_in,
            c) tr(ρ_A) = 1.
    
        The commutation relation:
            d) ∀ i, ∀U ∈ U(2) [L_i, \1_A ⊗ U^{⊗N} ⊗ U* ] = 0.
    
        The objective function to maximize:
            e) f = 0.5 × ∑_i (<\1_{2^N}| ⊗ <i|) × L_i × (|\1_{2^N}> ⊗ |i>)
    """

    L = [ComplexVariable(2^(2N+1), 2^(2N+1)) for _=1:2]
    constraints = [Li in :SDP for Li in L]
    constraints += get_parallel_network_properties(L, N)
    
    P = c_projector(N)
    constraints += [P * vec(Li) == vec(Li) for Li in L]

    f = 1/2 * real(
        sum(
            L[i+1][2^(N+1)*b + 2b + i + 1, 2^(N+1)*a + 2a + i + 1] 
            for i=0:1 for a=0:(2^N-1) for b=0:(2^N-1)
        )
    )
    
    problem = maximize(f, constraints)
    solve!(problem, () -> SCS.Optimizer(verbose=false, eps = 1e-5))
    problem.optval
end

N = 1
calculate_fidelity(N)

# N / result: 1 / 0.7499999608916904
#           : 2 / 0.8333302340400098
#           : 3 / 