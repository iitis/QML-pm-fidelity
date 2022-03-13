using LinearAlgebra
using Combinatorics
using QuantumInformation
using SparseArrays

function superoperator(N, atol=1e-3)
    """
        Returns the superoperator of the channel:
            ϕ(Y) = ∫_U dU (U* ⊗ U^{⊗N}) Y (U* ⊗ U^{⊗N})^†
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

    sparse(permutesystems(P, [2, 2^N, 2, 2^N], [3, 2, 1, 4]))    
end

function diff_q_op(N, atol=1e-03)
    """
        Returns the dynamical matrix of the difference of channels:
            ϕ_0(X) = ∫_U dU (U* ⊗ U^{⊗N}) (|0><0| ⊗ X) (U* ⊗ U^{⊗N})^†
            ϕ_1(X) = ∫_U dU (U* ⊗ U^{⊗N}) (|1><1| ⊗ X) (U* ⊗ U^{⊗N})^†
    """

    S = superoperator(N, atol)
    Φ_0 = S * (ket(1,2) ⊗ I(2^N) ⊗ ket(1,2) ⊗ I(2^N))
    Φ_1 = S * (ket(2,2) ⊗ I(2^N) ⊗ ket(2,2) ⊗ I(2^N))
    return DynamicalMatrix(reshuffle(Φ_0 - Φ_1), 2^N, 2^(N+1))
end

function calculate_p_opt(N, atol=1e-03)
    """
        Calulate the maximal value of probability (p_opt) of discrimination of
        channels:
            ϕ_0(X) = ∫_U dU (U* ⊗ U^{⊗N}) (|0><0| ⊗ X) (U* ⊗ U^{⊗N})^†
            ϕ_1(X) = ∫_U dU (U* ⊗ U^{⊗N}) (|1><1| ⊗ X) (U* ⊗ U^{⊗N})^†
        
        p_opt = 0.5 + 0.25 * |||ϕ_0 - ϕ_1|||,
        where ||| ⋅ ||| is the diamond norm. 
    """

    1/2 + 1/4 * norm_diamond(diff_q_op(N), Val(:primal), atol)
end

function calculate_p_opt_dual(N, atol=1e-03)
    """
        Calulate the dual value for calculate_p_opt(N)
    """

    1/2 + 1/4 * norm_diamond(diff_q_op(N), Val(:dual), atol)
end

N = 1
print(calculate_p_opt_dual(N), " ≤ p_opt ≤ ", calculate_p_opt(N))
