using Random:
    AbstractRNG, MersenneTwister, randperm, seed!, shuffle!
using Statistics: mean

using Graphs:
    getRNG, sample!

using Graphs, SimpleWeightedGraphs
"""
    kronecker(SCALE, edgefactor, A=0.57, B=0.19, C=0.19; seed=-1)
Generate a directed [Kronecker graph](https://en.wikipedia.org/wiki/Kronecker_graph)
with the default Graph500 parameters.
###
References
- http://www.graph500.org/specifications#alg:generator

Modified by Eric Aubanel, March 2020, add create weighted
graph with random weights between 0 and 1. Skips self-edges
"""
function kroneckerwt(SCALE, edgefactor, A=0.57, B=0.19, C=0.19; seed::Int=-1)
    N = 2^SCALE
    M = edgefactor * N
    ij = ones(Int, M, 2)
    ab = A + B
    c_norm = C / (1 - (A + B))
    a_norm = A / (A + B)
    rng = getRNG(seed)

    for ib = 1:SCALE
        ii_bit = rand(rng, M) .> (ab)  # bitarray
        jj_bit = rand(rng, M) .> (c_norm .* (ii_bit) + a_norm .* .!(ii_bit))
        ij .+= 2^(ib - 1) .* (hcat(ii_bit, jj_bit))
    end

    p = randperm(rng, N)
    ij = p[ij]

    p = randperm(rng, M)
    ij = ij[p, :]

    g = SimpleWeightedDiGraph(N)
    for (s, d) in zip(@view(ij[:, 1]), @view(ij[:, 2]))
        if s == d
            continue
        end
        add_edge!(g, s, d, rand(rng,1)[1])
    end
    return g
end