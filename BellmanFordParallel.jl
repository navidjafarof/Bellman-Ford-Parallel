using CUDA, Graphs, SimpleWeightedGraphs, DataStructures

# Relax neighbours of i
function relax(g, D, InList, i)
    @inbounds for j in neighbors(g,i)
        # weights[j,i] gives weight of edge i->j
        dist = D[i] + g.weights[j,i]
        if D[j] > dist
            atomic_min!(D, j, dist)
            InList[j] = true
        end
    end
end

function relax_kernel(g, D, InList, active_nodes)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(active_nodes)
        i = active_nodes[idx]
        relax(g, D, InList, i)
    end
    return nothing
end

function ssspbfgraph(g, source)
    n = nv(g)
    m = ne(g)

    # Sparse representation of active list
    InList = zeros(Bool,n)

    # Initialize distance array
    D = fill(typemax(eltype(g.weights)),n)
    D[source] = zero(eltype(g.weights))
    InList[source] = true

    # Move data to GPU
    g_d = CuGraph(g)
    D_d = CuArray(D)
    InList_d = CuArray(InList)

    # Calculate number of threads and blocks for CUDA kernel
    threads = 256
    blocks = ceil(Int, n / threads)

    while any(InList_d)
        # Find active nodes
        active_nodes = findall(x -> x == true, InList_d)

        # Reset InList for next iteration
        fill!(InList_d, false)

        # Call CUDA kernel
        @cuda threads=threads blocks=blocks relax_kernel(g_d, D_d, InList_d, active_nodes)

        # Synchronize to ensure all threads have completed
        CUDA.synchronize()
    end

    # Copy the result back to CPU
    D = Array(D_d)

    return D
end
