using Graphs, SimpleWeightedGraphs, CUDA, SparseArrays

function graph_to_cusparse_csr(g)
    adj_matrix = Array{Float64}(weights(g))
    sparse_matrix = sparse(adj_matrix)
    cusparse_matrix = CUSPARSE.CuSparseMatrixCSR{Float64}(sparse_matrix)
    return cusparse_matrix
end

function relax_gpu_kernel(csr_matrix, distances, n)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if idx <= n
        row_start = csr_matrix.rowPtr[idx]
        row_end = csr_matrix.rowPtr[idx + 1] - 1

        for j = row_start:row_end
            col_idx = csr_matrix.colVal[j]
            weight = csr_matrix.nzVal[j]
            new_distance = distances[idx] + weight
            if distances[col_idx] > new_distance
                distances[col_idx] = new_distance
            end
        end
    end
    return nothing
end

function bellman_ford_gpu(graph, source)
    n = nv(graph)
    csr_matrix = graph_to_cusparse_csr(graph)
    distances = fill(Float64(Inf), n) |> CuArray
    distances[source] = 0.0

    for i in 1:n-1
        @cuda threads=256 blocks=ceil(Int, n/256) relax_gpu_kernel(csr_matrix, distances, n)
        synchronize()
    end
    return Array(distances)
end
