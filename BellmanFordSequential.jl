using Graphs, SimpleWeightedGraphs, DataStructures
# Relax neighbours of i
function relax(g, D, List, InList, i)
    @inbounds for j in neighbors(g,i)
       # weights[j,i] gives weight of edge i->j
       if D[j] > D[i] + g.weights[j,i]
          D[j] = D[i] + g.weights[j,i]
          if(!InList[j])
             push!(List, j)
             InList[j] = true
          end
       end
    end
 end
 
 function ssspbfgraph(g, source)
    n = nv(g)
    m = ne(g)
 
    # Circular buffer to hold FIFO list
    List = CircularBuffer{Int}(n)
    # Sparse representation of list
    InList = zeros(Bool,n)
 
    D = fill(typemax(eltype(g.weights)),n)
    D[source] = zero(eltype(g.weights))
    push!(List, source)
    InList[source] = true
 
    while length(List) > 0
       i = popfirst!(List)
       InList[i] = false
       relax(g, D, List, InList, i)
    end
    return D
 end