# relax neighbours of all vertices in list
function relax(g, D, subList, List, InList, vperthread, nverts)
    #show(List[1:nverts])
    @threads for i in List[1:nverts]
       @inbounds for j in neighbors(g,i)
          # weights[j,i] gives weight of edge i->j
          # atomic_min returns old value
          if atomic_min!(D[j], D[i][] + g.weights[j,i]) > D[j][]
             # atomic to avoid duplicates, then add to thread's list
             # atomic_cas returns old value
             if(atomic_cas!(InList[j],0,1) === 0)
                id = threadid()
                listSize = length(subList[id])
                listSize <= vperthread[id] && resize!(subList[id], 2*listSize)
                vperthread[id] += 1
                subList[id][vperthread[id]] = j
             end
          end
       end
    end
 end

 function copyVerts!(List, InList, sub, start, nper)
    for i in 1:nper
      InList[sub[i]][] = 0
      List[start+i] = sub[i]
    end
 end

 function ssspbfpargraph(g, source)
    n = nv(g)
    m = ne(g)
 
    # Vertex list
    List = Array{Int64,1}(undef,n)
    nt = nthreads()
    # Per thread vertex list buffers; will grow if too small
    subList = [similar(List, 10) for i = 1:nt]
    vperthread = zeros(Int64,nt) # Per thread vertex count
 
    T = eltype(g.weights)
    D = [Atomic{T}(typemax(T)) for k = 1:n]
    InList = [Atomic{Int}(0) for k = 1:n]
 
    D[source][] = zero(T)
    List[1] = source
    nverts = 1 # no. of vertices in list
 
    while nverts > 0
       relax(g, D, subList, List, InList, vperthread, nverts)
       # write thread private vertices to shared List
       start = cumsum(vperthread)
       nverts = start[nt]
       start -= vperthread # exclusive prefix sum
       @threads for i = 1:nt
          copyVerts!(List, InList, subList[i], start[i], vperthread[i])
       end
       vperthread .= 0
    end
    return D
 end