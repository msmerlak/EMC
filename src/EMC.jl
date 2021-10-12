using LightGraphs
using Agents
import LinearAlgebra:dot

@agent Replicator GraphAgent begin
    fitness::Float64
end

function initialize(A; α = 1.)

    p = Dict{Symbol, Any}()
    p[:A] = A

    G = Graph(A)
    
    p[:counts] = zeros(Int, length(vertices(G)))
    p[:degrees] = degree(G)
    p[:adjlist] = G.fadjlist

    p[:R] = NaN

    p[:n] = α * length(vertices(G)) |> ceil |> Int

    model = AgentBasedModel(Replicator, GraphSpace(G), properties = p)

    for i in 1:p[:n]
        add_agent!(Replicator(i, 1, 1.), model)
    end
    return model
end

function spawn!(replicator, model)
    # for _ in 1:model.degrees[replicator.pos]
        id = nextid(model)
        pos = rand(model.rng, model.adjlist[replicator.pos])
        add_agent_pos!(
            Replicator(id, pos, Float64(model.degrees[pos])), 
            model
            )
        kill_agent!(replicator, model)
    # end
end

function select!(model)
    for replicator in allagents(model)
        model.counts[replicator.pos] += 1
    end
    Agents.sample!(model, model.n, :fitness)
    model.R = dot(model.counts, model.A * model.counts)/dot(model.counts, model.counts)
end


