## No soliton case first

struct AKNSdet
    f1::SumFun
    f2::SumFun
    quad
end

function remove_deltas!(s::SumFun)
    i = 0
    while i < length(s.funs)
        i += 1
        if typeof(s.funs[i].space) <: DiracSpace
            deleteat!(s.funs,i)
            i -= 1
        end
    end
    return s
end

function AKNSdet(ρ₁,ρ₂,n,m)
    F = FourierTransform(1.0)
    f1 = F*Fun(ρ₁,OscLaurent(0.0),m) |> SumFun |> remove_deltas!
    f2 = F*Fun(ρ₂,OscLaurent(0.0),m) |> SumFun |> remove_deltas!



    quad = gausslegendre(n)
    AKNSdet(f1,f2,quad)


end
