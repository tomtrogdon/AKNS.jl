module AKNS
using Base, ApproxFun, ApproxFunBase, ApproxFunFourier, Reexport, FFTW, LinearAlgebra, ApproxFunOrthogonalPolynomials, ApproxFunRational, Speci
#, MacroTools#, Reexport, AbstractFFTs, FFTW, InfiniteArrays, FillArrays, FastTransforms, IntervalSets,

export AKNS, ODEFT, AKNS_rat, discreteRHP, Residue

struct ODEFT
    D::BandedMatrix
    T::BandedMatrix
    G::Fun
end

function ODEFT(g,n,m,L=1.0)
    sp1 = OscRational(0.,L);
    G = Fun(zai(g),sp1,n+1)
    D = Derivative(sp1)
    T = Array(I,n,n)
    T = hcat([1;1;zeros(n-2)],T)
    T = vcat(T,zeros(m-n,n+1))
    Dn = hcat(zeros(m),D[1:m,1:n])
    ODEFT(BandedMatrix(Dn,(1,3)),BandedMatrix(T,(1,3)),G)
end

function  (F::ODEFT)(k)
    ω = (4*pi*F.G.space.domain.L*exp(-F.G.space.domain.L*abs(k)))
    S = copy(F.D)
    S[:,2:end] -= 1im*k*F.T[:,2:end]
    S[:,1] -= F.T[:,1]
    m = size(S)[1]
    (SparseMatrixCSC(S)\pad(F.G.coefficients,m))[1]*ω
end


u1 = (x,w) -> w ≈ 0 ? 1im*pi + 2im*atan(x) : exp(1im*x*w)*( exp(-w + loggamma(0,1im*x*w-w)) - exp(w + loggamma(0,1im*x*w+w)) + exp(w)*( x >= 0 ? 2im*pi : 0.0))

ϕrat = (x,w) -> (w <= 0 ? u1(x,w) : -u1(x,-w) |> conj)/(2im)
ϕrat∞ = (w) -> pi*exp(-abs(w))
function Φrat(x,w)
    return ϕrat(x,w)/ϕrat∞(w)
end
rat = x -> 1/(1 + x^2)

σ = 1;
ϕG = (x,w) -> (0.5*sqrt(pi/σ))*exp.(-w^2/(4*σ))*(1 .+ erf.(x*sqrt(σ) .+ 1im*w/(2*sqrt(σ))))
ϕG∞ = (w) -> (sqrt(pi/σ))*exp.(-w^2/(4*σ))
function ΦG(x,w)
    exp.(1im*x*w).*(0.5*(1 .+ erf.(x*sqrt(σ) .+ 1im*w/(2*sqrt(σ)))))
end
Ga = x -> exp(-σ*x^2)

struct AKNS
    qΦG
    D::SparseMatrixCSC
    T::SparseMatrixCSC
    GaFun::Fun
    Mqmat::SparseMatrixCSC
    Mrmat::SparseMatrixCSC
    Q::Fun
    R::Fun
    ϕG
    ϕG∞
end

function AKNS_rat(f1,f2,n,m)
    L = 1.0
    sp1 = OscRational(0.,L)
    GaFun = Fun(zai(rat),sp1,m)
    Q = Fun(zai(q),sp1,n+2)
    R = Fun(zai(r),sp1,n+2)
    qΦG = (q,n,w) -> Fun(x -> Φrat(x,w)*q(x),sp1,n+2)
    D = Derivative(sp1)
    Dn = D[1:m,1:n+1] |> sparse
    Dn = hcat(pad(GaFun.coefficients[1:end]/ϕrat∞(0.0),m),Dn)
    Mr = Multiplication(R,sp1)
    Mq = Multiplication(Q,sp1)
    T = zeros(Complex{Float64},m,n+2)
    T[1:n+1,2:n+2] += Array(I,n+1,n+1)
    T = T |> SparseMatrixCSC
    Mqmat = Mq[1:m,1:n+1] |> sparse
    Mrmat = Mr[1:m,1:n+1] |> sparse
    AKNS(qΦG,Dn,T,GaFun,Mqmat,Mrmat,Q,R,ϕrat,ϕrat∞)
end

function AKNS(f1,f2,n,m)
    L = 1.0
    sp1 = OscRational(0.,L)
    GaFun = Fun(zai(Ga),sp1,m)
    Q = Fun(zai(q),sp1,n+2)
    R = Fun(zai(r),sp1,n+2)
    qΦG = (q,n,w) -> Fun(x -> ΦG(x,w)*q(x),sp1,n+2)
    D = Derivative(sp1)
    Dn = D[1:m,1:n+1] |> sparse
    Dn = hcat(pad(GaFun.coefficients[1:end]/ϕG∞(0.0),m),Dn)
    Mr = Multiplication(R,sp1)
    Mq = Multiplication(Q,sp1)
    T = zeros(Complex{Float64},m,n+2)
    T[1:n+1,2:n+2] += Array(I,n+1,n+1)
    T = T |> SparseMatrixCSC
    Mqmat = Mq[1:m,1:n+1] |> sparse
    Mrmat = Mr[1:m,1:n+1] |> sparse
    AKNS(qΦG,Dn,T,GaFun,Mqmat,Mrmat,Q,R,ϕG,ϕG∞)
end

function (A::AKNS)(k)
    if abs(k) > 1e15
        return Array(I,2,2)*(1+0.0im)
    end
    m = size(A.D)[1]
    n = size(A.D)[2] -1
    σ = 1
    osc = A.qΦG(A.Q,n,2k*σ)
    non_osc = A.qΦG(A.R,n,0.0)
    D_osc = deepcopy(A.D)
    D_osc[:,1] = D_osc[:,1]*A.ϕG∞(0.0)/A.ϕG∞(2k)
    D_osc -= 2im*k*σ*A.T
    Mn1 = -hcat(pad(osc.coefficients[1:end],m),A.Mqmat)
    Mn2 = -hcat(pad(non_osc.coefficients[1:end],m),A.Mrmat)

    S1 = vcat(hcat(A.D[:,1:end-1],Mn1[:,1:end-1]),hcat(Mn2[:,1:end-1],D_osc[:,1:end-1]))
    S1[:,n+1] *= A.ϕG∞(2k)
    b = [zeros(m,1);pad(A.R.coefficients[1:end],m)]
    c1 = (S1\b)[[1;n+1]]

    σ = -1
    osc = A.qΦG(A.R,n,2k*σ)
    non_osc = A.qΦG(A.Q,n,0.0)
    D_osc -= 4im*k*σ*A.T
    Mn1 = -hcat(pad(osc.coefficients[1:end],m),A.Mrmat)
    Mn2 = -hcat(pad(non_osc.coefficients[1:end],m),A.Mqmat)

    S1 = vcat(hcat(A.D[:,1:end-1],Mn1[:,1:end-1]),hcat(Mn2[:,1:end-1],D_osc[:,1:end-1]))
    S1[:,n+1] *= A.ϕG∞(2k)
    b = [zeros(m,1);pad(A.Q.coefficients[1:end],m)]
    c2 = (S1\b)[1:n:end]
    J = hcat(c1,c2 |>reverse) + I
    J[1,2] *= A.ϕG∞(2k)
    J[2,1] *= A.ϕG∞(2k)
    J

end

function λ(A::AKNS)
    n = size(A.D)[2]-1
    S1 = vcat(hcat(1im*A.D[1:n,2:n+1],-1im*A.Mqmat[1:n,1:n]),hcat(1im*A.Mrmat[1:n,1:n],-1im*A.D[1:n,2:n+1]))
    lam = eigvals(S1 |> Array)
    S = abs.(imag(lam)) .> 1e-1
    lam[S]
end

struct discreteRHP
    zp::Vector
    zm::Vector
    U::Array
end

function discreteRHP(zp::Vector,zm::Vector,cp::Vector,cm::Vector)
    ZP = repeat(zp,1,length(zm))
    ZM = repeat(zm,1,length(zp)) |> transpose

    M12 = -1.0./(ZP - ZM)
    M21 = -1.0./(ZM - ZP) |> transpose
    M12 = cp.*M12
    M21 = cm.*M21
    sol = vcat(hcat(I,M12),hcat(M21,I))\vcat(hcat(0*cp,cp),hcat(cm,0*cm))
    discreteRHP(zp,zm,sol)
end


function (M::discreteRHP)(z)
    r1 = [1+sum(M.U[1:length(M.zp),1]./(z .- M.zp));sum(M.U[length(M.zp)+1:end,1]./(z .- M.zm))]
    r2 = [sum(M.U[1:length(M.zp),2]./(z .- M.zp));1+sum(M.U[length(M.zp)+1:end,2]./(z .- M.zm))]
    hcat(r1,r2) |> transpose
end

function Residue(M::discreteRHP)
    r1 = [sum(M.U[1:length(zp),1]);sum(M.U[length(zp)+1:end,1])]
    r2 = [sum(M.U[1:length(zp),2]);sum(M.U[length(zp)+1:end,2])]
    hcat(r1,r2) |> transpose
end




end#module
