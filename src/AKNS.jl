module AKNS
using Base, ApproxFun, ApproxFunBase, SpecialFunctions, ApproxFunFourier, Reexport, LinearAlgebra,
    ApproxFunRational, BandedMatrices, SparseArrays
#, MacroTools#, Reexport, AbstractFFTs, FFTW, InfiniteArrays, FillArrays, FastTransforms, IntervalSets,

export AKNSscattering, ODEFT, AKNSscattering_rat, discreteRHP, Residue, ODEFT_Gaussian, ODEFT_solve, grideval

u1 = (x,w) -> w ≈ 0 ? 1im*pi + 2im*atan(x) : exp(1im*x*w)*( exp(-w + loggamma(0,1im*x*w-w)) - exp(w + loggamma(0,1im*x*w+w)) + exp(w)*( x >= 0 ? 2im*pi : 0.0))
ϕrat = (x,w) -> (w <= 0 ? u1(x,w) : -u1(x,-w) |> conj)/(2im)
ϕrat∞ = (w) -> pi*exp(-abs(w))
function Φrat(x,w)
    return ϕrat(x,w)/ϕrat∞(w)
end
rat = x -> 1/(1 + x^2)

σ = 1.;
ϕGl = (x,w) -> (0.5*sqrt(pi/σ))*exp.(-w^2/(4*σ))*(1 .+ erf.(x*sqrt(σ) .+ 1im*w/(2*sqrt(σ))))
ϕG_best = (x,w) -> (sqrt(pi/σ))*exp.(-w^2/(4*σ) + 1im*w*x) .- (0.5*sqrt(pi/σ))*exp(-σ*x^2)*erfcx.(x*sqrt(σ) .+ 1im*w/(2*sqrt(σ)))
ϕG = (x,w) -> ( x < -8 ? 0.0im : ϕG_best(x,w))

ϕG∞ = (w) -> (sqrt(pi/σ))*exp.(-w^2/(4*σ))
function ΦG(x,w)
    exp.(1im*x*w).*(0.5*(1 .+ erf.(x*sqrt(σ) .+ 1im*w/(2*sqrt(σ)))))
end
Ga = x -> exp(-σ*x^2)

ϕsech∞ = (w) -> 2*atan(exp(w))

struct ODEFT
    D::SparseMatrixCSC
    T::SparseMatrixCSC
    G::Fun
    ω
end

struct ODEFTsol
    uu::Fun
    c0::ComplexF64
    ϕ::ComplexF64
    k::Float64
end

function ODEFT(g,n::Integer,m::Integer,L=1.0)
    sp1 = OscRational(0.,L);
    G = Fun(zai(g),sp1,m+1)
    D = Derivative(sp1)
    T = Array(I,n,n)
    T = hcat([1;1;zeros(n-2)],T)
    T = vcat(T,zeros(m-n,n+1))
    Dn = hcat(zeros(m),D[1:m,1:n])
    ω = k -> (4*pi*G.space.domain.L*exp(-G.space.domain.L*abs(k)))
    ODEFT(sparse(Dn),sparse(T),G,ω)
end

function ODEFT_Gaussian(g,n::Integer,m::Integer,L=1.0)
    sp1 = OscRational(0.,L);
    G = Fun(zai(g),sp1,m+1)
    D = Derivative(sp1)
    T = Array(I,n,n)
    FF = Fun(zai(Ga),sp1,m+1)
    T = vcat(T,zeros(m-n,n))
    T = hcat(-FF.coefficients[1:m],T)
    Dn = hcat(zeros(m),D[1:m,1:n])
    ODEFT(sparse(Dn),sparse(T),G,ϕG∞)
end

function  (F::ODEFT)(k)
    #ω = (4*pi*F.G.space.domain.L*exp(-F.G.space.domain.L*abs(k)))
    S = copy(F.D)
    S[:,2:end] -= 1im*k*F.T[:,2:end]
    S[:,1] -= F.T[:,1]
    m = size(S)[1]
    S = SparseMatrixCSC(S)
    u = S\pad(F.G.coefficients,m)
    (u[1]*F.ω(k), norm(S*u-pad(F.G.coefficients,m)))
end

function ODEFT_solve(F::ODEFT,k)
    S = copy(F.D)
    S[:,2:end] -= 1im*k*F.T[:,2:end]
    S[:,1] -= F.T[:,1]
    m = size(S)[1]
    u = (SparseMatrixCSC(S)\pad(F.G.coefficients,m))
    c0 = u[1]
    sp = F.G.space
    uu = Fun(sp |> ApproxFunRational.OscRational_to_OscLaurent, u[2:end] |> ApproxFunRational.coefs_to_OscLaurent)
    ODEFTsol(uu,c0,F.ω(k),k)
end

function (F::ODEFTsol)(x)
    pts = points(F.uu)
    F.c0*ϕ*ΦG(x, F.k) + F.uu(x)
end

function grideval(F::ODEFTsol)
    pts = points(F.uu)
    (F.c0*map(x -> ϕG(x, F.k), pts) + values(F.uu),pts)
end

function normgrideval(F::ODEFTsol)
    pts = points(F.uu)
    (map(x -> ΦG(x, F.k), pts) + values(F.uu)/F.c0,pts)
end

struct AKNSscattering
    qΦG
    D::SparseMatrixCSC
    T::SparseMatrixCSC
    GaFun::Fun
    Mqmat::SparseMatrixCSC
    Mrmat::SparseMatrixCSC
    Q::Fun
    R::Fun
    ϕG∞
end

function rat_qΦG(q,n,w,sp)
    Fhat = ODEFT_Gaussian(rat,n+1,n+101)
    y = ODEFT_solve(Fhat,w) |> grideval
    u = transform(sp |> ApproxFunRational.OscRational_to_OscLaurent,y[1].*map(q,y[2]))
    Fun(sp, u |> ApproxFunRational.coefs_to_OscRational)
end

function sech_qΦG(q,n,w,sp)
    Fhat = ODEFT_Gaussian(sech,n+1,n+101)
    y = ODEFT_solve(Fhat,w) |> grideval
    u = transform(sp |> ApproxFunRational.OscRational_to_OscLaurent,y[1].*map(q,y[2]))
    Fun(sp, u |> ApproxFunRational.coefs_to_OscRational)
end

function AKNSscattering_rat(f1,f2,n,m)
    L = 1.0
    sp1 = OscRational(0.,L)
    GaFun = Fun(zai(rat),sp1,m)
    Q = Fun(zai(f1),sp1,n+2)
    R = Fun(zai(f2),sp1,n+2)
    #qΦG = (q,n,w) -> Fun(x -> ϕrat(x,w)*q(x),sp1,n+2)
    qΦG = (q,n,w) -> rat_qΦG(q,n,w,sp1)
    #return (qΦG1,qΦG)
    D = Derivative(sp1)
    Dn = D[1:m,1:n+1] |> sparse
    #Dn = hcat(pad(GaFun.coefficients[1:end]/ϕrat∞(0.0),m),Dn)
    Dn = hcat(pad(GaFun.coefficients[1:end],m),Dn)
    Mr = Multiplication(R,sp1)
    Mq = Multiplication(Q,sp1)
    T = zeros(Complex{Float64},m,n+2)
    T[1:n+1,2:n+2] += Array(I,n+1,n+1)
    T = T |> SparseMatrixCSC
    Mqmat = Mq[1:m,1:n+1] |> sparse
    Mrmat = Mr[1:m,1:n+1] |> sparse
    AKNSscattering(qΦG,Dn,T,GaFun,Mqmat,Mrmat,Q,R,ϕrat∞)
end

function AKNSscattering_sech(f1,f2,n,m)
    L = 1.0
    sp1 = OscRational(0.,L)
    GaFun = Fun(zai(sech),sp1,m)
    Q = Fun(zai(f1),sp1,n+2)
    R = Fun(zai(f2),sp1,n+2)
    #qΦG = (q,n,w) -> Fun(x -> ϕrat(x,w)*q(x),sp1,n+2)
    qΦG = (q,n,w) -> sech_qΦG(q,n,w,sp1)
    #return (qΦG1,qΦG)
    D = Derivative(sp1)
    Dn = D[1:m,1:n+1] |> sparse
    #Dn = hcat(pad(GaFun.coefficients[1:end]/ϕrat∞(0.0),m),Dn)
    Dn = hcat(pad(GaFun.coefficients[1:end],m),Dn)
    Mr = Multiplication(R,sp1)
    Mq = Multiplication(Q,sp1)
    T = zeros(Complex{Float64},m,n+2)
    T[1:n+1,2:n+2] += Array(I,n+1,n+1)
    T = T |> SparseMatrixCSC
    Mqmat = Mq[1:m,1:n+1] |> sparse
    Mrmat = Mr[1:m,1:n+1] |> sparse
    AKNSscattering(qΦG,Dn,T,GaFun,Mqmat,Mrmat,Q,R,ϕsech∞)
end

function AKNSscattering(f1,f2,n,m,L = 1.0)
    sp1 = OscRational(0.,L)
    GaFun = Fun(zai(Ga),sp1,m)
    Q = Fun(zai(f1),sp1,n+2)
    R = Fun(zai(f2),sp1,n+2)
    qΦG = (q,n,w) -> Fun(x -> ϕG(x,w)*q(x),sp1,n+2)
    D = Derivative(sp1)
    Dn = D[1:m,1:n+1] |> sparse
    Dn = hcat(pad(GaFun.coefficients[1:end],m),Dn)
    Mr = Multiplication(R,sp1)
    Mq = Multiplication(Q,sp1)
    T = zeros(Complex{Float64},m,n+2)
    T[1:n+1,2:n+2] += Array(I,n+1,n+1)
    T = T |> SparseMatrixCSC
    Mqmat = Mq[1:m,1:n+1] |> sparse
    Mrmat = Mr[1:m,1:n+1] |> sparse
    AKNSscattering(qΦG,Dn,T,GaFun,Mqmat,Mrmat,Q,R,ϕG∞)
end

function _evalAKNS(A,k,bool)
    if abs(k) > 1e15
        return Array(I,2,2)*(1+0.0im)
    end
    m = size(A.D)[1]
    n = size(A.D)[2] -1
    σ = 1
    osc = A.qΦG(A.Q,m,2k*σ)
    non_osc = A.qΦG(A.R,m,0.0)
    D_osc = copy(A.D)
    #D_osc[:,1] = D_osc[:,1]*A.ϕG∞(0.0)
    D_osc -= 2im*k*σ*A.T
    Mn1 = -hcat(pad(osc.coefficients[1:end],m),A.Mqmat)
    Mn2 = -hcat(pad(non_osc.coefficients[1:end],m),A.Mrmat)

    S1 = vcat(hcat(A.D[:,1:end-1],Mn1[:,1:end-1]),hcat(Mn2[:,1:end-1],D_osc[:,1:end-1]))
    #S1 = vcat(hcat(D_osc[:,1:end-1],Mn2[:,1:end-1]),hcat(Mn1[:,1:end-1],A.D[:,1:end-1]))
    #S1[:,n+1] *= A.ϕG∞(2k)
    b = [zeros(m,1);pad(A.R.coefficients[1:end],m)]
    u1 = (S1\b)
    c1 = u1[[1;n+1]]

    σ = -1
    osc = A.qΦG(A.R,n,2k*σ)
    non_osc = A.qΦG(A.Q,n,0.0)
    D_osc -= 4im*k*σ*A.T
    Mn1 = -hcat(pad(osc.coefficients[1:end],m),A.Mrmat)
    Mn2 = -hcat(pad(non_osc.coefficients[1:end],m),A.Mqmat)

    S1 = vcat(hcat(A.D[:,1:end-1],Mn1[:,1:end-1]),hcat(Mn2[:,1:end-1],D_osc[:,1:end-1]))
    #S1[:,n+1] *= A.ϕG∞(2k)
    b = [zeros(m,1);pad(A.Q.coefficients[1:end],m)]
    u2 = (S1\b)
    c2 = u2[1:n:end]
    J = hcat(c1 ,c2 |>reverse)
    S = copy(J)
    J[1,2] *= A.ϕG∞(-2k)
    J[2,1] *= A.ϕG∞(2k)
    J[1,1] *= A.ϕG∞(0)
    J[2,2] *= A.ϕG∞(0)
    #(J + I,u1,u2)
    if bool
        return (S,u1,u2,non_osc.space)
    else
        return J + I
    end
end


function _evalAKNS1(A,k)
    if abs(k) > 1e15
        return Array(I,2,2)*(1+0.0im)
    end
    m = size(A.D)[1]
    n = size(A.D)[2] -1
    σ = 1
    osc = A.qΦG(A.Q,m,2k*σ)
    non_osc = A.qΦG(A.R,m,0.0)
    D_osc = copy(A.D)
    #D_osc[:,1] = D_osc[:,1]*A.ϕG∞(0.0)
    D_osc -= 2im*k*σ*A.T
    Mn1 = -hcat(pad(osc.coefficients[1:end],m),A.Mqmat)
    Mn2 = -hcat(pad(non_osc.coefficients[1:end],m),A.Mrmat)

    S1 = vcat(hcat(A.D[:,1:end-1],Mn1[:,1:end-1]),hcat(Mn2[:,1:end-1],D_osc[:,1:end-1]))
    #S1 = vcat(hcat(D_osc[:,1:end-1],Mn2[:,1:end-1]),hcat(Mn1[:,1:end-1],A.D[:,1:end-1]))
    #S1[:,n+1] *= A.ϕG∞(2k)
    b = [zeros(m,1);pad(A.R.coefficients[1:end],m)]
    u1 = (S1\b)
    c1 = u1[[1;n+1]]

    c1[2] *= A.ϕG∞(2k)
    c1[1] *= A.ϕG∞(0)
    c1[1] += 1.0
    return c1
end

function _evalAKNS2(A,k)
    if abs(k) > 1e15
        return Array(I,2,2)*(1+0.0im)
    end
    m = size(A.D)[1]
    n = size(A.D)[2] -1
    # σ = 1
    # osc = A.qΦG(A.Q,m,2k*σ)
    # non_osc = A.qΦG(A.R,m,0.0)

    # #D_osc[:,1] = D_osc[:,1]*A.ϕG∞(0.0)
    # D_osc -= 2im*k*σ*A.T
    # Mn1 = -hcat(pad(osc.coefficients[1:end],m),A.Mqmat)
    # Mn2 = -hcat(pad(non_osc.coefficients[1:end],m),A.Mrmat)
    #
    # S1 = vcat(hcat(A.D[:,1:end-1],Mn1[:,1:end-1]),hcat(Mn2[:,1:end-1],D_osc[:,1:end-1]))
    # #S1 = vcat(hcat(D_osc[:,1:end-1],Mn2[:,1:end-1]),hcat(Mn1[:,1:end-1],A.D[:,1:end-1]))
    # #S1[:,n+1] *= A.ϕG∞(2k)
    # b = [zeros(m,1);pad(A.R.coefficients[1:end],m)]
    # u1 = (S1\b)
    # c1 = u1[[1;n+1]]

    σ = -1
    D_osc = copy(A.D)
    osc = A.qΦG(A.R,n,2k*σ)
    non_osc = A.qΦG(A.Q,n,0.0)
    D_osc -= 2im*k*σ*A.T
    Mn1 = -hcat(pad(osc.coefficients[1:end],m),A.Mrmat)
    Mn2 = -hcat(pad(non_osc.coefficients[1:end],m),A.Mqmat)

    S1 = vcat(hcat(A.D[:,1:end-1],Mn1[:,1:end-1]),hcat(Mn2[:,1:end-1],D_osc[:,1:end-1]))
    #S1[:,n+1] *= A.ϕG∞(2k)
    b = [zeros(m,1);pad(A.Q.coefficients[1:end],m)]
    u2 = (S1\b)
    c2 = u2[1:n:end] |> reverse
    c2[1] *= A.ϕG∞(-2k)
    #J[2,1] *= A.ϕG∞(2k)
    #J[1,1] *= A.ϕG∞(0)
    c2[2] *= A.ϕG∞(0)
    c2[2] += 1.0
    #(J + I,u1,u2)
    return c2
end

function (A::AKNSscattering)(k)
    _evalAKNS(A,k,false)
end

function (A::AKNSscattering)(k,j)
    if j == 1
        _evalAKNS1(A,k)
    else
        _evalAKNS2(A,k)
    end
end

struct Jost
    k1::Number
    k2::Number
    c1::Number
    c2::Number
    u1::Fun
    u2::Fun
end

function (J::Jost)(x)
    [J.c1*ϕG(x,J.k1) + J.u1(x); J.c2*ϕG(x,J.k2) + J.u2(x)]
end

function Jost(A::AKNSscattering,k)
    (S,u1,u2,sp) = _evalAKNS(A,k,true)
    n = (length(u1) ÷ 2)-1

    (c1,c2) = S[:,1]
    f1 = Fun(sp,u1[2:n])
    f2 = Fun(sp,u1[n+3:end])
    J1 = Jost(0.0,2k,c1,c2,f1,f2)

    (c1,c2) = S[:,2]
    f2 = Fun(sp,u2[2:n])
    f1 = Fun(sp,u2[n+3:end])
    J2 = Jost(-2k,0.0,c1,c2,f1,f2)
    return (J1,J2)
end

function λ(A::AKNSscattering)
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
    r1 = [sum(M.U[1:length(M.zp),1]);sum(M.U[length(M.zp)+1:end,1])]
    r2 = [sum(M.U[1:length(M.zp),2]);sum(M.U[length(M.zp)+1:end,2])]
    hcat(r1,r2) |> transpose
end




end#module
