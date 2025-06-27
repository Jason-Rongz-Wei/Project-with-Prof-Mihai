using JuMP, Ipopt, Plots
using DelimitedFiles
using Random


I = Int(1e2 +1)
Δx = 1/(I - 1)
δ = 0.03
x = range(0, stop=1, length=I)
U_init = [sin(π * x[i]) for i in 1:I]

model = Model(Ipopt.Optimizer)

@variable(model, U[2:I-1])
for i in 2:I-1
    set_start_value(U[i], U_init[i])
end

U_full = Dict{Int, Any}()
for i in 1:I
    if i == 1 || i == I
        U_full[i] = 0.0
    else
        U_full[i] = U[i]
    end
end

@NLobjective(model, Min, (1/2) * Δx * sum((2 / δ) * (1 - U_full[i]^2)^2 / 4 +0.5 * δ * (((U_full[i+1] - U_full[i]) / Δx)^2 +((U_full[i] - U_full[i-1]) / Δx)^2) for i in 2:I-1))

optimize!(model)

U_vals = [value(U[i]) for i in 2:I-1]
U_sol = [0.0; U_vals; 0.0]
writedlm("U_initial.csv", U_sol, ',')

plot(x, U_sol, xlabel="x", ylabel="U(x)", title="Minimizer of E(U)", lw=2)


