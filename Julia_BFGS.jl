using JuMP, Ipopt, Plots
using DelimitedFiles
using Random

U_sol = readdlm("U_initial.csv", ',')

I = Int(1e2+1)
J = Int(1e2+1)
Δx = 1/(I-1)
Δt = 1/(J-1)

x = range(0, stop=1, length=I)

model = Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
set_optimizer_attribute(model, "limited_memory_update_type", "bfgs")
set_optimizer_attribute(model, "max_iter", 100000)
set_optimizer_attribute(model, "tol", 1e-10)

@variable(model, U[1:J, 1:I])

for i in 1:I
    fix(U[1, i], U_sol[i]; force = true)      
    fix(U[end, i], -U_sol[i]; force = true)   
end
for j in 1:J
    fix(U[j, 1], 0.0; force = true)           
    fix(U[j, end], 0.0; force = true)      
end

for j in 2:I-1
    τ = (j - 1) / (J - 1)
    for i in 2:J-1
        u_top = U_sol[j]
        u_bottom = -U_sol[j]
        U₀ = (1 - τ) * u_top + τ * u_bottom
        set_start_value(U[i, j], U₀)
    end
end

function V_prime(u)
    return -u * (1 - u^2)
end
register(model, :V_prime, 1, V_prime; autodiff = true)

@NLobjective(model, Min,
    (1/2) * Δx * Δt * sum(((U[j+1,i] - U[j,i]) / Δt + (1/δ) * V_prime((U[j+1,i] + U[j,i]) / 2) - (δ/2) * ((U[j+1,i+1] - 2*U[j+1,i] + U[j+1,i-1]) / Δx^2 + (U[j,i+1]   - 2*U[j,i]   + U[j,i-1])   / Δx^2))^2
        for j in 1:J-1, i in 2:I-1)
)

optimize!(model)

U_opt = value.(U)
writedlm("U_optimized.csv", U_opt, ',')


