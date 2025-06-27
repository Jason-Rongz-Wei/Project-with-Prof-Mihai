using JuMP, Ipopt, Plots
using DelimitedFiles
using Random

U_sol = readdlm("U_initial.csv", ',')
I = Int(1e2 + 1)
J = Int(1e2 + 1)
Δx = 1 / (I - 1)
Δt = 1 / (J - 1)
δ = 0.03 

x = range(0, stop=1, length=I)

model = Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
set_optimizer_attribute(model, "limited_memory_update_type", "bfgs")
set_optimizer_attribute(model, "max_iter", 100)
@variable(model, U_inner[1:J-2, 1:I-2])

function construct_U_full(U_inner, U_sol, J, I)
    U_full = zeros(J, I)
    for i in 1:I
        U_full[1, i] = U_sol[i]
        U_full[end, i] = -U_sol[i]
    end
    for j in 1:J
        U_full[j, 1] = 0.0
        U_full[j, end] = 0.0
    end
    for j in 2:J-1, i in 2:I-1
        U_full[j, i] = U_inner[j-1, i-1]
    end
    return U_full
end

function V_prime(u)
    return -u * (1 - u^2)
end

function V_double_prime(u)
    return -(1 - 3u^2)
end

function ST_impl(U_vec::Vector{Float64})
    U_inner = reshape(U_vec, J-2, I-2)
    U_full = construct_U_full(U_inner, U_sol, J, I)

    total = 0.0
    for j in 1:J-1, i in 2:I-1
        u1 = U_full[j, i]
        u2 = U_full[j+1, i]
        u_avg = (u1 + u2)/2
        lap1 = (U_full[j, i+1] - 2u1 + U_full[j, i-1]) / Δx^2
        lap2 = (U_full[j+1, i+1] - 2u2 + U_full[j+1, i-1]) / Δx^2
        P = (u2 - u1)/Δt + (1/δ) * V_prime(u_avg) - (δ/2) * (lap1 + lap2)
        total += P^2
    end
    return 0.5 * Δx * Δt * total
end

function grad_ST_impl(U_vec::Vector{Float64})
    J_m = J - 2
    I_m = I - 2
    U_inner = reshape(U_vec, J_m, I_m)
    U_full = construct_U_full(U_inner, U_sol, J_m+2, I_m+2)

    P = zeros(J_m + 2, I_m + 2)
    grad = zeros(J_m, I_m)

    # ========== 1. 构造 P ==========
    for j in 1:J_m + 1
        for i in 2:I_m + 1
            term1 = (U_full[j+1, i] - U_full[j, i]) / Δt
            term2 = (1/δ) * V_prime((U_full[j+1, i] + U_full[j, i]) / 2)
            term3 = (U_full[j+1, i+1] - 2*U_full[j+1, i] + U_full[j+1, i-1]) / Δx^2
            term4 = (U_full[j, i+1]   - 2*U_full[j, i]   + U_full[j, i-1]) / Δx^2
            P[j, i] = term1 + term2 - (δ / 2) * (term3 + term4)
        end
    end

    for j in 1:J_m + 1
        for I in I_m+2:I_m + 2
            term5 = (U_full[j+1,I] - U_full[j,I])/Δt
            term6 = (1/δ)* V_prime((U_full[j+1, I] + U_full[j, I]) / 2)
            term7 = (2*U_full[j+1, I-1] - 2*U_full[j+1, I]) / Δx^2
            term8 = ( - 2*U_full[j, I]   + 2*U_full[j, I-1]) / Δx^2
            P[j,I] = term5 + term6 - (δ / 2) * (term7 + term8)
        end
    end

    for j in 1:J_m + 1
        for I in 1:1
            term5 = (U_full[j+1,I] - U_full[j,I])/Δt
            term6 = (1/δ)* V_prime((U_full[j+1, I] + U_full[j, I]) / 2)
            term7 = (2*U_full[j+1, I+1] - 2*U_full[j+1, I]) / Δx^2
            term8 = (-2*U_full[j, I]   + 2*U_full[j, I+1]) / Δx^2
            P[j,I] = term5 + term6 - (δ / 2) * (term7 + term8)
        end
    end

    for j in J_m+2:J_m + 2
        for I in I_m+2:I_m + 2
            term5 = (U_full[j-1,I] - U_full[j,I])/Δt
            term6 = (1/δ)* V_prime((U_full[j-1, I] + U_full[j, I]) / 2)
            term7 = (2*U_full[j-1, I-1] - 2*U_full[j-1, I]) / Δx^2
            term8 = (-2*U_full[j, I] + 2*U_full[j, I-1]) / Δx^2
            P[j,I] = term5 + term6 - (δ / 2) * (term7 + term8)
        end
    end

    for j in J_m+2:J_m + 2
        for I in 1:1
            term5 = (U_full[j-1,I] - U_full[j,I])/Δt
            term6 = (1/δ)* V_prime((U_full[j-1, I] + U_full[j, I]) / 2)
            term7 = (2*U_full[j-1, I+1] - 2*U_full[j-1, I]) / Δx^2
            term8 = (-2*U_full[j, I]   + 2*U_full[j, I+1]) / Δx^2
            P[j,I] = term5 + term6 - (δ / 2) * (term7 + term8)
        end
    end

    for j in J_m + 2: J_m+2
        for I in 2:I_m+1
            term1 = (U_full[j-1,I] - U_full[j,I])/Δt
            term2 = (1/δ)* V_prime((U_full[j-1, I] + U_full[j, I]) / 2)
            term3 = (U_full[j-1, I+1] - 2*U_full[j-1, I] + U_full[j-1, I-1]) / Δx^2
            term4 = (U_full[j, I+1] - 2*U_full[j,i]  + U_full[j, I-1]) / Δx^2
            P[j,I] = term1 + term2 - (δ / 2) * (term3 + term4)
        end
    end

    for j in 2:J_m + 1
        for i in 2:I_m + 1
            u      = U_full[j, i]
            u_prev = U_full[j-1, i]
            u_next = U_full[j+1, i]

            Vpp_b = -1 + 3 * (0.5 * (u_prev + u))^2
            Vpp_f = -1 + 3 * (0.5 * (u + u_next))^2

            P_b = P[j-1, i]
            P_f = P[j, i]

            term1 = (2*Δx + Δx*Δt/δ * Vpp_b + 2*δ*Δt/Δx) * P_b
            term2 = (2*Δx - Δx*Δt/δ * Vpp_f - 2*δ*Δt/Δx) * P_f

            cross = δ * Δt/Δx * (
                P[j-1, i-1] + P[j, i-1] + P[j-1, i+1] + P[j, i+1]
            )

            grad[j-1, i-1] = term1 - term2 - cross
        end
    end

    return reshape(grad, J_m * I_m)
end

function ST(args...)
    U_vec = collect(args)
    return ST_impl(U_vec)
end

function grad_ST(args...)
    U_vec = collect(args)
    return grad_ST_impl(U_vec)
end

JuMP.register(model, :ST, (J-2)*(I-2), ST, grad_ST)

vars = vec(U_inner) 
@NLobjective(model, Min, ST(vars...))


for j in 1:J-2, i in 1:I-2
    τ = j / (J - 1)
    U₀ = (1 - τ) * U_sol[i+1] + τ * (-U_sol[i+1])
    set_start_value(U_inner[j, i], U₀)
end

optimize!(model)

U_inner_opt = value.(U_inner)
U_opt = construct_U_full(U_inner_opt, U_sol, J, I)
writedlm("U_optimized.csv", U_opt, ',')