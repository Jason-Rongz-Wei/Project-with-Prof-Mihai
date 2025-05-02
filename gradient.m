function S_T_val = S_T_value(U_inner, U_full)
    [m_inner, n_inner] = size(U_inner);
    m = m_inner + 2;
    n = n_inner + 2;

    Delta_t = 1/100;
    Delta_x = 1/100;
    delta = 0.05;

    V_prime = @(x) x.^3 - x;
    U = U_full;
    U(2:end-1, 2:end-1) = U_inner;
    P = zeros(m-2, n-1);

    for i = 2:m-1
        for j = 1:n-1
            term1 = (U(i,j+1) - U(i,j)) / Delta_t;
            term2 = delta^(-1) * V_prime((U(i,j+1) + U(i,j)) / 2);
            term3 = (U(i+1,j+1) - 2*U(i,j+1) + U(i-1,j+1)) / Delta_x^2;
            term4 = (U(i+1,j) - 2*U(i,j) + U(i-1,j)) / Delta_x^2;
            P(i-1,j) = term1 + term2 - (delta / 2) * (term3 + term4);
        end
    end

    S_T_val = 0.5 * Delta_x * Delta_t * sum(P(:).^2);
end


function [S_T_val, grad_ST, grad_ST_vector] = ST_and_Gradient(U_inner, U_full)
    [m_inner, n_inner] = size(U_inner);
    S_T_val = S_T_value(U_inner, U_full);

    h = 1e-6;
    grad_ST = zeros(m_inner, n_inner);

    for i = 1:m_inner
        for j = 1:n_inner
            U_perturbed = U_inner;
            U_perturbed(i,j) = U_perturbed(i,j) + h;
            S_T_perturbed = S_T_value(U_perturbed, U_full);
            grad_ST(i,j) = (S_T_perturbed - S_T_val) / h;
        end
    end

    grad_ST_vector = reshape(grad_ST, [], 1);
end

function U_final = BFGS_Method(U_full)
    % Optimize interior of U using BFGS
    %[m, n] = size(U_full);
    U_inner = U_full(2:end-1, 2:end-1);

    epsilon = 1e-4;
    max_iter = 100;
    c1 = 1e-4;
    c2 = 0.9;
    num_save = 10;
    history_U = cell(1, num_save);
    save_count = 0;

    [S_T_val, ~, grad_ST_vec] = ST_and_Gradient(U_inner, U_full);
    H_k = eye(numel(U_inner));

    for k = 1:max_iter
        disp(k);
        p_k = -H_k * grad_ST_vec;

        alpha_k = 1;
        while true
            U_inner_new = U_inner + alpha_k * reshape(p_k, size(U_inner));
            U_full_new = U_full;
            U_full_new(2:end-1, 2:end-1) = U_inner_new;
            [S_T_new, ~, grad_ST_vec_new] = ST_and_Gradient(U_inner_new, U_full_new);
            if S_T_new <= S_T_val + c1 * alpha_k * (grad_ST_vec' * p_k)
                if grad_ST_vec_new' * p_k >= c2 * grad_ST_vec' * p_k
                    break;
                end
            end
            alpha_k = 0.5 * alpha_k;
        end

        s_k = alpha_k * p_k;
        y_k = grad_ST_vec_new - grad_ST_vec;
        rho_k = 1 / (y_k' * s_k);
        I = eye(length(p_k));
        H_k = (I - rho_k * (s_k * y_k')) * H_k * (I - rho_k * (y_k * s_k')) + rho_k * (s_k * s_k');

        if mod(k - 1, floor(max_iter / num_save)) == 0 && save_count < num_save
            save_count = save_count + 1;
            U_full_new(2:end-1, 2:end-1) = U_inner_new;
            history_U{save_count} = U_full_new;
        end

        U_inner = U_inner_new;
        grad_ST_vec = grad_ST_vec_new;
        S_T_val = S_T_new;

        if norm(grad_ST_vec) <= epsilon
            break;
        end
    end

    U_final = U_full;
    U_final(2:end-1, 2:end-1) = U_inner;
    save('U_history.mat', 'history_U');
end


%% Initialization
x = linspace(0, 1, 101);
t = linspace(0, 1, 101);
U = zeros(101, 101);
for idx_t = 1:length(t)
    current_t = t(idx_t);
    U(102 - idx_t, :) = (1 - 2 * current_t) * sin(pi * x);
end
U(:,1) = 0;
U(:,end) = 0;
U(1,:) = sin(pi * x);
U(end,:) = -sin(pi * x);

U_final = BFGS_Method(U);

load('U_history.mat', 'history_U');
figure;
for i = 1:length(history_U)
    subplot(2, 5, i);
    surf(x, t, history_U{i});
    shading interp;
    title(['Iteration ' num2str(i)]);
    xlabel('x'); ylabel('t'); zlabel('U');
end

%%
load('U_history.mat', 'history_U');
x = linspace(0, 1, 101);  
t = linspace(0, 1, 101);  
figure;
for i =1:length(history_U)
    subplot(2,5,i)
    U = history_U{i};  

    [X, T] = meshgrid(x, t);   
    Z = U;                      
    contour(X', Z', T', 30, 'LineColor', 'k'); 
    xlabel('x');
    ylabel('U(x,t)');
    title(sprintf('Side view of U(x,t)'));
    set(gca, 'YDir', 'normal');
end

%%
function U_final_new = BFGS_Energy_Minimization()
    % Parameters
    delta = 0.05;
    dx = 1/100;
    epsilon = 1e-4; 
    max_iter = 1000;
    c1 = 1e-2;
    %max_line_search_iter = 20;

    x = linspace(0, 1, 101);
    U = sin(pi * x)';
    U(1) = 0;
    U(end) = 0;

    U_interior = U(2:end-1);
    n = length(U_interior);

    E_val = energy(U_interior, delta, dx);
    grad = energy_grad(U_interior, delta, dx);
    H = eye(n);

    for k = 1:max_iter
        fprintf('Iteration %d, Norm of grad = %.6e\n', k, norm(grad));

        if norm(grad) < epsilon
            break;
        end

        p = -H * grad;

        % Line search with Armijo condition only
        alpha = 1;
        %count = 0;
        while true
            U_new = U_interior + alpha * p;
            E_new = energy(U_new, delta, dx);
            if E_new <= E_val + c1 * alpha * (grad' * p)
                break;
            end
            alpha = alpha * 0.5;
            %count = count + 1;
            %if count >= max_line_search_iter
                %warning('Line search failed at iteration %d. Using last alpha = %.2e', k, alpha);
                %break;
            %end
        end

        grad_new = energy_grad(U_new, delta, dx);
        s = alpha * p;
        y = grad_new - grad;
        rho = 1 / (y' * s);
        H = (eye(n) - rho * s * y') * H * (eye(n) - rho * y * s') + rho * (s * s');

        % Update
        U_interior = U_new;
        grad = grad_new;
        E_val = E_new;
    end

    U_final_new = [0; U_interior; 0];  % Add fixed boundaries back
end

function E = energy(U, delta, dx)
    V = @(u) 0.5 * (1 - u.^2).^2;
    dU = (U(3:end) - U(1:end-2)) / (2*dx);  % central diff
    V_mid = V(U(2:end-1));
    q = delta * dU.^2 + 2 * delta^(-1) * V_mid;
    E = 0.5 * dx * sum(q);
end

function grad = energy_grad(U, delta, dx)
    n = length(U);
    grad = zeros(n, 1);

    for i = 1:n
        u = U(i);
        V_prime = (1 - u^2) * (-u);

        if i == 1
            dF_forward = (U(i+2) - U(i)) / (2*dx);
            grad(i) = 0.5 * dx * (delta^(-1) * V_prime + 2 * delta * dF_forward * (-1/(2*dx)));

        elseif i == 2
            dF_backward = (U(i)) / (2*dx);
            dF_forward = (U(i+2) - U(i)) / (2*dx);
            grad(i) = 0.5 * dx * (2 * delta * dF_backward * (1/(2*dx)) + delta^(-1) * V_prime + 2 * delta * dF_forward * (-1/(2*dx)));

        elseif 2 < i && i <= n-2
            dF_backward = (U(i) - U(i-2)) / (2*dx);
            dF_forward = (U(i+2) - U(i)) / (2*dx);
            grad(i) = 0.5 * dx * (2 * delta * dF_backward * (1/(2*dx)) + delta^(-1) * V_prime + 2 * delta * dF_forward * (-1/(2*dx)));

        elseif i == n-1
            dF_backward = (U(i) - U(i-2)) / (2*dx);
            dF_forward = (-U(i)) / (2*dx);
            grad(i) = 0.5 * dx * (2 * delta * dF_backward * (1/(2*dx)) + delta^(-1) * V_prime + 2 * delta * dF_forward * (-1/(2*dx)));

        elseif i == n
            dF_backward = (U(i) - U(i-2)) / (2*dx);
            grad(i) = 0.5 * dx * (2 * delta * dF_backward * (1/(2*dx)) + delta^(-1) * V_prime);
        end
    end
end

U_final_new = BFGS_Energy_Minimization();

x = linspace(0, 1, 101);
plot(x, U_final_new, 'LineWidth', 2);
xlabel('x'); ylabel('U(x)');
title('Stationary State U(x)');
grid on;

%%
function U_final_new = Newton_Energy_Minimization()
    % Parameters
    delta = 0.05;
    dx = 1/400;
    epsilon = 1e-6;
    max_iter = 100;

    % Discretize space
    x = linspace(0, 1, 401);
    U = sin(pi * x)';
    U(1) = 0;
    U(end) = 0;
    
    U_interior = U(2:end-1);  % U_1 to U_99
    %n = length(U_interior);

    for k = 1:max_iter
        grad = energy_grad(U_interior, delta, dx);
        H = energy_hessian_newton(U_interior, delta, dx);

        fprintf('Iteration %d, Norm of grad = %.6e\n', k, norm(grad));
        if norm(grad) < epsilon
            break;
        end

        % Newton direction: p = -H^{-1} * grad
        p = -H \ grad;

        % Line search: fixed small alpha
        alpha = 1;
        U_interior = U_interior + alpha * p;
    end

    U_final_new = [0; U_interior; 0];
end

function H = energy_hessian_newton(U, delta, dx)
    n = length(U);
    H = zeros(n, n);
    eps = 1e-6;
    for i = 1:n
        e = zeros(n, 1);
        e(i) = eps;
        grad_plus = energy_grad(U + e, delta, dx);
        grad_minus = energy_grad(U - e, delta, dx);
        H(:, i) = (grad_plus - grad_minus) / (2 * eps);
    end
end

% Run and plot
U_final_new = Newton_Energy_Minimization();
x = linspace(0, 1, 401);
plot(x, U_final_new, 'LineWidth', 2);
xlabel('x'); ylabel('U(x)');
title('Stationary State U(x) using Newton method');
grid on;

%%
function U_final_gd = GradientDescent_Energy_Minimization()
    % Parameters
    delta = 0.05;
    dx = 1/100;
    epsilon = 1e-6;
    max_iter = 1000;
    alpha = 0.1;  % fixed step size for gradient descent

    % Discretize space
    x = linspace(0, 1, 101);
    U = sin(pi * x)';
    U(1) = 0;
    U(end) = 0;

    U_interior = U(2:end-1);  % U_1 to U_99
 
    for k = 1:max_iter
        grad = energy_grad(U_interior, delta, dx);

        fprintf('Iteration %d, Norm of grad = %.6e\n', k, norm(grad));
        if norm(grad) < epsilon
            break;
        end

        % Gradient descent update
        U_interior = U_interior - alpha * grad;
    end

    U_final_gd = [0; U_interior; 0];
end

% Run and plot
U_final_gd = GradientDescent_Energy_Minimization();
x = linspace(0, 1, 101);
plot(x, U_final_gd, 'LineWidth', 2);
xlabel('x'); ylabel('U(x)');
title('Stationary State U(x) using Gradient Descent');
grid on;

