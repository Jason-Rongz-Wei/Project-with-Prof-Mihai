function S_T_val = S_T_value(U_inner, U_full)
    [m_inner, n_inner] = size(U_inner);
    m = m_inner + 2;
    n = n_inner + 2;

    Delta_t = 1/100;
    Delta_x = 1/100;
    delta = 0.03;

    V_prime = @(x) x.^3 - x;
    U = U_full;
    U(2:end-1, 2:end-1) = U_inner;
    P = zeros(n-1, m-2);

    for j= 1:n-1
        for i = 2:m-1
            term1 = (U(j+1,i) - U(j,i)) / Delta_t;
            term2 = delta^(-1) * V_prime((U(j+1,i) + U(j,i)) / 2);
            term3 = (U(j+1,i+1) - 2*U(j+1,i) + U(j+1,i-1)) / Delta_x^2;
            term4 = (U(j,i+1) - 2*U(j,i) + U(j,i-1)) / Delta_x^2;
            P(j,i-1) = term1 + term2 - (delta / 2) * (term3 + term4);
        end
    end

    S_T_val = 0.5 * Delta_x * Delta_t * sum(P(:).^2);
end


function [S_T_val, grad_ST, grad_ST_vector] = ST_and_Gradient(U_inner, U_full)
    [m_inner, n_inner] = size(U_inner);
    
    Delta_x = 1/100;
    Delta_t = 1/100;
    delta = 0.03;
    
    V_prime = @(x) x.^3 - x;
    V_double_prime = @(x) 3*x.^2 - 1;

    % Compute S_T value
    S_T_val = S_T_value(U_inner, U_full);

    % Compute gradient
    grad_ST = zeros(m_inner, n_inner);
    
    for i = 1:n_inner  % time direction (1 to 99)
        for j = 1:m_inner  % space direction (1 to 99)
            jj = j + 1;  % map to full U index (interior starts from row 2)
            ii = i + 1;

            % Fetch U from full matrix
            U = U_full;

            % Precompute terms safely with padding assumed in U_full
            if i == 1
                % Left boundary column
                grad_ST(j,i) = compute_left_gradient(U, jj, ii, Delta_x, Delta_t, delta, V_prime, V_double_prime);
            elseif i == n_inner
                % Right boundary column
                grad_ST(j,i) = compute_right_gradient(U, jj, ii, Delta_x, Delta_t, delta, V_prime, V_double_prime);
            else
                % Interior
                grad_ST(j,i) = compute_inner_gradient(U, jj, ii, Delta_x, Delta_t, delta, V_prime, V_double_prime);
            end
        end
    end

    grad_ST_vector = reshape(grad_ST, [], 1);
end

function val = compute_inner_gradient(U, j, i, dx, dt, delta, Vp, Vpp)
    % Implements central inner gradient expression for 2 <= i <= 98
    term = @(j_,i_) (U(j_,i_+1) - 2*U(j_,i_) + U(j_,i_-1))/dx^2;

    A = (U(j,i) - U(j-1,i))/dt + delta^(-1)*Vp((U(j,i)+U(j-1,i))/2) ...
        - 0.5*delta*(term(j,i) + term(j-1,i));
    B = (U(j+1,i) - U(j,i))/dt + delta^(-1)*Vp((U(j+1,i)+U(j,i))/2) ...
        - 0.5*delta*(term(j+1,i) + term(j,i));

    DA = (1/dt + 0.5*delta^(-1)*Vpp((U(j,i)+U(j-1,i))/2) + delta/dx^2);
    DB = (-1/dt + 0.5*delta^(-1)*Vpp((U(j+1,i)+U(j,i))/2) + delta/dx^2);

    left1 = (U(j,i-1) - U(j-1,i-1))/dt + delta^(-1)*Vp((U(j,i-1)+U(j-1,i-1))/2) ...
            - 0.5*delta*( (U(j,i)-2*U(j,i-1)+U(j,i-2))/dx^2 + (U(j-1,i)-2*U(j-1,i-1)+U(j-1,i-2))/dx^2 );
    left2 = (U(j+1,i-1) - U(j,i-1))/dt + delta^(-1)*Vp((U(j+1,i-1)+U(j,i-1))/2) ...
            - 0.5*delta*( (U(j+1,i)-2*U(j+1,i-1)+U(j+1,i-2))/dx^2 + (U(j,i)-2*U(j,i-1)+U(j,i-2))/dx^2 );

    right1 = (U(j,i+1) - U(j-1,i+1))/dt + delta^(-1)*Vp((U(j,i+1)+U(j-1,i+1))/2) ...
            - 0.5*delta*( (U(j,i+2)-2*U(j,i+1)+U(j,i))/dx^2 + (U(j-1,i+2)-2*U(j-1,i+1)+U(j-1,i))/dx^2 );
    right2 = (U(j+1,i+1) - U(j,i+1))/dt + delta^(-1)*Vp((U(j+1,i+1)+U(j,i+1))/2) ...
            - 0.5*delta*( (U(j+1,i+2)-2*U(j+1,i+1)+U(j+1,i))/dx^2 + (U(j,i+2)-2*U(j,i+1)+U(j,i))/dx^2 );

    val = 0.5*dx*dt * ( ...
        - left1 * (delta/dx^2) ...
        - left2 * ( delta/dx^2) ...
        - right1 * ( delta/dx^2) ...
        - right2 * ( delta/dx^2) ...
        + 2*A*DA + 2*B*DB );
end

function val = compute_left_gradient(U, j, i, dx, dt, delta, Vp, Vpp)
    % i = 1 case (left edge of interior)
    term = @(j_,i_) (U(j_,i_+1) - 2*U(j_,i_) + U(j_,i_-1))/dx^2;

    A = (U(j,i) - U(j-1,i))/dt + delta^(-1)*Vp((U(j,i)+U(j-1,i))/2) ...
        - 0.5*delta*(term(j,i) + term(j-1,i));
    B = (U(j+1,i) - U(j,i))/dt + delta^(-1)*Vp((U(j+1,i)+U(j,i))/2) ...
        - 0.5*delta*(term(j+1,i) + term(j,i));

    DA = (1/dt + 0.5*delta^(-1)*Vpp((U(j,i)+U(j-1,i))/2) + delta/dx^2);
    DB = (-1/dt + 0.5*delta^(-1)*Vpp((U(j+1,i)+U(j,i))/2) + delta/dx^2);

    right1 = (U(j,i+1) - U(j-1,i+1))/dt + delta^(-1)*Vp((U(j,i+1)+U(j-1,i+1))/2) ...
            - 0.5*delta*( (U(j,i+2)-2*U(j,i+1)+U(j,i))/dx^2 + (U(j-1,i+2)-2*U(j-1,i+1)+U(j-1,i))/dx^2 );
    right2 = (U(j+1,i+1) - U(j,i+1))/dt + delta^(-1)*Vp((U(j+1,i+1)+U(j,i+1))/2) ...
            - 0.5*delta*( (U(j+1,i+2)-2*U(j+1,i+1)+U(j+1,i))/dx^2 + (U(j,i+2)-2*U(j,i+1)+U(j,i))/dx^2 );

    val = 0.5*dx*dt * ( 2*A*DA + 2*B*DB - right1*delta/dx^2 - right2*delta/dx^2 );
end

function val = compute_right_gradient(U, j, i, dx, dt, delta, Vp, Vpp)
    term = @(j_,i_) (U(j_,i_+1) - 2*U(j_,i_) + U(j_,i_-1))/dx^2;

    A = (U(j,i) - U(j-1,i))/dt + delta^(-1)*Vp((U(j,i)+U(j-1,i))/2) ...
        - 0.5*delta*(term(j,i) + term(j-1,i));
    B = (U(j+1,i) - U(j,i))/dt + delta^(-1)*Vp((U(j+1,i)+U(j,i))/2) ...
        - 0.5*delta*(term(j+1,i) + term(j,i));

    DA = (1/dt + 0.5*delta^(-1)*Vpp((U(j,i)+U(j-1,i))/2) + delta/dx^2);
    DB = (-1/dt + 0.5*delta^(-1)*Vpp((U(j+1,i)+U(j,i))/2) + delta/dx^2);

    left1 = (U(j,i-1) - U(j-1,i-1))/dt + delta^(-1)*Vp((U(j,i-1)+U(j-1,i-1))/2) ...
            - 0.5*delta*( (U(j,i)-2*U(j,i-1)+U(j,i-2))/dx^2 + (U(j-1,i)-2*U(j-1,i-1)+U(j-1,i-2))/dx^2 );
    left2 = (U(j+1,i-1) - U(j,i-1))/dt + delta^(-1)*Vp((U(j+1,i-1)+U(j,i-1))/2) ...
            - 0.5*delta*( (U(j+1,i)-2*U(j+1,i-1)+U(j+1,i-2))/dx^2 + (U(j,i)-2*U(j,i-1)+U(j,i-2))/dx^2 );

    val = 0.5*dx*dt * ( 2*A*DA + 2*B*DB - left1*delta/dx^2 - left2*delta/dx^2 );
end

function U_final = BFGS_Method(U_full)
    % Optimize interior of U using BFGS
    %[m, n] = size(U_full);
    U_inner = U_full(2:end-1, 2:end-1);

    epsilon = 1e-4;
    max_iter = 200;
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
            U_inner = reshape(U_inner, [], 1);
            U_inner_new = U_inner + alpha_k * p_k;
            U_inner_new_mat = reshape(U_inner_new, 99, 99);
            U_full_new = U_full;
            U_full_new(2:100, 2:100) = U_inner_new_mat;
            [S_T_new, ~, grad_ST_vec_new] = ST_and_Gradient(U_inner_new_mat, U_full_new);
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
            %U_full_new(2:end-1, 2:end-1) = U_inner_new;
            history_U{save_count} = U_full_new;
        end

        U_inner = U_inner_new_mat;
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
U = zeros(101, 101);
U(:,1) = 0;
U(:,end) = 0;
U(1,:) = U_final_new;
U(end,:) = -U_final_new;

for j = 2:100  
    tau = (j - 1) / 100; 
    U(2:100, j) = (1 - tau) * U(1, j) + tau * U(end, j);
end

U_final = BFGS_Method(U);
%%
x = linspace(0, 1, 101);
t = linspace(0, 1, 101);
load('U_history.mat', 'history_U');
figure;
for i = 1:length(history_U)
    subplot(2, 5, i);
    surf(x, t, history_U{i});
    shading interp;
    title(['Iteration ' num2str(i)]);
    xlabel('x'); ylabel('t'); zlabel('U');
end

figure;
for i = 1:10
    subplot(2, 5, i);
    U = history_U{i};

    hold on;
    for ti = 1:length(t)
        plot(x, U(ti, :), 'k');
    end
    xlabel('x');
    ylabel('U(x,t)');
    title('Side view of U(x,t)');
    ylim([-1.1, 1.1]);
end

%%
%function U_final_new = BFGS_Energy_Minimization()
    % Parameters
    %delta = 0.05;
    %dx = 1/100;
    %epsilon = 1e-4; 
    %max_iter = 1000;
    %c1 = 1e-2;
    %max_line_search_iter = 20;

    %x = linspace(0, 1, 101);
    %U = sin(pi*x)';
    %U(1) = 0;
    %U(end) = 0;

    %U_interior = U(2:end-1);
    %n = length(U_interior);

    %E_val = energy(U_interior, delta, dx);
    %grad = energy_grad(U_interior, delta, dx);
    %H = eye(n);

    %for k = 1:max_iter
        %fprintf('Iteration %d, Norm of grad = %.6e\n', k, norm(grad));

        %if norm(grad) < epsilon
            %break;
        %end

        %p = -H * grad;

        % Line search with Armijo condition only
        %alpha = 1;
        %count = 0;
        %while true
            %U_new = U_interior + alpha * p;
            %E_new = energy(U_new, delta, dx);
            %if E_new <= E_val + c1 * alpha * (grad' * p)
                %break;
            %end
            %alpha = alpha * 0.5;
            %count = count + 1;
            %if count >= max_line_search_iter
                %warning('Line search failed at iteration %d. Using last alpha = %.2e', k, alpha);
                %break;
            %end
        %end

        %grad_new = energy_grad(U_new, delta, dx);
        %s = alpha * p;
        %y = grad_new - grad;
        %rho = 1 / (y' * s);
        %H = (eye(n) - rho * s * y') * H * (eye(n) - rho * y * s') + rho * (s * s');

        % Update
        %U_interior = U_new;
        %grad = grad_new;
        %E_val = E_new;
    %end

    %U_final_new = [0; U_interior; 0];  % Add fixed boundaries back
%end

%function E = energy(U, delta, dx)
    %V = @(u) 0.25 * (1 - u.^2).^2;
    %dU = (U(3:end) - U(1:end-2)) / (2*dx);  % central diff
    %V_mid = V(U(2:end-1));
    %q = delta * dU.^2 + 2 * delta^(-1) * V_mid;
    %E = 0.5 * dx * sum(q);
%end

function grad = energy_grad(U, delta, dx)
    n = length(U);
    grad = zeros(n, 1);

    for i = 1:n
        u = U(i);
        V_prime = -2 * u * (1 - u^2);

        if i == 1
            grad(i) = 0.5 * dx * (delta^(-1) * V_prime - 2 * delta * (U(i+1) - U(i))/dx^2 + delta * U(i) / dx^2);
        elseif i == n
            grad(i) = 0.5 * dx * (2 * delta * (U(i) - U(i-1)) / dx^2 - 2 * delta^(-1) * (1 - U(i)^2) * U(i) + delta * U(i) / dx^2);
        else
            grad(i) = 0.5 * dx * (2 * delta * (U(i) - U(i-1)) / dx^2 - 2 * delta^(-1) * (1 - U(i)^2) * U(i) - 2 * delta * (U(i+1) - U(i)) / dx^2);
        end
    end
end

%U_final_new = BFGS_Energy_Minimization();

%x = linspace(0, 1, 101);
%plot(x, U_final_new, 'LineWidth', 2);
%xlabel('x'); ylabel('U(x)');
%title('Stationary State U(x)');
%grid on;

function U_final_new = Newton_Energy_Minimization()

    delta = 0.05;
    dx = 1/100;
    epsilon = 1e-6;
    max_iter = 100;

    x = linspace(0, 1, 101);
    U = sin(pi * x)';
    U(1) = 0;
    U(end) = 0;
    
    U_interior = U(2:end-1);
    %n = length(U_interior);

    for k = 1:max_iter
        grad = energy_grad(U_interior, delta, dx);
        H = energy_hessian_newton(U_interior, delta, dx);

        fprintf('Iteration %d, Norm of grad = %.6e\n', k, norm(grad));
        if norm(grad) < epsilon
            break;
        end

        p = -H \ grad;

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

U_final_new = Newton_Energy_Minimization();
x = linspace(0, 1, 101);
plot(x, U_final_new, 'LineWidth', 2);
xlabel('x'); ylabel('U(x)');
title('Stationary State U(x) using Newton method');
grid on;


%%
function U_final_gd = GradientDescent_Energy_Minimization(gap)

    delta = 0.05;
    dx = 1/gap;
    epsilon = 1e-6;
    max_iter = 1000;
    alpha = 0.1;

    x = linspace(0, 1, gap+1);
    U = sin(pi * x)';
    U(1) = 0;
    U(end) = 0;

    U_interior = U(2:end-1);
 
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

tr = 100;
U_final_gd = GradientDescent_Energy_Minimization(tr);
x = linspace(0, 1, tr+1);
plot(x, U_final_gd, 'LineWidth', 2);
xlabel('x'); ylabel('U(x)');
title('Stationary State U(x) using Gradient Descent');
grid on;