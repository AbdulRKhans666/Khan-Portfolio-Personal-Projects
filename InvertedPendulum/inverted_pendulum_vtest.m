% ================================================================
%  INVERTED PENDULUM — LQR Controller
% ================================================================

clear; close all; clc;


pend_mass  = 50;     % pendulum bob mass    (kg)
cart_mass  = 10;     % cart mass            (kg)
arm_length = 3;     % pendulum arm length  (m)
g          = -9.81;   % gravity (m/s^2) (negative = downward)
d          = 1;     % cart friction coefficient
b          = 1;     % pendulum coupling coefficient

fprintf('=== INVERTED PENDULUM (LQR) ===\n');
fprintf('Cart mass: %g kg    Pendulum mass: %g kg\n', cart_mass, pend_mass);
fprintf('Arm length: %g m    g = %g m/s^2\n\n', arm_length, g);

A = [0,    1,                          0,    0;
     0,   -d/cart_mass,                b*pend_mass*g/cart_mass,    0;
     0,    0,                          0,    1;
     0,   -b*d/(cart_mass*arm_length), -b*(pend_mass+cart_mass)*g/(cart_mass*arm_length),  0];

B = [0;
     1/cart_mass;
     0;
     b/(cart_mass*arm_length)];

C = eye(4);   % observe all states (for LQR full-state feedback)
D = zeros(4,1);

fprintf('A =\n'); disp(A);
fprintf('B =\n'); disp(B);

% Open-loop eigenvalues
fprintf('Open-loop eigenvalues:\n'); disp(eig(A));

% Controllability check
co   = ctrb(A, B);
rnk  = rank(co);
fprintf('Controllability matrix rank: %d  (need %d)\n\n', rnk, size(A,1));



Q = eye(4);       % equal weight on all states
R = 0.0001;       % very small control cost → aggressive controller

K = lqr(A, B, Q, R);
fprintf('LQR gain K =\n'); disp(round(K, 4));

% Closed-loop check
Acl = A - B*K;
fprintf('Closed-loop eigenvalues:\n'); disp(eig(Acl));



wr = [0; 0; pi; 0];           % reference / target state
x0 = [-2; 0; pi + 0.5; 0];   % initial state

t_start = 0;
t_end   = 120;
dt      = 0.01;
t       = (t_start:dt:t_end)';
N       = length(t);

fprintf('Simulating %.1fs from x0 = [%.1f, %.1f, %.4f, %.1f]\n\n', ...
        t_end, x0(1), x0(2), x0(3), x0(4));


opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
[t_sol, sol] = ode45(@(t, x) pendcart(x, K, wr, pend_mass, cart_mass, ...
                                        arm_length, g, d), ...
                      t, x0, opts);

fprintf('Simulation complete.\n');
fprintf('Final state: x=%.3f  v=%.3f  theta=%.4f (pi=%.4f)  omega=%.4f\n\n', ...
        sol(end,1), sol(end,2), sol(end,3), pi, sol(end,4));

figure('Name','Closed-Loop State Response', ...
       'Color',[0.05 0.07 0.12], 'Position',[80 80 1100 600]);

state_labels = {'x  (cart position, m)', ...
                'v  (cart velocity, m/s)', ...
                '\theta  (pendulum angle, rad)', ...
                '\omega  (angular rate, rad/s)'};
state_cols   = {[0.30 0.70 1.00], [0.30 1.00 0.65], ...
                [1.00 0.80 0.25], [1.00 0.40 0.50]};
ref_vals     = [wr(1), wr(2), wr(3), wr(4)];

for i = 1:4
    ax = subplot(2, 2, i);
    set(ax, 'Color',[0.04 0.06 0.10], 'XColor',[0.5 0.6 0.7], ...
        'YColor',[0.5 0.6 0.7], 'GridColor',[0.12 0.18 0.25], 'GridAlpha',0.5);
    hold on; grid on;

    % Reference line
    yline(ref_vals(i), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);

    % State trajectory
    plot(t_sol, sol(:,i), '-', 'Color', state_cols{i}, 'LineWidth', 2);

    title(ax, state_labels{i}, 'Color',[0.70 0.85 1.00], 'FontSize', 11);
    xlabel('Time (s)', 'Color',[0.50 0.60 0.70]);
    xlim([0, t_end]);
end

annotation('textbox', [0 0.96 1 0.04], ...
    'String', 'Closed-Loop Response — Inverted Pendulum LQR Controller', ...
    'Color', [0.40 0.65 1.00], 'FontSize', 13, 'FontWeight', 'bold', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
    'FontName', 'Courier New');


% animations part
fig_anim = figure('Name', 'Inverted Pendulum Animation', ...
                  'Color', [0.05 0.07 0.12], 'Position', [80 50 900 500]);

ax_a = axes('Parent', fig_anim, 'Color', [0.04 0.06 0.10], ...
    'XColor', [0.4 0.5 0.6], 'YColor', [0.4 0.5 0.6], ...
    'GridColor', [0.1 0.15 0.2], 'GridAlpha', 0.5);
hold(ax_a, 'on'); grid(ax_a, 'on');
xlim(ax_a, [-5, 5]); ylim(ax_a, [0, 5]);
xlabel(ax_a, 'Position (m)', 'Color', [0.5 0.6 0.7]);
title(ax_a, 'Inverted Pendulum Animation', ...
      'Color', [0.65 0.82 1.0], 'FontSize', 12, 'FontName', 'Courier New');

% Static ground line
plot(ax_a, [-5, 5], [0, 0], '-', 'Color', [0.4 0.5 0.6], 'LineWidth', 2);
% Reference x position marker
xline(ax_a, wr(1), '--', 'Color', [0.3 0.7 0.3 0.5], 'LineWidth', 1);

% Dynamic graphic handles
h_wheel_l = scatter(ax_a, 0, 0.25, 600, [0.3 0.3 0.35], 'filled');
h_wheel_r = scatter(ax_a, 0, 0.25, 600, [0.3 0.3 0.35], 'filled');
h_body    = plot(ax_a, [0 0], [0.65 0.65], '-', 'Color', [0.2 0.5 0.9], 'LineWidth', 20);
h_arm     = plot(ax_a, [0 0], [0.7 2.7],   '-', 'Color', [0.85 0.85 0.85], 'LineWidth', 3);
h_bob     = scatter(ax_a, 0, 2.7, 600, [1.0 0.3 0.2], 'filled');

% Text overlays
h_txt_t = text(ax_a, -4.7, 4.6, 'T = 0.00s',   'Color',[0.65 0.82 1.0],'FontSize',10,'FontName','Courier New');
h_txt_x = text(ax_a, -4.7, 4.2, 'x = 0.00m',   'Color',[0.30 0.70 1.0],'FontSize',10,'FontName','Courier New');
h_txt_a = text(ax_a, -4.7, 3.8, '\theta = 0.00','Color',[1.00 0.80 0.25],'FontSize',10,'FontName','Courier New');

fprintf('Playing animation (%d frames)...\n', N);

skip = 1;   % draw every Nth frame (increase to speed up)
for i = 1:skip:N
    x_c   = sol(i, 1);    % cart position
    theta = sol(i, 3);    % pendulum angle

    % Cart body and wheels (matches Python layout)
    x_c_vis = max(min(x_c, 4.5), -4.5);   % clamp to view
    set(h_wheel_l, 'XData', x_c_vis - 0.5, 'YData', 0.25);
    set(h_wheel_r, 'XData', x_c_vis + 0.5, 'YData', 0.25);
    set(h_body,    'XData', [x_c_vis - 0.8, x_c_vis + 0.8], 'YData', [0.65, 0.65]);

    % Pendulum arm tip — exact Python formula:
    %   tip_x = x - arm*sin(-theta)
    %   tip_y = 0.7 - arm*cos(-theta)
    tip_x = x_c_vis - arm_length * sin(-theta);
    tip_y = 0.7     - arm_length * cos(-theta);
    set(h_arm, 'XData', [x_c_vis, tip_x], 'YData', [0.7, tip_y]);
    set(h_bob, 'XData', tip_x, 'YData', tip_y);

    % Bob colour: green = near upright, yellow = warning, red = far
    angle_err = abs(theta - pi);
    if angle_err < 0.05
        set(h_bob, 'CData', [0.15 0.90 0.25]);
    elseif angle_err < 0.3
        set(h_bob, 'CData', [1.00 0.80 0.15]);
    else
        set(h_bob, 'CData', [1.00 0.25 0.15]);
    end

    % Text
    set(h_txt_t, 'String', sprintf('T = %.2f s',  t_sol(i)));
    set(h_txt_x, 'String', sprintf('x = %.3f m',  x_c));
    set(h_txt_a, 'String', sprintf('theta = %.3f rad', theta));

    drawnow limitrate;
end

fprintf('Animation complete.\n');












function dxdt = pendcart(x, K, wr, pend_mass, cart_mass, arm_length, g, d)
    x = x(:);   % ensure column vector

    Sx = sin(x(3));
    Cx = cos(x(3));

    % System determinant (same as Python's D)
    Det = pend_mass * (arm_length^2) * (pend_mass + cart_mass*(1 - Cx^2));

    % LQR reference-tracking control: u = -K*(x - wr)
    u = -K * (x - wr);

    % Equations of motion (direct translation of Python pendcart)
    x1_dot = x(2);

    x2_dot = (1/Det) * ( -(pend_mass^2) * (arm_length^2) * g * Cx * Sx ...
                         + pend_mass * (arm_length^2) * (pend_mass*arm_length*(x(4)^2)*Sx - d*x(2)) ) ...
             + pend_mass * (arm_length^2) * (1/Det) * u;

    x3_dot = x(4);

    x4_dot = (1/Det) * ( (pend_mass + cart_mass) * pend_mass * g * arm_length * Sx ...
                         - pend_mass * arm_length * Cx * (pend_mass*arm_length*(x(4)^2)*Sx - d*x(2)) ) ...
             - pend_mass * arm_length * Cx * (1/Det) * u;

    dxdt = [x1_dot; x2_dot; x3_dot; x4_dot];
end