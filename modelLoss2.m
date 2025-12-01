function [loss, gradients] = modelLoss2(dlnet, dl_t_f, dl_t_b, ...
                                        P0, I0, rhoP, rhoI, K, alpha)
% modelLoss2: función de pérdida para PINN (valores normalizados)
% Entradas:
%   dlnet   - red dlnetwork
%   dl_t_f  - dlarray de puntos de física (CB)
%   dl_t_b  - dlarray de puntos de condición inicial (CB)
%   P0, I0  - condiciones iniciales normalizadas (escala 0..1)
%   rhoP, rhoI, K, alpha - parámetros (K no se usa en la versión normalizada)
%
% Salidas:
%   loss - valor escalar de pérdida (dlarray)
%   gradients - gradientes para actualizar dlnet

% SoftPlus numéricamente estable (funciona con dlarray y con doubles)
softplus = @(x) log(1 + exp(-abs(x))) + max(x,0);

%%%%%%%%%%%%%%  PUNTOS INTERNOS  %%%%%%%%%%%%%%%%%%%%%
Y_f = forward(dlnet, dl_t_f);      % Y_f es dlarray 2xN
P_f = softplus(Y_f(1,:));          % P_f normalizado (dlarray 1xN)
I_f = softplus(Y_f(2,:));          % I_f normalizado (dlarray 1xN)

% Derivadas temporales (respecto a la variable dl_t_f)
dPdt = dlgradient(sum(P_f), dl_t_f);
dIdt = dlgradient(sum(I_f), dl_t_f);

% Normalizamos N = P + I (ya está en escala 0..1)
N_f = P_f + I_f;

% Ecuaciones normalizadas (1 - N_f) ya está correcto
f1 = dPdt - (rhoP .* P_f .* (1 - N_f) - alpha .* P_f .* I_f);
f2 = dIdt - (rhoI .* I_f .* (1 - N_f) + alpha .* P_f .* I_f);

%%%%%%%%%%%%%%  CONDICIÓN INICIAL  %%%%%%%%%%%%%%%%%%%%%
Y_b = forward(dlnet, dl_t_b);
P_b = softplus(Y_b(1,:));
I_b = softplus(Y_b(2,:));

%%%%%%%%%%%%%%  FUNCIÓN DE PÉRDIDA  %%%%%%%%%%%%%%%%%%%%%
loss_f = mean(f1.^2 + f2.^2);                 % residuo físico
loss_b = mean((P_b - P0).^2 + (I_b - I0).^2); % condición inicial (normalizada)
loss_reg = 1e-6 * (mean(1./(1 + P_f)) + mean(1./(1 + I_f))); % leve penalización para evitar colapso

loss = loss_f + loss_b + loss_reg;

%%%%%%%%%%%%%%  GRADIENTES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gradients = dlgradient(loss, dlnet.Learnables);
end
