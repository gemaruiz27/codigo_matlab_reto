%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% principal.m
% PINN para el sistema tumoral P–I (script principal)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

% Parámetros del modelo (valores "reales")
rhoP  = 0.6;      % tasa proliferativa
rhoI  = 0.15;     % tasa invasiva
alpha = 0.03;     % conversión P -> I
beta  = 0.05;     % actualmente no usado en la versión P-I normalizada
K     = 1e9;      % capacidad de carga (se usa para escalamiento)

% Intervalo temporal
t0 = 0;
tf = 200;
Nf = 2000; % puntos internos para física
Nb = 1;    % condición inicial

% Condiciones iniciales (valores en unidades reales)
P0 = 2e5;   % células proliferativas iniciales
I0 = 2e5;   % células invasivas iniciales

rng(1);

% === ESCALAMIENTO ===
scale = K;       % normalizamos respecto a K (dominio [0,1])
P0_n = P0 / scale;
I0_n = I0 / scale;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construcción del dominio
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t_f = t0 + (tf - t0) * rand(Nf, 1); % puntos internos aleatorios en (t0,tf)
t_b = t0 * ones(Nb, 1);             % punto de condición inicial

% Conversión a dlarray con formato 'CB' (Channel x Batch)
dl_t_f = dlarray(t_f', 'CB');
dl_t_b = dlarray(t_b', 'CB');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Arquitectura de la PINN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
layers = [
    featureInputLayer(1, "Normalization", "none")
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(2)  % salidas: [P_hat, I_hat] (normalizadas)
];
dlnet = dlnetwork(layers);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Entrenamiento con ADAM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numEpochs = 6000;
learnRate = 1e-3;
avgGrad = [];
avgSqGrad = [];

for epoch = 1:numEpochs
    % NOTA: pasamos P0_n e I0_n (condiciones iniciales normalizadas)
    [loss, grads] = dlfeval(@modelLoss2, dlnet, dl_t_f, dl_t_b, ...
                         P0_n, I0_n, rhoP, rhoI, K, alpha);
    [dlnet, avgGrad, avgSqGrad] = adamupdate(dlnet, grads, ...
                                avgGrad, avgSqGrad, epoch, learnRate);

    if mod(epoch, 500) == 0
        fprintf("Epoch %d - Loss = %.4e\n", epoch, double(loss));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluación del modelo (desescalado para graficar en unidades reales)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t_test = linspace(t0, tf, 400)';
dl_t_test = dlarray(t_test', 'CB');

Y_pred = extractdata(forward(dlnet, dl_t_test));  % output numérico 2xN

% SoftPlus estable (numérico) aplicado a salidas (estas salidas son normalizadas)
softplus = @(x) log(1 + exp(-abs(x))) + max(x,0);

% Primero salida en dominio normalizado, luego desescalo multiplicando por scale
P_pred = scale * softplus(Y_pred(1,:))';
I_pred = scale * softplus(Y_pred(2,:))';
N_pred = P_pred + I_pred;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gráficas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Position',[100 100 700 600]);
subplot(3,1,1);
plot(t_test, P_pred, 'LineWidth', 2);
xlabel('t'); ylabel('P(t)'); title('Células proliferativas');

subplot(3,1,2);
plot(t_test, I_pred, 'LineWidth', 2);
xlabel('t'); ylabel('I(t)'); title('Células invasivas');

subplot(3,1,3);
plot(t_test, N_pred, 'LineWidth', 2);
xlabel('t'); ylabel('N = P + I'); title('Carga tumoral total');
