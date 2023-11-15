% Crear un vector de valores de k
k_values = [5, 6, 7, 8, 9, 10, 11, 12];
#k_values = [5, 6, 7, 8];

% Inicializar listas para almacenar los tiempos de ejecución de svd y svdCompact
execution_times_svd = [];
execution_times_svdCompact = [];

for k = k_values
    % Generar una matriz aleatoria A de tamaño 2^k x 2^(k-1)
    A = rand(2^k, 2^(k-1));
    
    % Medir el tiempo de ejecución de la función svd
    tic; % Inicia el temporizador
    [U, S, V] = svd(A);
    time_svd = toc; % Detiene el temporizador
    
    % Medir el tiempo de ejecución de la función svdCompact
    tic; % Inicia el temporizador
    [Ur, Sr, Vr] = svdCompact(A);
    time_svdCompact = toc; % Detiene el temporizador
    
    % Almacena los tiempos de ejecución en las listas
    execution_times_svd = [execution_times_svd, time_svd];
    execution_times_svdCompact = [execution_times_svdCompact, time_svdCompact];
end

% ... (código anterior)

% Graficar los tiempos de ejecución de svd y svdCompact
plot(k_values, execution_times_svdCompact, 'o-');
hold on;
plot(k_values, execution_times_svd, 'o-');
xlabel('Valor de k');
ylabel('Tiempo de ejecución (segundos)');
title('Tiempo de ejecución de svdCompact y svd vs. k');
legend('svdCompact', 'svd');
grid on;


pause;