% Generate input data
x = linspace(-5, 5, 200)';  % Input data (200 points from -5 to 5)

% Define the covariance function
covfunc = {@covProd, {@covPeriodic, @covSEiso}};  % Product of periodic and SE covariance functions

% Set hyperparameters for the covariance function
hyp.cov = [-0.5; 0; 0; 2; 0];  % Log hyperparameters for the covariance functions

% Calculate the covariance matrix
K = feval(covfunc{:}, hyp.cov, x);  % Evaluate the covariance function
K = K + 1e-6 * eye(size(K));  % Add a small diagonal matrix for numerical stability

% Generate sample functions
L = chol(K, 'lower');  % Cholesky decomposition
f = L * randn(size(x, 1), 5);  % Generate 5 sample functions

% Plot the sample functions
figure;
hold on;
plot(x, f);  % Plot all sample functions
title('Sample Functions from GP with Periodic and SE Covariance');
xlabel('x');
ylabel('f(x)');
legend('Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5');
hold off;
