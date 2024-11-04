% Load the data from cw1a.mat
data = load('cw1a.mat');  % Load data
x = data.x;               % Input data (features)
y = data.y;               % Target data (outputs)

% Set initial hyperparameters for the Gaussian Process
hyp.cov = [0; 0; log(1)]; % Log length-scale, log signal variance, and log period (for covPeriodic)
hyp.lik = 0;              % Log noise variance (for likGauss)

% Define the GP model components
covfunc = @covPeriodic;   % Periodic covariance function
likfunc = @likGauss;      % Gaussian likelihood function (for regression)
meanfunc = [];            % No mean function

% Train the GP model by minimizing the negative log marginal likelihood
hyp = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

% Generate test points for predictions
xtest = linspace(min(x), max(x), 100)';  % Test points over the range of x

% Make predictions at the test points
[mu, s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xtest);

% Calculate the 95% confidence interval
lower_bound = mu - 1.96 * sqrt(s2);  % Lower bound of the 95% confidence interval
upper_bound = mu + 1.96 * sqrt(s2);  % Upper bound of the 95% confidence interval

% Plot the results
figure;
hold on;

% Plot the 95% confidence interval as a shaded area
fill([xtest; flipud(xtest)], [upper_bound; flipud(lower_bound)], [7 7 7]/8, 'EdgeColor', 'none');

% Plot the predictive mean
plot(xtest, mu, 'b-', 'LineWidth', 1.5);  % Plot the mean prediction as a blue line

% Plot the training data
plot(x, y, 'r+', 'MarkerSize', 8);  % Plot the training data as red crosses

% Add labels, title, and legend
title('Gaussian Process Regression with Periodic Covariance and 95% Predictive Error Bars');
xlabel('x');
ylabel('y');
legend('95% Prediction Interval', 'Predictive Mean', 'Training Data');
hold off;
