data = load('cw1a.mat');  
x = data.x;               
y = data.y;               

% Set initial hyperparameters for the Gaussian Process
hyp.cov = [-1; 0];        % Log length-scale and log signal variance (for covSEiso)
hyp.lik = 0;              % Log noise variance (for likGauss)

% Define the GP model components
covfunc = @covSEiso;      % Squared Exponential covariance function
likfunc = @likGauss;      % Gaussian likelihood function (for regression)
meanfunc = [];            % No mean function

% Train the GP model by minimizing the negative log marginal likelihood
hyp = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

xtest = linspace(min(x), max(x), 100)';  

[mu, s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xtest);

lower_bound = mu - 1.96 * sqrt(s2);  
upper_bound = mu + 1.96 * sqrt(s2);  

figure;
hold on;
fill([xtest; flipud(xtest)], [upper_bound; flipud(lower_bound)], [7 7 7]/8, 'EdgeColor', 'none');

plot(xtest, mu, 'b-', 'LineWidth', 1.5);  
plot(x, y, 'r+', 'MarkerSize', 8);        

title('Gaussian Process Regression with 95% Predictive Error Bars');
xlabel('x');
ylabel('y');
legend('95% Prediction Interval', 'Predictive Mean', 'Training Data');
hold off;
