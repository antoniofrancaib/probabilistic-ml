data = load('cw1a.mat');  
x = data.x;               
y = data.y;               

% Define the GP model components
covfunc = @covSEiso;      % Squared Exponential covariance function
likfunc = @likGauss;      % Gaussian likelihood function (for regression)
meanfunc = [];            % No mean function

initial_hyps = [
    struct('cov', [2; -1], 'lik', -1);    % First initialization
    struct('cov', [-1; 0], 'lik', 0)    % Second initialization
];

xtest = linspace(min(x), max(x), 100)';  

results = [];

for i = 1:2
    hyp = initial_hyps(i);

    hyp = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

    length_scale = exp(hyp.cov(1));
    signal_variance = exp(hyp.cov(2));
    noise_variance = exp(hyp.lik);
    nlml = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

    results = [results; i, nlml, length_scale, signal_variance, noise_variance];

    [mu, s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xtest);

    lower_bound = mu - 1.96 * sqrt(s2);  
    upper_bound = mu + 1.96 * sqrt(s2);  

    figure;
    hold on;
    
    fill([xtest; flipud(xtest)], [upper_bound; flipud(lower_bound)], ...
        [7 7 7]/8, 'EdgeColor', 'none');
    
    plot(xtest, mu, 'b-', 'LineWidth', 1.5);
    
    plot(x, y, 'r+', 'MarkerSize', 8);        

    title(sprintf('Gaussian Process Regression with Initialization %d', i));
    xlabel('x');
    ylabel('y');
    legend('95% Prediction Interval', 'Predictive Mean', 'Training Data');
    
    hold off;
end

results = sortrows(results, 2);

result_table = array2table(results, 'VariableNames', ...
    {'Initialization', 'NLML', 'Length_scale', 'Signal_variance', 'Noise_variance'});
disp(result_table);
