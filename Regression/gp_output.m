clear all
s = RandStream('mt19937ar','Seed',0);
data = csvread('reg_train_in.csv',1,1);
labels = csvread('reg_train_out.csv',1,1);
test = csvread('reg_test_gp.csv');



x_full = zscore(data(:,1)); mean_2 = mean(data(:,2)); std_2 = std(data(:,2));
y_full = labels;

x_test = (test(:,2)-mean_2)/std_2;

train_ind = randperm(s,34200,1500);
x = x_full(train_ind);
y = y_full(train_ind);
xs = linspace(-1.72, 1.72, 1000)';  

meanfunc = [];                    % empty: don't use a mean function
% covfunc =  @covSEiso;             % make isotropic stationary covariance periodic
meanfunc = []; hyp.mean = [];
covfunc = @covSEiso; hyp.cov = [0 0];  % [log(ell), log(p), log(sf)]
likfunc = @likGauss; hyp.lik = 0;         
% [mu s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[nlZ dnlZ] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
[mu_data s2_data] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x_full);
[mu_test s2_test] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x_test);

%% Filling the gaps

corr = [2, 3, 4, 5, 6, 8, 9 ,10, 12 ,13, 14];
y_test_pred = zeros(1800,1);
y_pred = zeros(34200,1);

for i = 1:34200
        y_pred(i) = randn*s2_data(i) + mu_data(i);
end
for i = 1:1800
        y_test_pred(i) = randn*s2_test(i) + mu_test(i);
end

% Plotting GPs

figure
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
hold on; scatter(x_test, mu_test, 'k'); plot(x_test, y_test_pred, 'r+'); plot(xs, mu, 'r');plot(x, y, 'go')
%scatter(x_t,y_t)
xlabel('x') % x-axis label
ylabel('y') % y-axis label


csvwrite('reg_pred_gp.csv',y_test_pred)




