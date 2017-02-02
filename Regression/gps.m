clear all
s = RandStream('mt19937ar','Seed',0);
data = csvread('reg_train_in.csv',1,1);
test = csvread('reg_test_in.csv',1,1);
ind = find(isnan(test(:,2)) == 1);


x_full = zscore(data(:,1)); mean_1 = mean(data(:,1)); std_1 = std(data(:,1));
y_full = data(:,2);

x_test = (test(ind,1)-mean_1)/std_1;

train_ind = randperm(s,34200,1500);
x = x_full(train_ind);
y = y_full(train_ind);
xs = linspace(-1.72, 1.72, 1000)';  

meanfunc = [];                    % empty: don't use a mean function
% covfunc = @covPERiso;             % make isotropic stationary covariance periodic
meanfunc = []; hyp.mean = [];
covfunc = @covPeriodic; hyp.cov = [0 0 0];  % [log(ell), log(p), log(sf)]
likfunc = @likGauss; hyp.lik = 0;         
% [mu s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[nlZ dnlZ] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
[mu_test s2_test] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x_test);




%% Filling the gaps

corr = [2, 3, 4, 5, 6, 8, 9 ,10, 12 ,13, 14];

for i =1:400
    rand('seed', i)
    for j = corr 
        rand('seed', j)
        test(ind(i),j) = randn*s2_test(i) + mu_test(i);
    end
    test(ind(i),7) = randn*std(data(:,7)) + mean(data(:,7));
    test(ind(i),7) = randn*std(data(:,11)) + mean(data(:,11));
end

x_t = (test(:,1)-mean_1)/std_1;
y_t = test(:,2);

%% Plotting GPs
% 
% figure
% f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
% fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
% hold on; plot(x_test, mu_test, 'r'); plot(x_test, zeros(400,1), 'b+'); plot(xs, mu, 'g');scatter(x_t,y_t)
% xlabel('x') % x-axis label
% ylabel('y') % y-axis label

csvwrite('reg_test_gp.csv',test)