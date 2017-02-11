clear all
s = RandStream('mt19937ar','Seed',0);
data = csvread('reg_train_in.csv',1,1);
test = csvread('reg_test_in.csv',1,1);
ind = find(isnan(test(:,2)) == 1);
nonnan = setdiff([1:1800],ind);

full_data = vertcat(data,test(nonnan,:));

[x_full,mean_1,std_1] = zscore(full_data(:,1));
y_full = full_data(:,2);

x_test = (test(ind,1)-mean_1)/std_1;

train_ind = randperm(s,35600,1000);
x = x_full(train_ind);
y = y_full(train_ind);
 
xs = linspace(-1.72, 1.72, 1000)';  


meanfunc = [];                    % empty: don't use a mean function
% covfunc = @covPERiso;             % make isotropic stationary covariance periodic
meanfunc = []; hyp.mean = [];
covfunc = @covPeriodic; hyp.cov = [0 0 0];  % [log(ell), log(p), log(sf)]
likfunc = @likGauss; hyp.lik = 0;         
% [mu s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

hyp2 = minimize(hyp, @gp, -80, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[nlZ dnlZ] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
[mu_test s2_test] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x_test);

 %% GP for features 7 & 10
% y7 = full_data(train_ind,7);
% y11 = full_data(train_ind,11);
% 
% meanfunc = []; hyp.mean = [];
% covfunc = @covSEiso; hyp.cov = [0 0];  % [log(ell), log(p), log(sf)]
% likfunc = @likGauss; hyp.lik = 0;         
% % [mu s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
% 
% hyp7 = minimize(hyp, @gp, -80, @infGaussLik, meanfunc, covfunc, likfunc, x, y7);
% 
% 
% 
% [mu7 s27] = gp(hyp7, @infGaussLik, meanfunc, covfunc, likfunc, x, y7, xs);
% [mu_data7 s2_data7] = gp(hyp7, @infGaussLik, meanfunc, covfunc, likfunc, x, y7, x_full);
% [mu_test7 s2_test7] = gp(hyp7, @infGaussLik, meanfunc, covfunc, likfunc, x, y7, x_test);
% 
% meanfunc = []; hyp.mean = [];
% covfunc = @covSEiso; hyp.cov = [0 0];  % [log(ell), log(p), log(sf)]
% likfunc = @likGauss; hyp.lik = 0;         
% % [mu s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
% hyp11 = minimize(hyp, @gp, -80, @infGaussLik, meanfunc, covfunc, likfunc, x, y11);
% [mu11 s211] = gp(hyp11, @infGaussLik, meanfunc, covfunc, likfunc, x, y11, xs);
% [mu_data11 s2_data11] = gp(hyp11, @infGaussLik, meanfunc, covfunc, likfunc, x, y11, x_full);
% [mu_test11 s2_test11] = gp(hyp11, @infGaussLik, meanfunc, covfunc, likfunc, x, y11, x_test);
% 

%% Filling the gaps

corr = [2, 3, 4, 5, 6, 8, 9 ,10, 12 ,13, 14];

for i =1:400
    rand('seed', i)
    for j = corr 
        rand('seed', j)
        test(ind(i),j) = randn*sqrt(s2_test(i)) + mu_test(i);
    end
    test(ind(i),7) = randn*std(full_data(:,7)) + mean(full_data(:,7));
    test(ind(i),11) = randn*std(full_data(:,11)) + mean(full_data(:,11));
end

x_t = (test(:,1)-mean_1)/std_1;
y_t = test(:,2);

%% Plotting GPs
% 
figure
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
hold on; plot(x_test, mu_test, 'g'); ; plot(xs, mu, 'k');scatter(x_t(ind),y_t(ind),'b+'),plot(x, y, 'r+')
xlabel('x') % x-axis label
ylabel('y') % y-axis label

csvwrite('reg_test_gp4.csv',test)
