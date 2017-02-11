function test = GP_inference(data);
    s = RandStream('mt19937ar','Seed',0);
    test_ind = randperm(s,size(data,1),1800);
    X_test = data(test_ind,:);
    
    test_nan = generate_nan(X_test,0.222);
    
    ind = find(isnan(test_nan(:,2)) == 1);
    nonnan = setdiff([1:size(test_nan,1)],ind);

    full_data = vertcat(data,test_nan(nonnan,:));

    [x_full,mean_1,std_1] = zscore(full_data(:,1));
    x_test = (test_nan(ind,1)-mean_1)/std_1;
    y_full = full_data(:,2);

    % Split data set
    train_ind = randperm(s,size(full_data,1),1500);
    x = x_full(train_ind);
    y = y_full(train_ind);

    meanfunc = []; hyp.mean = [];
    covfunc = @covPeriodic; hyp.cov = [0 0 0];  % [log(ell), log(p), log(sf)]
    likfunc = @likGauss; hyp.lik = 0;         
    hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    [nlZ dnlZ] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    [mu_test s2_test] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x_test);

     % GP for features 7 & 10
    y7 = full_data(train_ind,7);
    y11 = full_data(train_ind,11);
    
    meanfunc = []; hyp.mean = [];
    covfunc = @covSEiso; hyp.cov = [0 0];  % [log(ell), log(p), log(sf)]
    likfunc = @likGauss; hyp.lik = 0;         
    % [mu s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
    
    hyp7 = minimize(hyp, @gp, -80, @infGaussLik, meanfunc, covfunc, likfunc, x, y7);
    [mu_test7 s2_test7] = gp(hyp7, @infGaussLik, meanfunc, covfunc, likfunc, x, y7, x_test);
    
    meanfunc = []; hyp.mean = [];
    covfunc = @covSEiso; hyp.cov = [0 0];  % [log(ell), log(p), log(sf)]
    likfunc = @likGauss; hyp.lik = 0;         
    % [mu s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
    hyp11 = minimize(hyp, @gp, -80, @infGaussLik, meanfunc, covfunc, likfunc, x, y11);
    [mu_test11 s2_test11] = gp(hyp11, @infGaussLik, meanfunc, covfunc, likfunc, x, y11, x_test);
    

    %% Filling the gaps

    corr = [2, 3, 4, 5, 6, 8, 9 ,10, 12 ,13, 14];
    test = test_nan;
    mse_list = [];
    for i =1:size(ind,1)
        rand('seed', i)
        for j = corr 
            rand('seed', j)
            %test(ind(i),j) = randn*sqrt(s2_test(i)) + mu_test(i);
            test(ind(i),j) = 0 ;
        end
%         test(ind(i),7) = randn*std(full_data(:,7)) + mean(full_data(:,7));
%         test(ind(i),11) = randn*std(full_data(:,11)) + mean(full_data(:,11));
        %test(ind(i),7) = randn*sqrt(s2_test7(i)) + mu_test7(i);
        %test(ind(i),11) = randn*sqrt(s2_test11(i)) + mu_test11(i);
        test(ind(i),7) = 0;
        test(ind(i),11) = 0;
        mse_row = immse(test(ind(i),:),X_test(ind(i),:));
        mse_list = [mse_list, mse_row];
    end
    mse = mean(mse_list)
    
end
