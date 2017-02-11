function test_nan = generate_nan(X_test,per)
    s = RandStream('mt19937ar','Seed',0);
    N = size(X_test,1);
    nan_ind = randperm(s,N, floor(N*per));
    test_nan = X_test;
    for i = nan_ind
        for j = 2:14
        test_nan(i,j) = NaN;
    end
end