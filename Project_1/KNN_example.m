% Gaussian KNN
% Here we generate two vectors of length 100, x1 is gaussian with mean -1,
% and deviation 1, and x2 has mean 1 with deviation 1
x1 = randn(100,1) - 1;
x2 = randn(100,1)+ 1;

k = 5
% Break up the data into training and testing sets (this is a quick and
% dirty way of doing this)
x1_train = x1(1:50);
x2_train = x2(1:50);
x1_test = x1(51:100);
x2_test = x2(51:100);

% test x1_test
errors1 = 0; 
for i = 1: length(x1_test)
    % fidn the distance between the training data and the current test
    % candidate
    x1_diff = abs(x1_train - x1_test(i))
    x2_diff = abs(x2_train - x1_test(i));
    % sort the distance vectors to see the nearest points
    x1_diff = sort(x1_diff);
    x2_diff = sort(x2_diff);
    % find the classes of hte first k points
    x_diff = x1_diff - x2_diff;
    x_diff = sign(x_diff);
    x_diff = x_diff(1:k);
    % classify each point using the first k points
    if sum(x_diff > 0) 
        errors1 = errors1 + 1;
    end
end

% test x2_test
errors2 = 0; 
for i = 1: length(x2_test)
    % fidn the distance between the training data and the current test
    % candidate
    x1_diff = abs(x1_train - x2_test(i));
    x2_diff = abs(x2_train - x2_test(i));
    % sort the distance vectors to see the nearest points
    x1_diff = sort(x1_diff);
    x2_diff = sort(x2_diff);
    % find the classes of hte first k points
    x_diff = x1_diff - x2_diff;
    x_diff = sign(x_diff);
    x_diff = x_diff(1:k);
    % classify each point using the first k points
    if sum(x_diff < 0) 
        errors2 = errors2 + 1;
    end
end

%find the total number of errors by summing the test error from each class
errors1
errors2
errors = errors1 + errors2