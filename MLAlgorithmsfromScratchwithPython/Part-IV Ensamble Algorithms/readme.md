Ensemble Algorithms

Random Forest - A most popular and powerfull ensemble ml algorithm based on Bootstrap Aggregation(bagging) for predictive modeling.

Bootstrap Aggregation - It is a fundamental statistical technique for estimation of parametric quantities from samples. 

Ex-1 : Given a sample of X of size 100 values then to estimate the parameters of descriptive statistics like mean, median, variance and standard deviation etc can be done directly with its computational formulas. But our sample size is too small and computing our mean will have errors to it because sample may not be able to completely represent the population. So, how can improve accuracy for the estimates of parameters like means, median and standard deviation ? We use the Bootstrap  Method.


What is Bootstrap Method ?

Procedure of Bootstrap ?

1. Create the multiple subsamples of our dataset randomly with replacement means select same value multiple time but randomly.
2. Then compute the mean of each sub-sample.
3. In the end, compute the average of all of our collected means and use it as our improved accurate estimate for dataset.

This procedure can also be used to estimate the learned machine learning parameters to improved accurate estimate of it. But we needed to adapt aggregation so methods is now called Bootstrap Aggregation(Bagging) Method.

What is Bootstrap Aggregation(Bagging) Method ?

It is a powerfull ensemble statistical technique that combines the predictions from multiple ml algorithms togather to make prediction more accurate than any individual model.

It is used to reduce high veriance algorithms variance. Ex-1 Decision Tree for problems like classification and regression(CART) have high variance).

So, Bootstrap Aggregation(Bagging) is an application of the Bootstrap Procedure to a high variance ml algorithm(decision tree)
