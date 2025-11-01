# Algorithm Evaluation Methods

What is the goal of Predictive Modeling ?

Create models that make good predictions on new dataset.

How to estimate the performance of a model on new dataset ?

We use statistical methods like resampling, boostraping etc.

Constraints - We don't have new dataset during predictive modeling, what we should do to evaluate the model performance ?  Then we have to make best use of our training dataset for making training as well as evaluation of model using resampling methods.

What is the goal of resampling methods ?

To make best use of our training data in order to estimate the performance of model on new unseen dataset. Then this estimated performance of model used to choose/select model or model parameters. After model selection, now we can train it onto our entire dataset and start making prediction for building ml systems.

How to Choose/Select a model for training on entire dataset ?

How to Choose a resampling methods estimation of performance of model on new dataset ?

Resampling Methods

1. Train Test Split
2. k-fold Cross Validation : LOOCV(k=1) etc
3. Stratification
