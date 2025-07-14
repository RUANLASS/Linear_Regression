# Learning Process

Dataset used: California Housing Prices dataset from Kaggle. 

Different files:
1. basic.py => simple model using simple 3x4 and 1x3 matrix with given values, using linear regression model, with squared error cost function and gradient descent optimization algorithm.

2. data_1.py + linear_reg_1.py =>
   - model using an actual dataset, with linear regression model, squared error cost, gradient descent optimization
   - data preprocessing -> implemented data validation techniques to identify and resolve potentially error-causing or inconsistent entries, enhancing the dataset's reliability 
   - used feature scaling from scratch for optimal convergence of gradient descent
   - carried out data cleaning operations to fix errors in the dataset
   - prediction accuracy was, however, **extremely low**
  
3. data_2.py + linear_reg_2.py =>
   - introduced term for *regularization*
   - split the dataset into *training set, cross validation set, test set*.
   - continued to use a *linear function*  
   - used values of error on training set and cross validation set to evaluate whether the model had *high bias or variance*.
   - model gave mean error of around **29-30%** on test set.
  
4. data_3.py + linear_reg_3.py =>
   - since error/cost on training set and cross validation set seemed to be close to each other (approx. 0.011 and 0.014 respectively), implemented diagnostics to fix *high bias*, such as:
      - added *polynomial features* to the model
      - *hyperparameter tuning* on alpha (learning rate) and lambda (regularization strength)
   - final mean error of around **26-27%** on test set. 
