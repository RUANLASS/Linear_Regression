import numpy as np
import copy, math
import pandas as pd
from data_3 import create_data

#FEATURE SCALING
x,y,x_train,y_train,x_cv,y_cv,x_test,y_test = create_data()

#handling null values
x = x.fillna(0)
x_train = x_train.fillna(0) 
x_cv = x_cv.fillna(0)
x_test = x_test.fillna(0)

#converting to numpy array
x = x.values
x_train = x_train.values 
x_cv = x_cv.values
x_test = x_test.values

max_x = np.max(x,axis=0)
x_train_scaled = x_train/max_x
x_cv_scaled = x_cv/max_x
x_test_scaled = x_test/max_x

y = y.values 
y_train = y_train.values
y_cv = y_cv.values
y_test = y_test.values
max_y = np.max(y,axis=0)
'''max_y_train = np.max(y_train, axis=0)
max_y_cv = np.max(y_cv, axis=0)
max_y_test = np.max(y_test, axis=0)'''
y_train_scaled = y_train/max_y
y_cv_scaled = y_cv/max_y
y_test_scaled = y_test/max_y


#2. Cost fxn definition: J = summation ((f_x_i - y)^2)/2m 
def cost_fxn(x,y,w,b,l):
    cost = 0
    m = x.shape[0]
    for i in range(m): #summation part
        f_x = np.dot(x,w)+b 
        error_term = (f_x[i] - y[i])**2
        cost+=(error_term)
    cost = cost/(2*m) # dividing by 2m
    #adding regularization
    reg_coeff = l/(2*m)
    reg_term = reg_coeff*((np.sum(w**2)))
    cost = cost+reg_term
    return cost

#3. Gradient Descent: w -= alpha*(summation((f_x_i - y))*x_i)/m
def grad_fxn(x,y,w,b): # Giving the (summation)/m part of it 
    m,n = x.shape
    predictions = np.dot(x, w) + b
    errors = predictions - y 
    dw = (1/m) * np.dot(x.T, errors)
    db = (1/m) * np.sum(errors)
    return dw, db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha,l_in): #completing the function, updating the value 
    w = copy.deepcopy(w_in)
    b = b_in
    l=l_in
    max_iterations_safety = 50000
    iteration = 0
    dw, db = gradient_function(x, y, w, b)
    while np.all(np.abs(dw) > 1.0e-5) or abs(db) > 1.0e-5: #normally better to let this run until convergence is reached and stop it using a convergence test, using a limited number of iterations here. 
        dw, db = gradient_function(x,y,w,b)
        m,n = x.shape
        reg_coeff = l/m
        w-= alpha*dw
        w-= alpha*reg_coeff*np.sum(w) 
        b-= alpha*db
        iteration += 1
        if iteration >= max_iterations_safety:
            print(f"WARNING: Reached safety max iterations ({max_iterations_safety}) without converging.")
            break # Exit loop as a safeguard    
    return w, b

# initialize parameters for training set
m1,n1 = x_train_scaled.shape
initial_w_train = np.zeros(n1,)
initial_b_train = 0
m2,n2 = x_cv_scaled.shape
initial_w_cv = np.zeros(n2,)
initial_b_cv = 0
m3,n3 = x_test_scaled.shape
initial_w_test = np.zeros(n3,)
initial_b_test = 0
# some gradient descent settings
alpha = 8.0e-2
lamda = 5.0e-5
# run gradient descent 
w_final, b_final = gradient_descent(x_train_scaled, y_train_scaled, initial_w_train, initial_b_train,cost_fxn, grad_fxn, alpha,lamda)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

#running diagnostics by checking values of training error and cross validation error. 
train_error = cost_fxn(x_train_scaled,y_train_scaled,w_final,b_final,lamda)
print("Training error is:", train_error)
cv_error = cost_fxn(x_cv_scaled,y_cv_scaled,w_final, b_final,lamda)
print("Validation error is:", cv_error)

#running predictions vs the test set to check error percentage
predict = pd.Series(np.dot(x_test_scaled, w_final) + b_final)
predict_unscaled = predict*max_y
y_test_eval = pd.Series(y_test)
error = ((abs(predict_unscaled-y_test_eval))/y_test_eval)*100
evaluation = pd.DataFrame({"Predicted Value":predict_unscaled,"Actual Value":y_test_eval,"Error":error})
print(evaluation.head)
print("Average error:",error.mean())

