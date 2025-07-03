import numpy as np
import copy, math
features = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
right_answers = np.array([460, 232, 178])

#2. Cost fxn definition. --> J = summation ((f_x_i - y)^2)/2m 
def cost_fxn(x,y,w,b):
    cost = 0
    m = x.shape[0]
    for i in range(m): #summation part
        f_x = np.dot(w,x.T)+b 
        error_term = (f_x[i] - y[i])**2
        cost+=(error_term)
    cost = cost/(2*m) # dividing by 2m
    return cost

#3. Gradient Descent --> w -= alpha*(summation((f_x_i - y))*x_i)/m
def grad_fxn(x,y,w,b): # Giving the summation/m part of it 
    m,n = x.shape
    dw = np.zeros(n,)
    db = 0
    for i in range(m):
        error_term = np.dot(x[i], w) + b - y[i]
        for j in range(n):
            dw[j] += error_term*x[i,j]
        db +=error_term
    dw = dw/m
    db = db/m
    return dw, db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): #completing the function, updating the value 
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters): #normally better to let this run until convergence is reached and stop it using a convergence test, using a limited number of iterations here. 
        dw, db = gradient_function(x,y,w,b)
        w-= alpha*dw
        b-= alpha*db
    return w, b

# initialize parameters
initial_w = np.zeros(4,)
initial_b = 0
# some gradient descent settings
iterations = 20000
alpha = 8.0e-7
# run gradient descent 
w_final, b_final = gradient_descent(features, right_answers, initial_w, initial_b,cost_fxn, grad_fxn, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = features.shape
for i in range(m):
    print(f"prediction: {np.dot(features[i], w_final) + b_final:0.2f}, target value: {right_answers[i]}")



## Pulling Jth column in numpy: array[:,j]