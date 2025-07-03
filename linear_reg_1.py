import numpy as np
import copy, math
from data import create_data
features_df_n,right_answers_df = create_data()
features_df = features_df_n.fillna(0)
features_unscaled = features_df.values #converting to numpy array
max_values = np.max(features_unscaled, axis=0)
features = features_unscaled/max_values
right_answers_unscaled = right_answers_df.values
max_answers = np.max(right_answers_unscaled, axis=0)
right_answers = right_answers_unscaled/max_answers 


#2. Cost fxn definition: J = summation ((f_x_i - y)^2)/2m 
def cost_fxn(x,y,w,b):
    cost = 0
    m = x.shape[0]
    for i in range(m): #summation part
        f_x = np.dot(x,w)+b 
        error_term = (f_x[i] - y[i])**2
        cost+=(error_term)
    cost = cost/(2*m) # dividing by 2m
    return cost

#3. Gradient Descent: w -= alpha*(summation((f_x_i - y))*x_i)/m
def grad_fxn(x,y,w,b): # Giving the (summation)/m part of it 
    m,n = x.shape
    predictions = np.dot(x, w) + b
    errors = predictions - y 
    '''dw = np.zeros(n,)
    db = 0
    for i in range(m):
        error_term = np.dot(x[i], w) + b - y[i]
        for j in range(n):
            dw[j] += error_term*x[i,j]
        db +=error_term'''
    dw = (1/m) * np.dot(x.T, errors)
    db = (1/m) * np.sum(errors)
    return dw, db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha): #completing the function, updating the value 
    w = copy.deepcopy(w_in)
    b = b_in
    max_iterations_safety = 50000
    iteration = 0
    dw, db = gradient_function(x, y, w, b)
    while np.all(np.abs(dw) > 1.0e-5) or abs(db) > 1.0e-5: #normally better to let this run until convergence is reached and stop it using a convergence test, using a limited number of iterations here. 
        dw, db = gradient_function(x,y,w,b)
        w-= alpha*dw
        b-= alpha*db
        iteration += 1
        if iteration >= max_iterations_safety:
            print(f"WARNING: Reached safety max iterations ({max_iterations_safety}) without converging.")
            break # Exit loop as a safeguard    
    return w, b

# initialize parameters
m,n = features.shape
initial_w = np.zeros(n,)
initial_b = 0
# some gradient descent settings
alpha = 8.0e-2
# run gradient descent 
w_final, b_final = gradient_descent(features, right_answers, initial_w, initial_b,cost_fxn, grad_fxn, alpha)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

age = int(input("Enter age of the house: "))
rooms = int(input("Enter number of rooms: "))
max_age = max_values[2]
max_rooms = max_values[3]
scaled_age = age / max_age
scaled_rooms = rooms / max_rooms
x_pred = np.array([scaled_age, scaled_rooms])
w_pred = np.array([w_final[2], w_final[3]])
prediction = np.dot(x_pred,w_pred) + b_final
final_predicted_price = prediction * max_answers.item()
print("The predicted price of your house is", final_predicted_price)




## Pulling Jth column in numpy: array[:,j]
