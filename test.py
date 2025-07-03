import pandas as pd
import numpy as np
fil = pd.read_csv("housing.csv")
column_headers = fil.columns.tolist()
m = fil.shape[0]
'''d = {}
for i in range(l):
    d[column_headers.index(i)] = np.array(fil[i].tolist(), dtype=float)
xt = np.array(list(d.values()), dtype=float)
'''
x1 = fil["total_rooms"]
x2 = fil["total_bedrooms"]
x3=fil["population"]
x=[x1,x2,x3]
n = len(x)
w0 = [i for i in range (1,4)]
w = np.array(w0, dtype=float)
b = 1
f_x = np.dot(w,x) + b
y = fil["median_house_value"]
alpha = 0.5

#cost_function(f_x, y)
def cost_function(model, right_answer):
    error_term=0
    for i in range(m):
        error_term += ((model[i]-right_answer[i])**2)
    J = error_term/(2*m)
    return J
J = cost_function(f_x, y)

def gradient_descent(model, right_answer, cost_fxn, num_iterations):
    for i in range(0, num_iterations):
        w_derivative={"w1":0, "w2":0, "w3":0}
        w_keys=list(w_derivative.keys())
        b_derivative=0
        for i in range(0,n):
            for j in range(0, m):
                w_derivative[w_keys[i]] += ((model[j]-right_answer[j])*x[i])
                b_derivative += (model[j]-right_answer[j])
        w1_diff = (alpha*w_derivative["w1"])/m
        w2_diff=(alpha*w_derivative["w2"])/m
        w3_diff=(alpha*w_derivative["w3"])/m
        b_diff = (alpha*b_derivative)/m
        if abs(w1_diff)<0.05 and abs(b_diff)<0.05 and abs(w2_diff)<0.05 and abs(w3_diff)<0.05:
            break
        else:
            w_diff= np.array([w1_diff, w2_diff, w3_diff])
            w-=w_diff
            b-=b_diff
            model= np.dot(w,x) + b 
            cost_fxn= cost_function(model, right_answer)
    return model, cost_fxn
final_model, min_cost_fxn = gradient_descent(f_x, y, J, 900)
print(final_model, min_cost_fxn)



            

