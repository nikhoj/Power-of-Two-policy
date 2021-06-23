import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = 'POT Project Data.xls'
df = pd.read_excel(file, sheet_name = 'Data')
data = np.array(df)

def tau_1(k,h0, h,d):
    tau = np.sqrt(k / ((.5 * h0 * d) + (.5* h* d)))
    
    return tau

def tau_2(k,h,d):
    tau = np.sqrt(k / (.5 * h * d))
    
    return tau

def setup_cost(x,y, alpha, C):
    x = x * 100
    y = y * 100
    dist = np.sqrt((4.7 - x)**2 + (3.9 - y)**2)
    setup_cost = alpha * dist + C
    
    return setup_cost

def system_cost(T0,tau,data):
    
    K0 = data[0,-1]
    total = K0/(T0/365)
    
    H0 = 1
    for i in range (24):
        K = data[i+1,-1]
        D = data[i+1,-2]
        H = data[i+1, 3]
    
        if T0 < tau[i,0]:
            T = tau[i,0]/365
            cost = 2* np.sqrt(K * (.5 * D * H)) + T * (.5 * H0 * D)

            
        elif T0 > tau[i,1]:
            
            cost = 2 * np.sqrt(K * (.5* H* D + .5* H0 * D))
            
        elif T0 >= tau[i,0] and T0 <= tau[i,1]:
            
            cost = (K / (T0/365)) + (T0/365 * (.5* H* D + .5* H0* D))
         
        total += cost
    
    
    
            
    return total



def find_best_T0(baseT, tau, data):
    T0 = np.zeros((14,2))
    
    i = -1
    for k in range(6,20):
        i = i + 1
        T0[i,0] = 2**k * baseT
        T0[i,1] = system_cost(T0 = T0[i,0], tau = tau, data = data)
        T0[i,1] = (T0[i,1]).astype(int)
        
        
    return T0

def T0_star_base(data, tau, best_T0):
    #best_T0 = best_T0/365
    K = data[0,-1]
    G = 0
    H0 = 1
    for i in range(24):
        k = data[i+1, -1]
        D = data[i+1,-2]
        H = data[i+1, 3]
        if best_T0 >= tau[i,0] and best_T0 <= tau[i,1]:
            K += k
            G += .5 * H * D + .5 * H0 * D
        elif best_T0 > tau[i,1]:
            G +=  .5 * H0 * D
        
    T0_star_base = np.sqrt(K / G)
    
    return T0_star_base

def find_T_star(T0_star_base, tau, data):
    #initialization
    time_cost_matrix = np.zeros((3000,2))
    TL = T0_star_base * 365
    min_cost = system_cost(TL,tau,data)
    time_cost_matrix[0,0] = TL
    time_cost_matrix[0,1] = min_cost
    
    for i in range(2999):
        T0 = (i+1)
        time_cost_matrix[i+1,0] = T0
        cost = system_cost(T0,tau,data)
        time_cost_matrix[i+1,1] = cost
        
        
    
    
    return time_cost_matrix

retailer_k = np.array(setup_cost(data[1:,1], data[1:,2], .32, 150)).reshape((24,1))
wholeseller_k = np.array([400]).reshape((1,1))
k = np.vstack((wholeseller_k, retailer_k))
data = np.hstack((data,k))

tau1 = tau_1(data[1:,5], 1, data[1:,3], data[1:,4]).reshape((24,1))
tau2 = tau_2(data[1:,5], data[1:,3], data[1:,4]).reshape((24,1))
tau = np.hstack((tau1,tau2))
tau *= 365
tau = tau.astype(int)

sorted_tau = np.sort(tau.reshape((1,48)))

#day


T0 = find_best_T0(1, tau, data)
best_T0 = T0[np.argmin(T0[:,1]),0]
best_cost = T0[np.argmin(T0[:,1]),1]
T_star = T0_star_base(data, tau, best_T0)
print("Best system cycle value will be " + str(best_T0/365) + " years.")
print("Base T value will be " + str(T_star) + " year(s).")
print("Total average cost with this distribution strategy is " + "$" + str(best_cost) + "/year.")


plt.plot(T0[:,0], T0[:,1])
plt.show()

plt.scatter(data[0,1],data[0,2], c = 'r' )
plt.scatter(data[1:,1],data[1:,2], c = 'b', s=data[1:,3])
plt.show()

