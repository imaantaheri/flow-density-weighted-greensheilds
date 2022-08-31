import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# input (a csv file containing speed and volume observations) Columns should be named "Volume" and "Speed"
df = pd.read_csv('29-S.csv') # data location 
df = df[df.select_dtypes(include=[np.number]).ge(0).all(1)]
df['Volume'] = df['Volume']*4 # make the volume per hour
df = df.loc[df['Speed'] != 0] # delet speed data with zero value
df['Density'] = df['Volume']/df['Speed'] # density calculation

# weight calculation for new Weighted Least Squares method (WLS) 
#(ref: Qu et al "On the fundamental diagram for freeway traffic: A novel calibration approach for single-regime models.")

df = df.sort_values('Density') # sort the data based on density values (it is neccessary for weighted method)
df = df.reset_index(drop=True)
x = pd.concat(g for _, g in df.groupby("Density") if len(g) > 1)
a = x['Density'].unique()
uniq = []
check = []
for i in range(len(a)):
    l = x.index[x['Density'] == a[i]].tolist()
    uniq.append(l)
    check.extend(l)
for j in range(len(df)):
        if j in check:
            continue
        else:
            uniq.append([j])
uniq = sorted(uniq, key=lambda x: x[0])
C = []
for n in range(len(uniq)):
    if n == 0:
        w = (df['Density'].iloc[uniq[n+1][0]] - df['Density'].iloc[uniq[n][0]])/len(uniq[n])
    if n == len(uniq)-1:
        w = (df['Density'].iloc[uniq[n][0]] - df['Density'].iloc[uniq[n-1][0]])/len(uniq[n])
    elif n > 0 & n < len(uniq)-1 :
        w = (df['Density'].iloc[uniq[n+1][0]] - df['Density'].iloc[uniq[n-1][0]])/(2*len(uniq[n]))
    C.append(w)
    
weights = []
for i in range(len(C)):
    y = [C[i]]*len(uniq[i])
    weights.extend(y)
df['Weight'] = weights # weights are calculated


def WLS (df): # a function for WLS fitting
    y = df.Speed  
    X = df.Density
    X = sm.add_constant(X)
    mod_wls = sm.WLS(y, X, weights=df['Weight'])
    res_wls = mod_wls.fit()
    a = res_wls.params.Density
    b = res_wls.params.const
    return a, b

def OLS (df): # a function for OLS fitting
    y = df.Speed  
    X = df.Density  
    X = sm.add_constant(X)  
    mod_wls = sm.OLS(y, X, weights=df['Weight'])
    res_wls = mod_wls.fit()
    a = res_wls.params.Density
    b = res_wls.params.const
    return a, b

a1, b1 = WLS (df)
a3, b3 = OLS (df)
x = np.linspace(-10,10000,1000000)
xp = np.linspace(-10,10000,1000000)
V01 = b1  # WLS free flow speed
Kj1 = V01/(-a1) #WLS jam density
V03 = b3  # OLS free flow speed
Kj3 = V03/(-a3)  # OLS jam density
y1 = [a1*i+b1 for i in x]
y3 = [a3*i+b3 for i in x]
def f(x,z,w):
        return x*z-(z/w)*(x**2)
    
# plot the results based on Greensheilds equation
fig = plt.figure(figsize=(6,4), tight_layout=True)
plt.scatter(df['Density'], df['Volume'], c= 'green', s = 30, linewidths=0.05, edgecolors = 'k')
plt.plot(xp, f(xp, V01, Kj1), color='red',linewidth=1.5, label = 'WLS')
plt.plot(xp, f(xp, V03, Kj3), color='orange',linewidth=1.5, label = 'OLS')
plt.xlim(0,np.max(df['Density']) +50)
plt.ylim(0,np.max(df['Volume']) +150)
plt.xlabel('Density (Veh/Km)')
plt.ylabel('Volume (Veh/h)')
plt.legend()

fig = plt.figure(figsize=(6,4), tight_layout=True)
plt.scatter(df['Density'], df['Speed'], c= 'green', s = 30, linewidths=0.05, edgecolors = 'k')
plt.plot(x, y1, color = 'red', linewidth=1.5, label = 'WLS')
plt.plot(x, y3, color = 'orange', label= 'OLS', linewidth=1.5)
plt.ylim(0,np.max(df['Speed']) +10)
plt.xlim(-5,np.max(df['Density']) +50)
plt.legend()
plt.xlabel('Density (Veh/Km)')
plt.ylabel('Speed (Km/h)')