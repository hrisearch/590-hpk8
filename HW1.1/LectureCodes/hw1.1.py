import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from   scipy.optimize import minimize


#FUNCTION TO OPTIMZE
def f(mc, y, x):
	x = np.array(x)
	y = np.array(y)
	#print(type(y))
	out=(y-(float(mc[0])*x+float(mc[1])))
	out = np.linalg.norm(out)
	# out=(x+10*np.sin(x))**2.0
	return out

#FUNCTION TO OPTIMZE
def f2(mc, y, x):
	a = mc[0]
	b = mc[1]
	c = mc[2]
	d = mc[3]
	x = np.array(x)
	y = np.array(y)
	#print(type(y))
	out=(y-(a/(1 + (2.73)**(-(x-c)/d)) + b))
	out = np.linalg.norm(out)
	# out=(x+10*np.sin(x))**2.0
	return out

#FUNCTION TO OPTIMZE
def f3(mc, y, x):
	a = mc[0]
	b = mc[1]
	c = mc[2]
	d = mc[3]
	x = np.array(x)
	y = np.array(y)
	#print(type(y))
	out=(y-(a/(1 + (2.73)**(-(x-c)/d)) + b))
	out = np.linalg.norm(out)
	# out=(x+10*np.sin(x))**2.0
	return out

class Data:
	def read(self):
		# json file



		# Opening JSON file
		f = open('weight.json',)

		# returns JSON object as
		# a dictionary
		data = json.load(f)

		#for i in data['x']:
		#	print(i)
		age = data['x']
		weight = data['y']
		isadult = data['is_adult']
		#print(isadult)
		f.close()
		return age, weight, isadult

dat = Data()
age, weight, isadult = dat.read()


plt.figure() #INITIALIZE FIGURE 
FS=18   #FONT SIZE
plt.xlabel('age', fontsize=FS)
plt.ylabel('weight', fontsize=FS)
plt.plot(age,weight,'ro')
#plt.plot(age,weight,'-')

xo = age[0]
res = minimize(f, x0 = [1, 1], args=(weight, age,) , method='Nelder-Mead', tol=1e-5)
popt=res.x
print("OPTIMAL PARAM:",popt)

xp = np.linspace(0,100,100)
yp = popt[0]*xp + popt[1] 

plt.plot(xp, yp, '-b')


plt.show()



plt.figure() #INITIALIZE FIGURE 
FS=18   #FONT SIZE
plt.xlabel('age', fontsize=FS)
plt.ylabel('weight', fontsize=FS)
plt.plot(age,weight,'ro')
#plt.plot(age,weight,'-')

xo = age[0]
res = minimize(f2, x0 = [0.1, 0.1, 0.1, 0.1], args=(weight, age,) , method='Nelder-Mead', tol=1e-8)
popt=res.x
print("OPTIMAL PARAM:",popt)

xp = np.linspace(0,100,100)
#yp = popt[0]*xp + popt[1] 
yp=((popt[0]/(1 + (2.73)**(-(xp-popt[2])/popt[3])) + popt[1]))

plt.plot(xp, yp, '-b')


plt.show()


plt.figure() #INITIALIZE FIGURE 
FS=18   #FONT SIZE
plt.xlabel('weight', fontsize=FS)
plt.ylabel('is_adult', fontsize=FS)
plt.plot(weight,isadult,'ro')
#plt.plot(age,weight,'-')

res = minimize(f3, x0 = [0.9, 0.9, 0.9, 0.9], args=(isadult, weight,) , method='Nelder-Mead', tol=1e-11)
popt=res.x
print("OPTIMAL PARAM:",popt)

xp = np.linspace(0,300,300)
#yp = popt[0]*xp + popt[1] 
#popt = [0.4, 0.120, 0.20, 0.04]
yp=((popt[0]/(1 + (2.73)**(-(xp-popt[2])/popt[3])) + popt[1]))

plt.plot(xp, yp, '-b')


plt.show()


