#--------------------------------
# UNIVARIABLE REGRESSION EXAMPLE
#    -USING SciPy FOR OPTIMIZATION
#--------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize


#------------------------
#CODE PARAMETERS
#------------------------

#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"


OPT_ALGO='BFGS'	#HYPER-PARAM

#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
model_type="logistic"; NFIT=4; X_KEYS=['x']; Y_KEYS=['y']
# model_type="linear";   NFIT=2; X_KEYS=['x']; Y_KEYS=['y']
# model_type="logistic"; NFIT=4; X_KEYS=['y']; Y_KEYS=['is_adult']

#RANDOM INITIAL GUESS FOR FITTING PARAMETERS
p=np.random.uniform(0.5,1.,size=NFIT)

#SAVE HISTORY FOR PLOTTING AT THE END
iteration=0; iterations=[]; loss_train=[];  loss_val=[]

#------------------------
#DATA CLASS
#------------------------

class DataClass:

    #INITIALIZE
	def __init__(self,FILE_NAME):

		if(FILE_TYPE=="json"):

			#READ FILE
			with open(FILE_NAME) as f:
				self.input = json.load(f)  #read into dictionary

			#CONVERT DICTIONARY INPUT AND OUTPUT MATRICES #SIMILAR TO PANDAS DF   
			X=[]; Y=[]
			for key in self.input.keys():
				if(key in X_KEYS): X.append(self.input[key])
				if(key in Y_KEYS): Y.append(self.input[key])

			#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
			self.X=np.transpose(np.array(X))
			self.Y=np.transpose(np.array(Y))
			self.been_partitioned=False

			#INITIALIZE FOR LATER
			self.YPRED_T=1; self.YPRED_V=1

			#EXTRACT AGE<18
			if(model_type=="linear"):
				self.Y=self.Y[self.X[:]<18]; 
				self.X=self.X[self.X[:]<18]; 

			#TAKE MEAN AND STD DOWN COLUMNS (I.E DOWN SAMPLE DIMENSION)
			self.XMEAN=np.mean(self.X,axis=0); self.XSTD=np.std(self.X,axis=0) 
			self.YMEAN=np.mean(self.Y,axis=0); self.YSTD=np.std(self.Y,axis=0) 
		else:
			raise ValueError("REQUESTED FILE-FORMAT NOT CODED"); 

	def report(self):
		print("--------DATA REPORT--------")
		print("X shape:",self.X.shape)
		print("X means:",self.XMEAN)
		print("X stds:" ,self.XSTD)
		print("Y shape:",self.Y.shape)
		print("Y means:",self.YMEAN)
		print("Y stds:" ,self.YSTD)

	def partition(self,f_train=0.8, f_val=0.15,f_test=0.05):
		#TRAINING: 	 DATA THE OPTIMIZER "SEES"
		#VALIDATION: NOT TRAINED ON BUT MONITORED DURING TRAINING
		#TEST:		 NOT MONITORED DURING TRAINING (ONLY USED AT VERY END)


		if(f_train+f_val+f_test != 1.0):
			raise ValueError("f_train+f_val+f_test MUST EQUAL 1");

		#PARTITION DATA
		rand_indices = np.random.permutation(self.X.shape[0])
		CUT1=int(f_train*self.X.shape[0]); 
		CUT2=int((f_train+f_val)*self.X.shape[0]); 
		self.train_idx, self.val_idx, self.test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]
		self.been_partitioned=True

	def model(self,x,p):
		if(model_type=="linear"):   return  p[0]*x+p[1]  
		if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.0001))))

	def predict(self,p):
		self.YPRED_T=self.model(self.X[self.train_idx],p)
		self.YPRED_V=self.model(self.X[self.val_idx],p)
		self.YPRED_TEST=self.model(self.X[self.test_idx],p)

	def normalize(self):
		self.X=(self.X-self.XMEAN)/self.XSTD 
		self.Y=(self.Y-self.YMEAN)/self.YSTD  

	def un_normalize(self):
		self.X=self.XSTD*self.X+self.XMEAN 
		self.Y=self.YSTD*self.Y+self.YMEAN 
		self.YPRED_V=self.YSTD*self.YPRED_V+self.YMEAN 
		self.YPRED_T=self.YSTD*self.YPRED_T+self.YMEAN 
		self.YPRED_TEST=self.YSTD*self.YPRED_TEST+self.YMEAN 

	#------------------------
	#DEFINE LOSS FUNCTION
	#------------------------
	def loss(self, p, method, rand_indices1=None, rand_indices2=None):
		global iteration,iterations,loss_train,loss_val

		#MAKE PREDICTIONS FOR GIVEN PARAM
		self.predict(p)

		#LOSS
		if method == 'batch':
			training_loss=(np.mean((self.YPRED_T-self.Y[self.train_idx])**2.0))  #MSE
			validation_loss=(np.mean((self.YPRED_V-self.Y[self.val_idx])**2.0))  #MSE

		elif method == 'mini-batch':
			mini_ids = self.train_idx[rand_indices1[:int(0.5*len(self.train_idx))]]
			training_loss=(np.mean((self.model(self.X[mini_ids], p) - self.Y[mini_ids])**2.0))  #MSE

			mini_ids = self.val_idx[rand_indices2[:int(1*len(self.val_idx))]]
			validation_loss=(np.mean((self.model(self.X[mini_ids], p) - self.Y[mini_ids])**2.0))  #MSE
	
		elif method == 'stochastic':
			mini_ids = self.train_idx[rand_indices1[0]]
			training_loss=(np.mean((self.model(self.X[mini_ids], p) - self.Y[mini_ids])**2.0))  #MSE

			mini_ids = self.val_idx[rand_indices2[:int(1*len(self.val_idx))]]
			validation_loss=(np.mean((self.model(self.X[mini_ids], p) - self.Y[mini_ids])**2.0))  #MSE

		loss_train.append(training_loss); loss_val.append(validation_loss)
		iterations.append(iteration)

		iteration+=1

		return training_loss

	def get_grad(self, q, r, method, loss_func, h):
		if method == 'mini-batch' or method == 'stochastic':
			rand_indices1 = np.random.permutation(len(self.train_idx))
			rand_indices2 = np.random.permutation(len(self.val_idx))
			return (loss_func([q[0], q[1], q[2], q[3]], method, rand_indices1, rand_indices2) - loss_func([r[0], r[1], r[2], r[3]], method,  rand_indices1, rand_indices2))/2/h
		elif method == 'batch':
			return (loss_func([q[0], q[1], q[2], q[3]], method) - loss_func([r[0], r[1], r[2], r[3]], method))/2/h

	def optimizer(self, loss_func, p, algo='GD', LR=0.001, method='batch'):
		gradient = np.array([1.0, 0.0, 0.0, 0.0])
		count = 0
		pold = 0
		while np.linalg.norm(gradient) > 1e-3 and count < 10000:
			count += 1
			h = 1e-12

			
			g0 = self.get_grad([p[0]+h, p[1], p[2], p[3]], [p[0]-h, p[1], p[2], p[3]], method, loss_func, h)
			g1 = self.get_grad([p[0], p[1]+h, p[2], p[3]], [p[0], p[1]-h, p[2], p[3]], method, loss_func, h)
			g2 = self.get_grad([p[0], p[1], p[2]+h, p[3]], [p[0], p[1], p[2]-h, p[3]], method, loss_func, h)
			g3 = self.get_grad([p[0], p[1], p[2], p[3]+h], [p[0], p[1], p[2], p[3]-h], method, loss_func, h)
			gradient = np.array([g0, g1, g2, g3])
			#np.add(gradient, np.array([g0, g1, g2, g3]))
			temp = p
			alpha = 0.001 * 0.9**(count)
			if algo == 'GD':
				p = p - LR*gradient 
			else: #GDM momentum
				p = p - LR*gradient -alpha*pold
			pold = temp
	#		print(p)
		return p

	def fit(self):
		#TRAIN MODEL USING SCIPY MINIMIZ 
		# res = minimize(self.loss, p, method=OPT_ALGO, tol=1e-15)
		# popt=res.x; print("OPTIMAL PARAM:",popt)
		res = self.optimizer(self.loss, p, 'GDM', 0.05, 'batch')# batch, mini-batch, stochastic
		popt=res; print("OPTIMAL PARAM:",popt)

		#PLOT TRAINING AND VALIDATION LOSS AT END
		if(IPLOT):
			fig, ax = plt.subplots()
			ax.plot(iterations, loss_train, 'o', label='Training loss')
			ax.plot(iterations, loss_val, 'o', label='Validation loss')
			plt.xlabel('optimizer iterations', fontsize=18)
			plt.ylabel('loss', fontsize=18)
			plt.legend()
			plt.show()

	#FUNCTION PLOTS
	def plot_1(self,xla='x',yla='y'):
		if(IPLOT):
			fig, ax = plt.subplots()
			ax.plot(self.X[self.train_idx]    , self.Y[self.train_idx],'o', label='Training') 
			ax.plot(self.X[self.val_idx]      , self.Y[self.val_idx],'x', label='Validation') 
			ax.plot(self.X[self.test_idx]     , self.Y[self.test_idx],'*', label='Test') 
			ax.plot(self.X[self.train_idx]    , self.YPRED_T,'.', label='Model') 
			plt.xlabel(xla, fontsize=18);	plt.ylabel(yla, fontsize=18); 	plt.legend()
			plt.show()

	#PARITY PLOT
	def plot_2(self,xla='y_data',yla='y_predict'):
		if(IPLOT):
			fig, ax = plt.subplots()
			ax.plot(self.Y[self.train_idx]  , self.YPRED_T,'*', label='Training') 
			ax.plot(self.Y[self.val_idx]    , self.YPRED_V,'*', label='Validation') 
			ax.plot(self.Y[self.test_idx]    , self.YPRED_TEST,'*', label='Test') 
			plt.xlabel(xla, fontsize=18);	plt.ylabel(yla, fontsize=18); 	plt.legend()
			plt.show()




#------------------------
#MAIN 
#------------------------
D=DataClass(INPUT_FILE)		#INITIALIZE DATA OBJECT 
D.report()					#BASIC DATA PRESCREENING

D.partition()				#SPLIT DATA
D.normalize()				#NORMALIZE
D.fit()
D.plot_1()					#PLOT DATA
D.plot_2()					#PLOT DATA

D.un_normalize()			#NORMALIZE
D.plot_1()					#PLOT DATA
D.plot_2()					#PLOT DATA

# I HAVE WORKED THROUGH THIS EXAMPLE AND UNDERSTAND EVERYTHING THAT IT IS DOING