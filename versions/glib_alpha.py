import pandas
import matplotlib.pyplot as plt
import numpy as np


#linearregression model
def LinearReg(x,y,iter=100,lern=0.00001,c=0, m=0,plot='False'):

    n = len(x)

    if plot != 'False':
        fig, (right, left) = plt.subplots(1, 2, figsize=(13, 6.5))
        right.scatter(x, y, c="blue", label="actual points")
        right.set_title("Gradient Descent trace")
        right.set_xlabel("Feature")
        right.set_ylabel("Dependent value")
        plt.tight_layout()

    for i in range(iter):
        yperr = (m * x) + c

        # partial differentation of MSE
        md = -(2 / n) * sum(x * (y - yperr))
        cd = -(2 / n) * sum(y - yperr)
        mse = np.mean((y - yperr) ** 2)
        m -= lern * md
        c -= lern * cd

        if plot != 'False':
            print(f"c= {c} m= {m} itter= {i} cost= {mse}")
            left.cla()
            right.plot(x, yperr, c="red")
            left.scatter(x, y, c="blue", label="actual points")
            left.plot(x, yperr, c="red", label="predicted line")
            left.set_title("Gradient Descent live")
            left.set_xlabel("Feature")
            plt.pause(0.01)
            left.legend()

    if plot != 'False':
        right.legend()
        plt.show()


#lassoregression model
def Lasso(x,y,iter=100,lern=0.00001,lamda = 0.02,c=0, m=0,plot='False'):

   n = len(x)

   if plot != 'False':
       fig, (right, left) = plt.subplots(1, 2, figsize=(13, 6.5))
       right.scatter(x, y, c="blue", label="actual points")
       right.set_title("Gradient Descent trace")
       right.set_xlabel("Feature")
       right.set_ylabel("Dependent value")
       plt.tight_layout()

   for i in range(iter):
       yperr=(m*x)+c
       #partial differentation of MSE
       md=-(2/n)*sum(x*(y-yperr))+ (lamda * abs(m))
       cd=-(2/n)*sum(y-yperr)
       mse =np.mean((y - yperr)**2)
       m-=lern*md
       c-=lern*cd

       if plot!='False':
           print(f"c= {c} m= {m} itter= {i} cost= {mse}")
           left.cla()
           right.plot(x, yperr, c="red")
           left.scatter(x, y, c="blue", label="actual points")
           left.plot(x, yperr, c="red", label="predicted line")
           left.set_title("Gradient Descent live")
           left.set_xlabel("Feature")
           plt.pause(0.01)
           left.legend()

   if plot != 'False':
       right.legend()
       plt.show()


#logesticregression model
def LogesticReg(x,y,iter=100,lern=0.001,lamda = 0.9,c=-1, m=0,plot='False'):

   n = len(x)

   if plot != 'False':
       fig, (right, left) = plt.subplots(1, 2, figsize=(13, 6.5))
       right.scatter(x, y, c="blue", label="actual points")
       right.set_title("Gradient Descent trace")
       right.set_xlabel("Feature")
       right.set_ylabel("Dependent value")
       plt.tight_layout()

   for i in range(iter):
       yperr=1/(1+np.exp((-m*x)-c))
       #partial differentation of MSE
       md=-(2/n)*sum(((1/yperr)**-2)*(y-yperr)*(x*np.exp((-m*x)-c)))
       cd=-(2/n)*sum(((1/yperr)**-2)*(y-yperr)*(np.exp((-m*x)-c)))
       mse =np.mean((y - yperr)**2)
       m-=lern*md
       c-=lamda*cd

       if plot != 'False':
           print(f"c= {c} m= {m} itter= {i} cost= {mse}")
           left.cla()
           right.plot(x, yperr, c="red")
           left.scatter(x, y, c="blue", label="actual points")
           left.plot(x, yperr, c="red", label="predicted line")
           left.set_title("Gradient Descent live")
           left.set_xlabel("Feature")
           plt.pause(0.01)
           left.legend()

   if plot != 'False':
       right.legend()
       plt.show()


#powerregression model
def PowerReg(x,y,iter=100,lern=0.0000001,c=0, m=0,p=1,plot='False'):

    n = len(x)

    if plot != 'False':
        fig, (right, left) = plt.subplots(1, 2, figsize=(13, 6.5))
        right.scatter(x, y, c="blue", label="actual points")
        right.set_title("Gradient Descent trace")
        right.set_xlabel("Feature")
        right.set_ylabel("Dependent value")
        plt.tight_layout()


    for i in range(iter):
       yperr = (m * (x ** p)) + c
       # partial differentation of MSE
       mse = np.mean((y - yperr) ** 2)
       md = -(2 / n) * sum(x * (y - yperr))
       cd = -(2 / n) * sum(y - yperr)
       pdd = -(2 / n) * sum(m * p * x * (y - yperr))
       m -= lern * md
       c -= lern * cd
       p -= lern * pdd

       if plot != 'False':
           print(f"c= {c} m= {m} p= {p} itter= {i} cost= {mse}")
           left.cla()
           right.plot(x, yperr, c="red")
           left.scatter(x, y, c="blue", label="actual points")
           left.plot(x, yperr, c="red", label="predicted line")
           left.set_title("Gradient Descent live")
           left.set_xlabel("Feature")
           plt.pause(0.01)
           left.legend()

    if plot != 'False':
        right.legend()
        plt.show()


#ridgeregression model
def Ridge(x,y,iter=100,lern=0.00001,lamda = 0.02,c=0, m=0,plot='False'):

   n = len(x)

   if plot != 'False':
       fig, (right, left) = plt.subplots(1, 2, figsize=(13, 6.5))
       right.scatter(x, y, c="blue", label="actual points")
       right.set_title("Gradient Descent trace")
       right.set_xlabel("Feature")
       right.set_ylabel("Dependent value")
       plt.tight_layout()

   for i in range(iter):
       yperr=(m*x)+c
       #partial differentation of MSE
       md=-(2/n)*sum(x*(y-yperr))+ (lamda * m * m)
       cd=-(2/n)*sum(y-yperr)
       mse =np.mean((y - yperr)**2)
       m-=lern*md
       c-=lern*cd

       if plot!='False':
           print(f"c= {c} m= {m} itter= {i} cost= {mse}")
           left.cla()
           right.plot(x, yperr, c="red")
           left.scatter(x, y, c="blue", label="actual points")
           left.plot(x, yperr, c="red", label="predicted line")
           left.set_title("Gradient Descent live")
           left.set_xlabel("Feature")
           plt.pause(0.01)
           left.legend()

   if plot != 'False':
       right.legend()
       plt.show()