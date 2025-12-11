from glib import LinearReg
from glib import Lasso
from glib import Ridge
from glib import PowerReg
from glib import LogesticReg
x = [[i+4, i+1, i+2, i+1.2,i+1.1,i,i*0.9] for i in range(1,50)]  # 4 features
y = [i for i in range(1,50)]
y2 = [i**2 for i in range(1,50)]
y3 = [0  if i<15 else 1 for i in range(1,50)]

#LinearRegression model
model=LinearReg(iter=100, plot=True,track=True)
model.fit(x,y)

#Lasso model
model2=Lasso(iter=100, plot=True,track=True)
model2.fit(x,y)

#Lasso model
model3=Ridge(iter=100, plot=True,track=True)
model3.fit(x,y)

#Lasso model
model3=PowerReg(iter=130, lern=0.0000001,plot=True,track=True)
model3.fit(x,y2)

#LogesticRegression model
model4=LogesticReg(iter=290, plot=True,track=True)
model4.fit(x,y3)
