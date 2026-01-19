from glib import LogesticReg

# Dataset
x = [[i+4, i+1, i+2, i+1.2, i+1.1, i, i*0.9] for i in range(1, 50)]
y3 = [0 if i < 15 else 1 for i in range(1, 50)]

# Logistic Regression model
model = LogesticReg(iter=290, plot=True, track=True)
model.fit(x, y3)
