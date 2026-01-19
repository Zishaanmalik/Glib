from glib import PowerReg

# Dataset
x = [[i+4, i+1, i+2, i+1.2, i+1.1, i, i*0.9] for i in range(1, 50)]
y2 = [i**2 for i in range(1, 50)]

# Power Regression model
model = PowerReg(iter=130, lern=0.0000001, plot=True, track=True)
model.fit(x, y2)
