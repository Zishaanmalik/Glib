from glib import Lasso

# Dataset
x = [[i+4, i+1, i+2, i+1.2, i+1.1, i, i*0.9] for i in range(1, 50)]
y = [i for i in range(1, 50)]

# Lasso Regression model
model = Lasso(iter=100, plot=True, track=True)
model.fit(x, y)
