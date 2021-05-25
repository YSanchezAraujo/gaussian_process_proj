import numpy as np
import GPy
import scipy
from scipy import stats
import pandas as pd

# pick one of the saved CSV files
data_path = "/Users/yoelsanchezaraujo/Desktop/grid_data_pillow/csvfiles/Barbara_0511_5+_4+_3+_2+_1_T1C1_pos.csv"
data = pd.read_csv(data_path)
X = data.iloc[:, 0:2].values
y = data.iloc[:, 2].values

# also here the 1 might need to change to 2?
kernel = GPy.kern.RBF(1, variance=1.0, lengthscale=1.0) # Need to check that the variance is the same
# same as likelihood
poisson_likelihood = GPy.likelihoods.Poisson(gp_link=GPy.likelihoods.link_functions.Log())
# should be the same as in the julia code, roughly
gp_laplace = GPy.inference.latent_function_inference.Laplace()

model = GPy.core.GP(X=X, Y=y[:, None], likelihood=poisson_likelihood, 
	                inference_method=gp_laplace, kernel=kernel)

# run the model
model.optimize()

