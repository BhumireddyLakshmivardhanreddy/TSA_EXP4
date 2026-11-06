# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 

### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv('/content/AirPassengers.csv')

# Declare required variables and set figure size, and visualise the data
N = 1000
plt.rcParams['figure.figsize'] = [12, 6]

X = data['#Passengers']
plt.plot(X)
plt.title('Original Data')
plt.show()

plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()

# Fitting the ARMA(1,1) model and deriving parameters
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1) Process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()

# Fitting the ARMA(2,2) model and deriving parameters
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

# Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```
## OUTPUT:
SIMULATED ARMA(1,1) PROCESS:
<img width="1052" height="523" alt="image" src="https://github.com/user-attachments/assets/597a3e85-0916-4313-8906-fe3f930ad18d" />
Partial Autocorrelation
<img width="1547" height="587" alt="image" src="https://github.com/user-attachments/assets/496e3b11-c9ff-49f4-989e-7f2d0b1950a9" />
Autocorrelation
<img width="1052" height="527" alt="image" src="https://github.com/user-attachments/assets/ed8134cd-ff2c-45a1-8fb7-ef3b4381ba9d" />
SIMULATED ARMA(2,2) PROCESS:
<img width="1046" height="516" alt="image" src="https://github.com/user-attachments/assets/1d43a54b-7756-4b77-87a4-65bba77a5253" />
Partial Autocorrelation
<img width="1050" height="523" alt="image" src="https://github.com/user-attachments/assets/31e43777-9fc4-40e5-8470-34d27a04e724" />
Autocorrelation
<img width="1033" height="517" alt="image" src="https://github.com/user-attachments/assets/380fd58e-6fc4-43b0-9972-4f8517f69a57" />

## RESULT:
Thus, a python program is created to fir ARMA Model successfully.
