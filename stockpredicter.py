import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates= []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader= csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[2].split('-')[0]))
            prices.append(float(row[9]))
    return

def predict_prices(dates, prices, x):
    dates= np.reshape(dates, (len(dates), 1))
    
    svr_lin= SVR(kernel= 'linear', C=1e3)
    svr_poly= SVR(kernel= 'poly', C=1e3, degree= 2)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    
    plt.scatter(dates, prices, color='black', label= 'Data')
    plt.plot(dates, svr_lin.predict(dates), color= 'green', label='linear model')
    plt.plot(dates, svr_poly.predict(dates), color= 'blue', label='poly model')
    plt.xlabel('date')
    plt.ylabel('price')
    plt.title('S_V_R')
    plt.legend()
    plt.show()
    
    return svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('07-02-2019-TO-08-03-2019ITCALLN.csv')

predicted_price = predict_prices(dates, prices, 30)  

print(predicted_price)