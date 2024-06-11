# flow-density-weighted-greensheilds
Fits Greenshields traffic model to real traffic data with a new weighted algorithm


When fitting the Greansheilds relationship to real-data, one may suffer from the problem of insufficient data samples in the congested regime. 
This code is implemented based on the approach proposed by [Qu, X., Wang, S., & Zhang, J. (2015)](https://www.sciencedirect.com/science/article/abs/pii/S0191261515000041.) to address this problem. 

An illustration of the outputs is given in the below (WLS is the fitted cureve with the Weighted Least Square and OLS is fitted by Ordinary Least Square algorithm):

![image](https://user-images.githubusercontent.com/112522995/187576849-24a5e19e-2c2c-4aa3-8090-252cd16c85f7.png)
