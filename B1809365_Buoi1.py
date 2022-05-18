import pandas as pd
#cau 1
data=pd.read_csv("baitap1.csv",delimiter=",")
#cau 2
print(data)
#cau 3
print(data.iloc[:,2:3])
#cau 4
print(data.iloc[4:9,:])
#cau 5
print(data.iloc[4:5,0:2])
#cau 6
x=data.iloc[:,1:2]
y=data.iloc[:,2:3]
print(x)
print(y)
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.title("Bai thuc hanh 1")
plt.xlabel("Tuoi")
plt.ylabel("Can Nang")
plt.autoscale(tight=True)
plt.grid()
plt.show()

#cau 7
for i in range (1,51):
	if(i%2!=0):
		print(i,end=" ")



