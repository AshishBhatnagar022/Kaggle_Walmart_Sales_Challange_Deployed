from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import datetime as dt
import time

import numpy as np
from sklearn.externals import joblib
app = Flask(__name__)
# model=pickle.load(open('walsal.pkl','rb'))
a=open('walmart.pkl','rb')
model=joblib.load(a)
	# model=joblib.load(a)
# model=pickle.load(open('model1.pkl','rb'))
# a=open('walmod.pkl','rb')
# model=joblib.load(a)

@app.route('/')
def home():
    return render_template('indwal.html')

@app.route('/predict',methods=['GET','POST'])
def predict():

	wf=pd.read_csv('features.csv')
	ws=pd.read_csv('stores.csv')
	wt=pd.read_csv('waltrain.csv')
# wt=wt.merge(ws,how='left').merge(wf,how='left')
	train=wt.merge(ws,how='left').merge(wf,how='left')
	# import time

	# col=train['Date']
	# for c in col:
	train['Date']=pd.to_datetime(train['Date'])
  		
	train['Year'] = train["Date"].dt.year
	train['Month'] = train["Date"].dt.month
	train['Day'] = train["Date"].dt.day

	# train['Date'].fillna(0,inplace=True)

	# train['Date'] = pd.to_datetime(train['Date'])
	# cols = ['Date']
	# for col in cols:
    
	# train['Year'] = train["Date"].dt.year
	# train['Month'] = train["Date"].dt.month
	# train['Day'] = train["Date"].dt.day
	bins=np.linspace(min(train['Store']),max(train['Store']),6)
	g={'0-10','10-20','20-30','30-40','40-50'}
	train['Store_binned']=pd.cut(train['Store'],bins,labels=g,include_lowest=True)
	train['Store_binned'] = train['Store_binned'].map({"0-10":0, "10-20":1,"20-30":2, "30-40":3,"40-50":4})
	num_Store_one_hot = pd.get_dummies(train['Store_binned'],prefix='Store')
	bins=np.linspace(min(train['Dept']),max(train['Dept']),6)
	g={'0-20','20-40','40-60','60-80','80-100'}
	train['Dept_binned']=pd.cut(train['Dept'],bins,labels=g,include_lowest=True)
	train['Dept_binned'] = train['Dept_binned'].map({"0-20":0, "20-40":1,"40-60":2, "60-80":3,"80-100":4})
	num_Dept_one_hot = pd.get_dummies(train['Dept_binned'], 
                                     	prefix='Dept')
	train['Size'].loc[(train['Size'] >=0) & (train['Size'] <75000)] = 1
	train['Size'].loc[(train['Size'] >=75000) & (train['Size'] <125000)] = 2
	train['Size'].loc[(train['Size'] >=125000) & (train['Size'] <175000)] = 3
	train['Size'].loc[(train['Size'] >=175000) & (train['Size'] <225000)] = 4

	num_Size_one_hot = pd.get_dummies(train['Size'], 
                                     	prefix='Size')
	bins=np.linspace(min(train['Temperature']),max(train['Temperature']),7)
	g={'-20-0','0-20','20-40','40-60','60-80','80-105'}
	train['Temp_binned']=pd.cut(train['Temperature'],bins,labels=g,include_lowest=True)
	train['Temp_binned'] = train['Temp_binned'].map({"-20-0":0,"0-20":1, "20-40":2,"40-60":3, "60-80":4,"80-105":5})
	num_Temp_one_hot = pd.get_dummies(train['Temp_binned'], 
                                     	prefix='Temperature')
	bins=np.linspace(min(train['Fuel_Price']),max(train['Fuel_Price']),6)
	g={'2.0-2.5','2.5-3.0','3.0-3.5','3.5-4.0','4.0-4.5'}
	train['Fuel_Price']=pd.cut(train['Fuel_Price'],bins,labels=g,include_lowest=True)
	train['Fuel_Price'] = train['Fuel_Price'].map({"2.0-2.5":0,"2.5-3.0":1, "3.0-3.5":2,"3.5-4.0":3, "4.0-4.5":4})
	num_Fuel_one_hot = pd.get_dummies(train['Fuel_Price'], 
                                     	prefix='Fuel_Price')
	train['MarkDown1'].fillna(0,inplace=True)
	bins=np.linspace(min(train['MarkDown1']),max(train['MarkDown1']),6)
	g={'-10-20000','20000-40000','40000-60000','60000-80000','80000-100000'}
	train['MarkDown1']=pd.cut(train['MarkDown1'],bins,labels=g,include_lowest=True)
	train['MarkDown1'] = train['MarkDown1'].map({"-10-20000":0,"20000-40000":1, "40000-60000":2,"60000-80000":3,"80000-100000":4})
	num_MK1_one_hot = pd.get_dummies(train['MarkDown1'], 
                                     	prefix='MarkDown1')
	train['MarkDown2'].fillna(0,inplace=True)
	bins=np.linspace(min(train['MarkDown2']),max(train['MarkDown2']),6)
	g={'-300-20000','20000-40000','40000-60000','60000-80000','80000-106000'}
	train['MarkDown2']=pd.cut(train['MarkDown2'],bins,labels=g,include_lowest=True)
	train['MarkDown2'] = train['MarkDown2'].map({"-300-20000":0,"20000-40000":1, "40000-60000":2,"60000-80000":3,"80000-106000":4})
	num_MK2_one_hot = pd.get_dummies(train['MarkDown2'], 
                                     	prefix='MarkDown2')
	train['MarkDown3'].fillna(0,inplace=True)
	bins=np.linspace(min(train['MarkDown3']),max(train['MarkDown3']),6)
	g={'-30-20000','20000-40000','40000-60000','60000-80000','80000-145000'}
	train['MarkDown3']=pd.cut(train['MarkDown3'],bins,labels=g,include_lowest=True)
	train['MarkDown3'] = train['MarkDown3'].map({"-30-20000":0,"20000-40000":1, "40000-60000":2,"60000-80000":3,"80000-145000":4})
	num_MK3_one_hot = pd.get_dummies(train['MarkDown3'], 
                                     	prefix='MarkDown3')
	train['MarkDown4'].fillna(0,inplace=True)
	bins=np.linspace(min(train['MarkDown4']),max(train['MarkDown4']),6)
	g={'0-10000','10000-20000','20000-30000','30000-40000','40000-70000'}
	train['MarkDown4']=pd.cut(train['MarkDown4'],bins,labels=g,include_lowest=True)
	train['MarkDown4'] = train['MarkDown4'].map({"0-10000":0,"10000-20000":1, "20000-30000":2,"30000-40000":3,"40000-70000":4})
	num_MK4_one_hot = pd.get_dummies(train['MarkDown4'], 
                                     	prefix='MarkDown4')
	train['MarkDown5'].fillna(0,inplace=True)
	bins=np.linspace(min(train['MarkDown5']),max(train['MarkDown5']),6)
	g={'0-10000','10000-20000','20000-30000','30000-40000','40000-110000'}
	train['MarkDown5']=pd.cut(train['MarkDown5'],bins,labels=g,include_lowest=True)
	train['MarkDown5'] = train['MarkDown5'].map({"0-10000":0,"10000-20000":1, "20000-30000":2,"30000-40000":3,"40000-110000":4})
	num_MK5_one_hot = pd.get_dummies(train['MarkDown5'], 
                                    	 prefix='MarkDown5')
	bins=np.linspace(min(train['CPI']),max(train['CPI']),4)
	g={'120.0-140.0','140.0-200.0','200.0-220.0'}
	train['CPI']=pd.cut(train['CPI'],bins,labels=g,include_lowest=True)
	train['CPI'] = train['CPI'].map({"120.0-140.0":0,"140.0-200.0":1, "200.0-220.0":2})
	num_CPI_one_hot = pd.get_dummies(train['CPI'], 
                                     	prefix='CPI')
	bins=np.linspace(min(train['Unemployment']),max(train['Unemployment']),4)
	g={'2-6','6-10','10-15'}
	train['Unemployment']=pd.cut(train['Unemployment'],bins,labels=g,include_lowest=True)
	train['Unemployment'] = train['Unemployment'].map({"2-6":0,"6-10":1, "10-15":2})
	num_un_one_hot = pd.get_dummies(train['Unemployment'], 
                                     	prefix='Unemployment')
	num_ty_one_hot = pd.get_dummies(train['Type'], 
                                     	prefix='Type')
	train["IsHoliday"] = np.where(train["IsHoliday"]==True,1,0)

	final = pd.concat([num_Store_one_hot, 
                        	num_Dept_one_hot,
                         	num_Size_one_hot,
                         	num_Temp_one_hot,
                         	num_Fuel_one_hot,
                          	num_MK1_one_hot,
                          	num_MK2_one_hot,
                          	num_MK3_one_hot,
                          	num_MK4_one_hot,
                          	num_MK5_one_hot,
                          	num_CPI_one_hot,
                          	num_un_one_hot,
                         	num_ty_one_hot,
                        	train['Year'],train['IsHoliday'],train['Day'],train['Month'],train['Weekly_Sales']] 
                        	,axis=1)
	# final=pd.concat([num_con_enc,train],axis=1)
	# final.drop(['Type','Store','Date','Dept','Size','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment'],axis=1,inplace=True)
	final.drop(final[final['Weekly_Sales']>60000].index,inplace=True)
	y=final['Weekly_Sales']
	final.drop('Weekly_Sales',axis=1,inplace=True)
	from sklearn.model_selection import train_test_split
	X_train=final
	y_train=y
	from sklearn.ensemble import GradientBoostingRegressor
	new_final=final[['Dept_0','Size_1','Dept_1','Size_4','Dept_2','Type_B','Day','Month','Dept_4','Size_2','Dept_3','Store_4','Store_3','Store_2','CPI_0','Size_3','Year','CPI_2','Unemployment_1','Unemployment_2','IsHoliday','Fuel_Price_3','Temperature_2','Store_0','Temperature_1','Store_1','Fuel_Price_0','Temperature_4','Fuel_Price_4','Unemployment_0','Fuel_Price_1','Temperature_0']]
	Xtrain=new_final
	mod=GradientBoostingRegressor(n_estimators=150)
	# mod.fit(new_final,y)
	from sklearn.externals import joblib
	# joblib.dump(mod,'walmart.pkl')
	# a=open('walsal.pkl','rb')
	# model=joblib.load(a)
# acc_cv_log = fit_ml_algo(mod, Xtrain, ytrain,Xtest,ytest)
                                                               
                                                                    
# lr=LinearRegression()

                                     

	
	# comment = request.form['comment']
	# data = [comment]
	# vect = cv.transform(data).toarray()
	# if request.method == 'POST':

	Store=request.form['store']
	Dept=request.form['Dept']
	Date=request.form['Date']
	IsHoliday=request.form['IsHoliday']
	Type=request.form['Type']
	Size=request.form['Size']
	Temperature=request.form['Temperature']
	Fuel_Price=request.form['Fuel_Price']
	MarkDown1=request.form['MarkDown1']
	MarkDown2=request.form['MarkDown2']
	MarkDown3=request.form['MarkDown3']
	MarkDown4=request.form['MarkDown4']
	MarkDown5=request.form['MarkDown5']
	CPI=request.form['CPI']
	Unemployment=request.form['Unemployment']



	Dated = pd.to_datetime(Date)
	import datetime as dt

	# # cols = ['Date']
	# # for col in cols:
    

	Year = Dated.year
	Month = Dated.month
	Day = Dated.day
	
	if(IsHoliday=='True'):

  		IsHoliday=1
	elif(IsHoliday=='False' ):
		IsHoliday=2
	else:
		IsHoliday=3
		# print('ERROR IN IsHoliday')

  		
	# Type = pd.get_dummies(Type,prefix='Type')
	if(Type=='A'):
		Type=1

  		# Type=0
	elif(Type=='B'):
		Type=2

  		# Type=1
	elif (Type=='C'):
		Type=3

  		# Type=2
	else:
  		Type=4
  		# print('ERR')
  		# Type=3
  		# print('ERROR')
  	

	
	# newarr=[Store,Dept,Year,Month,Day,IsHoliday,Type,Size,Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment]
	if (Type!=4 & IsHoliday!=3):

		newarr=[]
		newarr=[Store,Dept,Year,Month,Day,IsHoliday,Type,Size,Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	# np.pad(newarr,15)
		final_features = [np.array(newarr)]

	







	
	
		prediction = model.predict(final_features)
	
	
		output = round(prediction[0], 2)
	
	# return render_template('indwal.html', prediction_text='Sales should be $ {}'.format(output)) 
		return render_template('indwal.html', prediction_text='Sales should be $ {}'.format(output))
	else:
		# print('Enter Correctly!!')
		return render_template('indwal.html', prediction_text='Sales cannot be predicted on improper values')
   


		
	    
	    

	    

 	

if __name__ == "__main__":
    app.run(debug=True)