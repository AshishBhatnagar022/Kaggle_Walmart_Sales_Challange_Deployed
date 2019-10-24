# Kaggle_Walmart_Sales_Challange

<b> Problem Statement </b>

You are provided with historical sales data for 45 Walmart stores located in different regions. Each store contains a number of departments, and you are tasked with predicting the department-wide sales for each store.

In addition, Walmart runs several promotional markdown events throughout the year.
These markdowns precede prominent holidays, the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and Christmas. 
The weeks including these holidays are weighted five times higher in the evaluation than non-holiday weeks. 
Part of the challenge presented by this competition is modeling the effects of markdowns on these holiday weeks in the absence of complete/ideal historical data.

<b>Data Description</b>

There are three files given to download: features.csv, stores.csv and test.csv,train.csv The train data consists of sample projects from the May 2009 to May 2015. The test data consists of projects from June 2015 to March 2017.

<b>The Goal</b>

Use historical markdown data to predict store sales


# Deployment using Flask

Project Structure This project has Three major parts :

1.Walmart_sales_pred.ipynb - This contains code for our model to predict the Sales.

2.appwal.py - This contains Flask APIs that receives the user's input, computes the sales based on our model and returns it.

3.templates - This folder contains the HTML template to allow user to enter the details required and displays the Sales according to the input.

Running the Project

Run appwal.py

You should be able to view the homepage as below :

![walmartUI](https://user-images.githubusercontent.com/50323219/67513810-4051dc00-f6b9-11e9-807d-a792ad910099.JPG)

Enter The details asked in the form and hit Predict

The Sales predicted will be displayed:

![WalmartPRED](https://user-images.githubusercontent.com/50323219/67513964-860ea480-f6b9-11e9-898c-24e1cccc59cb.JPG)

