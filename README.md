# Buying Your Home in Ames
## Predicting Housing Prices with the Power of Machine Learning


### Problem Statement

Buying a home is one of the most important investments you can make, and one that has a profound impact on your quality of life. You will, after all, be literally living there. However, there are nearly an infinite things that you have to consider when evaluating a house, from the big factors such as size and location, down to the details of the materials of the walls or whether the bathroom mirrors are big enough. With enough money, you can fix any problem or find the perfect house, but that's not always possible. How do we ensure that you are getting the best house for your budget?

### Executive Summary

In this report, we provide our findings in using machine learning in order to predict the sale prices of houses in Ames, Iowa using a multiple linear regression model. We analyzed a data set from 2006 - 2010 of over 2000 homes that included over 80 features to train our model. From this, we can determine which factors have the greatest impact on sale prices and provide information on whether a house has a competitive price in comparison to houses with similar attributes.  

### Project Organization

The project files are organized as follows:  
  
**Main Directory/**  
- README.md: Project summary document (this file)
- [presentation.pdf](presentation.pdf): Project presentation slides    
- **code**/
 - [general_functions.py](code/general_functions.py): Python file containing generic functions for exploring and cleaning datasets
 - [p2_eda.ipynb](code/p2_eda.ipynb): Notebook containing exploratory data analysis, cleaning, and preparation for modeling
 - [p2_model.ipynb](code/p2_model.ipynb): Notebook containing model instantiation, fitting, and evaluation
 - [p2_vis.ipynb](code/p2_vis.ipynb): Notebook containing visualization of the data and model metrics
- **datasets**/
 - [train.csv](datasets/train.csv): Housing data for the purpose of model training. Includes sale price
 - [test.csv](datasets/test.csv): Additional housing data for model testing purposes. Does not include sale price
 - [p2_lee_predict.csv](datasets/p2_lee_predict.csv): Model predicted sale prices for the housing data in the testing data set. 
- **images**/: Directory containing image files for generated figures and charts


### Data Dictionary

This dictionary provides information *only* on the data features that are utilized in our model. The explanation for the full list of features (and where the information below was taken from) can be found [here](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt)

|Feature|Type|Description|
|---|---|---|
Age|int|The number of years since the house was built, *or* since it was last remodeled. This was not included in the original dataset, but was extrapolated from year built/remodeled and the sale year
Bldg Type|str|Type of home that was sold: single-family detached (1Fam), two-family conversion (2FmCon), duplex (Duplx), townhouse end unit (Twnhse) or townhouse inside unit (TwnhsI)
Garage Area|int|The area measurement of the garage in square feet
Gr Liv Area|int|The area measurement of the above grade ground level living space in square feet
Neighborhood|str|The neighborhood where the house is placed, with 28 possible assignments. For the full list of included neighborhoods, consult the full documentation linked above
Overall Qual|int|Rating on the overall materials and finish of the house from 1 (very poor) to 10 (very excellent)
Sale Price|int|The amount in dollars that the house sold for (our target variable)
Total Bsmt SF|int|The area measurement of the basement in square feet

### Machine Learning Process

The variables included in the dictionary above and in the project code reflect only the final iteration of the finished model. During the process of generating the model, numerous combinations of features and various modeling techniques were implemented, evaluated, and then either retained or discarded based on the models' predictive power. For a more complete description of the tested models, refer to the comments in the model code [here](code/p2_model.ipynb), under model iterations. The finished model was selected for minimizing both bias and variance.

### Conclusion and Recommendations
For a more detailed analysis and examination of our results and how these conclusions were drawn, refer to the [results and visualization documentation](code/p2_vis.ipynb)

Using machine learning, we have fit a multiple linear regression model in order to predict the sale prices of homes in Ames, Iowa. Our model was 90% effective in accounting for the variance exhibited in the data we modeled, and utilized a relatively small number of features in order to create this model. From our results, we determined that several features make a relatively minor impact overall to housing prices. The location of the home, for most neighborhoods, is clustered within a relatively small range, although there are a few exceptions. Of course, there are outside reasons for these price differentials (infrastructure, community and so forth), if those areas are not of interest, then settling in one of the other neighborhoods will not see a drastic change in price, so you can select the neighborhood most to your liking without having to worry about the financial repurcussions. Similarly, age has a relatively small impact on the price; 3% decrease per 10 years. From a value perspective (need for repairs and other maintenance) it is likely more economical to purchase newer houses in general.

The other features had a much greater impact on project sale price. Overall quality of the house, unsurprisingly, plays a big factor in determining how much a home sells for. When evaluating a potential house, it will be important to consider the condition and quality of the house, and whether you are willing to pay a premium for a well maintained home, or if a home lower on the scale will suffice, as this can represent shifts of 20 to 40%. House size, especially the garage and living areas, also represents 20 to 30% differences in price per 1,000 square feet. Again, it is up to your personal preference as to how much the increased space is worth to you.

Ultimately, this model is not designed to select the perfect house for you. What this model provides are benchmarks for you to determine if you are getting a fair deal or not. If you are interested in a 2,000 sq. ft. home for 250,000 USD, but the 1,800 sq. ft. home next door is selling for 220,000, we can help you determine if you are overpaying for the increase in size (disregarding other factors of course). With our model, we can help you make those assessments and evaluate any homes you are interested in order to ensure that you are getting the best home you can for your money