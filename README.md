# AutoML
Why? 
1) I got tired of having to remake the same set up code and evaluations for individual models. 
2) I wanted a way to compare different model together that was simple and informative. 
3) I was wasting time trying to reimplment the same feature engieering tricks I learned. 

## Feature engineering
- There are lots of way to analyze data before running it into the model
  - So there are features I have cooking up to make that easier.
  - All the functions that start with 'discover' are meant to be used as tools for looking at the features during analysis. 

## STATS
- Ability to set up a confusion matrix
- Cross Validation Metrics
  
- Categorical Metrics and/or Continuous Metrics
  - Basic Categorical Metrics
    - Model Accuracy
    - Precision
    - Recall
    - F1 Score
    
  - Basic Continuous Metrics
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Mean Absolute Percentage Error (MAPE)
    - R-squared (R2) (Test)
   
  - Advanced Metrics
    - ROC AUC (higher is better)
    - Brier Score (Lower is better)
      
- Compare two models together
  - using compare_stats_catigorical_models(model1, model2)
    - its not based on the class so you can run it independently if needed  
  - Want to adjust this function more so that it will take any amount of models and sort from best to worst
 
Note:
- *I am using google collab for my Machine Learning tasks so the code is meant to be ran there, as individual cells. But, it can be ran locally with the files I provided to.* 
- *So if you question why the files are broken up its because I want to save updates per the cell I am editing not the entire code since thats not the intended way to use google collab*
