# Credit Risk Evaluation Application

This application helps a loan granting officer of a bank or financial organisation to evaluate the loan applications. This helps in making decisions with regard to the applied loans.  
**Background:** Risk for the applied loan can be evaluated using three methods - Rule based, Statistical based and Machine Learning based. Rule based models rely on well defined rules to judge a use case while Statistical and Machine Learning models rely on the data of other previously given loans. A decision can be suggested in this case by analysis of this data.  
**Usage of the application:** Points to note while using the application:

1.  **Users:** There are two users of the application: Loan Administrator and Loan Officer. Loan Admin is responsible for setting rules for the rule based model and adding dataset for statistical and ML based models. Loan Officer can add the dataset for testing the loan applicants and choose to see the results for a loan id using any of the three models for evaluation.
2.  **Setting rules:** For rule based model, add features using the "Add Feature" screen, configure the feature using "Configuration" screen with a weightage between 0 and 1 such that weightages of all the features add to 1\. Then, set criteria for a feature using "Criteria" screen.
3.  **Setting criteria:**
    *   Select the feature, product and category from the dropdown lists in the "Criteria" screen.
    *   Select the data source. If it is XML/JSON, specify the source url in API and provide the key. If it is SQL, write the SQL query from the database named "db". e.g. "SELECT loan_amnt from db". You need not provide key for SQL.
    *   Set criterias and their scores. If the feature is categorical, set criteria using "is feature_name". e.g. For a feature say "gender", set the score for criteria "is Male" and "is Female" differently.
4.  **Uploading datasets:**
    *   For uploading dataset of customers who have applied for loan, just uplaod a ".csv" file in the "Upload Dataset" screen of Loan Officer.
    *   For uploading dataset for training, upload a ".csv" file in "Upload Dataset" screen of Loan Admin.
5.  **Seeing Results:**
    *   You can see the results by entering a loan id and checking any of the models in the "Scorecard" screen of the Loan Officer if the screen shows that the model is ready for analysis.
    *   The rule based model gives the score (between 1-10) according to the specified rules.
    *   The statistical and machine learning based models give probability of approval of the loan according to the best machine learning/statistical model implemented.

