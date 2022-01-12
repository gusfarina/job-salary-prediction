from job_sal_pred_class import SalaryPredictor

path = 'datasets/Train_rev1.csv'
X_columns = ["Title", "FullDescription", "LocationNormalized", "ContractTime", "Company", "Category", "SourceName"]
y_column = 'SalaryNormalized'

sal_predictor = SalaryPredictor()

sal_predictor.train(path, X_columns, y_column)
