# import jwt
# import logging
from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS, cross_origin
from job_sal_pred_class import SalaryPredictor
# from db_logger import DBHandler
# import settings

log_error_level = 'WARNING'
app = Flask("Gria Job Prediction Model")
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["DEBUG"] = False

# logging.basicConfig(filename="smarthooks_ds_api.log")

# def log(db_name, message):
#     db_handler = DBHandler(settings.DB_HOST, db_name)
#     logging.getLogger(__name__).addHandler(db_handler)
#     logger = logging.getLogger(__name__)
#     logger.setLevel(log_error_level)
#     logger.exception(message)
#     logging.getLogger(__name__).removeHandler(db_handler)


@app.route('/api/predict', methods=['POST'])
@cross_origin()
def job_salary_prediction():
    title = request.headers.get('title')
    full_description = request.headers.get('full_description')
    location_normalized = request.headers.get('location_normalized')
    contract_time = request.headers.get('contract_time')
    company = request.headers.get('company')
    category = request.headers.get('category')
    sourceName = request.headers.get('sourceName')

    sal_predictor = SalaryPredictor()

    X = [title, full_description, location_normalized, contract_time, company, category, sourceName]
    print(X)
    prediction = sal_predictor.predict(X)

    response = jsonify(str(prediction))
    return response


if __name__ == '__main__':
    app.run(host='localhost', port=8080)