# Job Salary Prediction

## How to run the project

### Instal a virtual env using pip

    pip install virtualenv
    
### Create your virtual environment and then activate it

    virtualenv venv
    
    source venv/bin/activate
    
### Instal the required packages from the requirements file

    pip install -r requirements.txt

### Run the file model_class_training.py to train the AI model

    python model_class_training.py

### Start the API

    python api.py

### Fecth the api prediction using the url

    http://localhost:8080/api/predict

### Pass the prediction parameter in the header of the POST request