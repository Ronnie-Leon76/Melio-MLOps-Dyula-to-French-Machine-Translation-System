### Dyula to French Machine Translation Model Inference

- This folder contains the Deployment resources required for deploying the trained model onto Highwind. To get started first download the model from HuggingFace Hub by running the `Download_Model_From_HuggingFace.ipynb` notebook to download the model from Hugging Face and save it automatically to the `models` directory in the `Deployment` folder.

- Follow the instructions in the `README.md` file in the `Deployment` folder to build the container locally and give it a tag. After building the Kserve predictor image that contains your model, spin it up to test your model inference.

- Once, the dockerized model solution is up and running, run the `Inference.ipynb` notebook to make inference on the test data in `test_refined.csv` file.