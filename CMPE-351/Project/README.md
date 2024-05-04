# Advanced Data Analytics Group Project

This repostitory contains all of the code required to replicate the results found in the final report titled "Exploring Factors Influencing Restaurant Ratings: Insights From Google Reviews". Reviews from California, New York, and Texas were analyzed. The raw dataset was provided by the University of California San Diego and can be found [here](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal).

## Data
Start by cloning the repository. Next, download the Metadata and 10-Core datasets for California, New York, and Texas. Copy the gz files to the directory CMPE351/data/.

### California Data
* Metadata: https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/meta-California.json.gz
* 10-Core: https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/review-California_10.json.gz

### New York Data
* Metadata: https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/meta-New_York.json.gz
* 10-Core: https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/review-New_York_10.json.gz

### Texas Data
* Metadata: https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/meta-Texas.json.gz
* 10-Core: https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/review-Texas_10.json.gz

## Dependencies
This project has multiple package dependencies. It assumes that you have Python3 and Pip installed. Use the command `pip install -r requirements.txt` to install the required packages. The file requirements.txt is located in the root of the repository.

## RQ1
The files required for RQ1 are located in the directory CMPE351/src/.

1. Read through RQ1.ipynb and update the file path locations to match your system
2. Run all cells in RQ1.ipynb top down

## RQ2
The files required for RQ2 are located in the directory CMPE351/src/.

1. Read through the dataPrep.py file and update the paths to the data files on your system
2. Uncomment lines 162-163 in dataPrep.py, save the file, and then execute the file
3. After the above step has completed, execute the sentiment_analysis.py
4. Next, go back to the dataPrep.py file, and comment out lines 162-163 and uncomment lines 165-166, then execute the file
5. The above should create data/review_sentiments_geo.csv
6. Run all cells in view_rq2_data.ipynb top down

## RQ3
The files required for RQ3 are located in the directory CMPE351/src/.

1. After exectuing RQ2, open rq3_rand_forest_prediction.py and rq3_xgboost_prediction.py
2. Exectue both scripts
3. After execution, open view_rq3_data.ipynb, and execute this notebook

## Group Members
* Aidan Kealey
* Eloise Callaghan
* Logan Copeland
* Nic Macdonald
