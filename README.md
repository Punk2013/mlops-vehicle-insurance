# mlops-vehicle-insurance

Has bad design and bugs. Needs a lot more work. I hope to get some advice.

The model itself should predict whether claim was paid, but it wasn't tuned yet and tends to just always predict false.

# Usage:

python run.py -mode update - adds batch to the training data and fits the model with it. Data is from data/motor_data11-14lats.csv.
You can adjust batchsize in run.py

python run.py -mode summary - get some information about data quality and scores. Crashes if too little data is added

python run.py -mode inference -file <filename> - get predictions for data in file. Crashes if too little data is added

# TODO
- not using ohe on MAKE
- better data quality analysis
- more model quality scores
- partial_fit
- better handling of streamed data
- better tuned model
- second model for CLAIM_PAID regression
- separate directories for stages
- data quality metaparameters and data drift detection
- interface for using existing model vesioning system
- performance metrics
