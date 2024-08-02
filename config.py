## THIS IS THE CONFIGURATION PYTHON SCRIPT.
## FOR THIS SCRIPT TO RUN PERFECTLY TTHE PROJECT FOLDER STRUCTURE SHOULD NOT BE ALTERED INCLUDING FILDER AND FILE NAMES
## NO PATH OR LINE OF CODE IS EXPECTED TO BE CHANGED IN THIS SCRIPT

import os # import os python module

_project_dir = os.getcwd() # Get the current working Directory

_data_dir = _project_dir+'/Data/' # Path to Fetch the Dataset from

_preprocessed_data_dir = _data_dir + 'preprocessed_appended_dataset.xlsx' # Path to save the preprocessed dataset

_max_output_sequence = 25 ## Based on the data visualization performed in the visualization section

_max_vocab_length = 1500 ## Maximum Vocabulary length 

_save_model_dir = _project_dir+'/Outputs/' # Path to save model output

_Bilstm_model_dir = _save_model_dir+'BiLSTM_model_output/' # Path to BLSTM classifier model

_dnn_model_dir = _save_model_dir+'DNN_Model_Output/'

_cnn_model_dir = _save_model_dir+'cnn_output/'

_GRU_model_dir = _save_model_dir+'GRU_Model/'

_LSTM_model_dir = _save_model_dir+'LSTM_model_output/'

_Bilstm_tweet_classifier_model = _Bilstm_model_dir+'Tweet_Classifier'
