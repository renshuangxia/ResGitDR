# ResGitDR

[![DOI](https://zenodo.org/badge/659061242.svg)](https://zenodo.org/doi/10.5281/zenodo.11559736)

# Instructions for installing required libraries:

1.	Clone project repo form GitHub:
	* ssh:  git@github.com:renshuangxia/ResGitDR.git
	* http: https://github.com/renshuangxia/ResGitDR.git
2.	Prepare python environment with conda (for installing conda, please refer to https://docs.anaconda.com/free/miniconda/): 
	* conda create -n your_env_name python=3.10
3.	In a terminal, export PYTHONPATH to your repo directory with command: 
	* export PYTHONPATH=”{ABSOLUTE_PATH_TO_YOUR_GIT_REPO}”
4.	Go to project repo root directory, run command: 
	* pip install -r requirements.txt


# Guidelines to operate the software

## 1.	Data loading
	You can download all the input data from the following link:
	https://drive.google.com/drive/folders/1cdpyX-Qsp4ilhhf4Zay0FeG6X8cSXDyZ?usp=sharing
	After downloading, unzip the files and place them in your repository in a folder named “data”.
## 2.	Cross validation
To obtain the cross-validation results, you must first run ResGit_train_cross_validation.py and then run DR_train_cross_validation.py.
Our model has two parts: the first part (ResGit) generates hidden representations, which are then used by the second part for drug sensitivity prediction.
### 2.1	ResGit: using following command to run
* python ResGit_train_cross_validation.py
### 2.2	Drug sensitivity prediction: using following command to run
* python DR_train_cross_validation.py
## 3.	Parameter analysis
To get the all the parameter and then do the parameter analysis, please first run ResGit_train_parameter_analysis.py and then run python DR_train_parameter_analysis.py
### 3.1	ResGit: using following command to run
* python ResGit_train_parameter_analysis.py
### 3.2	Drug sensitivity prediction: using following command to run
* python DR_train_parameter_analysis.py
## 4.	End to End Prediction Models with cross validation
To train the two end to end models (multi-task or SGA2DR), you will need to choose the “model name” in Multitask_and_SGA2DR_train_cross_validation.py to be either "Multitask" or "SGA2DR", and then run: 
* python Multitask_and_SGA2DR_train_cross_validation.py


# Description of files and their purposes
## 1. Model "ResGitDR" contains two parts (please see Fig.1 in the manuscript): 
### 1.1	Representation learning module which used RseGit to predict gene expression by taking the Somatic genome alterations (SGAs) and cancer type as input. “model/ResGit.py” is the model file. 
#### 1.1.1.	To train this part with cross validation, please run the file named "ResGit_train_cross_validation.py". The result was shown in Fig.2a&b in the manuscript.
#### 1.1.2.	In our experiment, we all did the parameter analysis which used all data as training data to get the parameters of the model, please run the file named "ResGit_train_parameter_analysis.py". The result was shown in Fig.2c-e in the manuscript.
### 1.2.	Drug response prediction module, which used elastic net to predict drug sensitivity by taking the hidden representations learned in the first module and SGAs as input. 
#### 1.2.1.	After training the ResGit with cross validation, please run the file named "DR_train_cross_validation.py" to predict the drug response with cross validation. (This file must run after running “python ResGit_train_cross_validation.py”). The result was shown in Fig.3 in the manuscript.
#### 1.2.2.	After training the ResGit with parameter analysis, please run the file named "DR_train_parameter_analysis.py" to get the parameter of elastic net for each drug. (This file must run after running “python ResGit_train_parameter_analysis.py”). The result was shown in Fig.4 in the manuscript.
## 2.	We also try to train end-to-end models to predict the drug response with the input of Somatic genome alterations (SGAs) and cancer type. We developed two kinds of models: 
### 2.1.	 "models/Multi_task.py", which predict the drug response and gene expression values at the same time using the same architecture of ResGit;
### 2.2.	 “models/SGA2DR.py", which predict the drug response using the same architecture of ResGit;
### 2.3.	 Using these two models to get the cross-validation results, please run the file named "Multitask_and_SGA2DR_train_cross_validation.py".The result was shown in Supplementary Fig.S4 in the manuscript.
