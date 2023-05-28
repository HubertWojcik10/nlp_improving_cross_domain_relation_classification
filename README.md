# NLP - Improving Cross-Domain Relation Classification by Adding Domain-Specific Context

A natural language processing university project, where we used a cross-domain dataset and a state-of-the-art neural network classifier to investigate the performance of adding more context to the dataset.

*What we discovered was that by adding a special token with a domain name (e.g. [music]), we can slightly increase the performance of a cross-domain relation classification model.*

## Important Details
- group work: our team consisted of 3 people
- our work is based on [Bassignana and Plank (2022a)](https://aclanthology.org/2022.findings-emnlp.263/)
- the submission was an academic paper + code


## How to run the code? (required guide for the exam)
- clone the repository
- run "cd baseline" to enter the baseline folder
- open run.sh to change the test domain as well as specify whether you want to run the model or baseline. Remember to change the seed as well as the domain variable in the run.sh file
- run "./run.sh" in the terminal to run the code


## The structure of the code
- baseline folder: code with model built on the baseline
- analysis folder: code with model built on the analysis needed for the report
- generate.py: code to generate the merged dataset
