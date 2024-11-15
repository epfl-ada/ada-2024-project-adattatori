# All you need is love🫂… and a good chemical affinity🧪

## Abstract
As social beings, we humans tend to create meaningful romantic relationships. Nevertheless, jumping into this boat implies assuming the many risks that come along with the decision of being part of a couple. Amongst these numerous risks, a generalized one is related to infections and viral transmissions between the individuals concerned. Emblematically, sexually transmitted infections and subsequent diseases (STDs) are a representative example of this, to the extent that the World Health Organisation still recognises them as a prominent problem, integrating their potential eradication in its 2030 agenda purposes [1]. 
However, not only humans live their life in couples but so do chemical compounds, which are created with the only purpose and sole aim of belonging to their future partner: “their target protein”. 
With this project we are keen to see how “couples” in the chemical domain strive to combat this aforementioned human problem, contributing to the health of human relationships. Thus, we would like to trace development of the drugs against STDs, with a particular focus on HIV, and try to depict the main changes in drug affinity for their targets over time. We will embark on a quest to seek similarities in the ligands that can better predict affinity and therefore show the tip of the iceberg of the enormous potential that chemical binding has to pursue this aim.  

References
[1] https://www.who.int/publications/i/item/9789240053779


## Research Questions
Time independent questions:

- What are the main areas in the world in which HIV research is conducted?

- How many different types of target proteins for HIV are in BindingDB?

- What are the drugs that have the highest binding affinity to their targets?

- Which structures do these drugs have in common?

Time dependent questions:

- Are medical innovations and scientific advances related to improvement in drug discoveries and affinities throughout the years?
- Given a single target or a single ligand, does the binding affinity change over time? 


## Proposed dataset

We will not be using any additional datasets for our analysis. We consider that BindingDB provides us with the necessary information to carry out our project proposal.


## Methods

**Part 1: Preparing and cleaning the dataset** 

- Step 1: After inspection of the original BindingDB dataset, we build the master dataset selecting only rows (ligand-target interactions) that are linked to STDs. We manually inspect the target source organisms and selected those related to STDs. 

- Step 2: Cleaning the dataset: This step is crucial for obtaining interpretable results in the next parts. In order to do so, we plotted the availability of binding affinity metrics, and seeing that IC50 was the most abundant one, we selected only rows with IC50 available values. Then we removed rows were NaN values were too abundant.

- Step 3: General exploration of this dataset and selection of HIV, as this immunodeficient disease proved to be the one on which we had the most data available.

**Part 2: Focusing on HIV**

- Step 1: We extract interesting features for posterior visualization in our project (year of publication of the associated articles).

- Step 2: General analysis of ligand-target couples in our HIV dataset. In order to carry this out, we study the structure of ligands in detail, thanks to the Python Rdkit library (after an additional cleaning). Extraction of molecular features from the SMILES string permitted us to study the correlation between these features and binding affinity. For now, we have not uncovered any significant relationships. This might lead to further exploration of other features for P3 and other Rdkit functionalities. Furthermore, we performed K-means clustering on the similarity matrix of fingerprints.

- Step 3: Literature review and decision to study some of the most common drug classes that are used to treat HIV. We analyzed what proteins were targeted by each drug class. Also, we considered it was a good idea to show the drug class distribution over the years, as well as the IC50 distribution for each drug class in order to get closer to our project goal.


## Organization of the repository
The structure of our repository is the following:
```
├── data                        <- BindingSTD dataset
│
├── src                         <- Source code
│   ├── data                            <- Data directory 
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Complete performed analysis
│
├── results.ipynb               <- Notebook showing our results
│
├── environment.yml             <- File for installing python dependencies
└── README.md
```
In the 'utils' folder there are 3 files: 'data_utils.py' contains the functions that can be applied to the dataset to format it in a usable shape; 'evaluation_utils.py' contains the functions used to plot the results.
In the 'tests' folder we added all the analysis and explorations that we performed.  
The 'result.ipynb' main file consists for now of the exploratory data analysis that has been performed in the scope of milestone 2.
In 'environment.yml' file there are all the python dependicies needed to run our code. 


## Proposed timeline 

| Week  | Objective/Task                                                                                           | Description/Details                                                                                      |
|-------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Week 12 | **Analysis of Drug Discovery Timeline**                                                                  | Review the historical evolution of drug discovery for STDs. Analyze and compare past methods with current state-of-the-art techniques. |
|       | **Exploration of Protein-Ligand Binding**                                                                  | Explore amino acid sequences of target proteins and identify relevant structures for predicting binding affinities. |
|       | **Supervised Learning Model**                                                                             | Investigate the possibility of training a supervised learning model to predict binding affinity based on protein-ligand characteristics. |
| Week 13 | **Data Story Creation**                                                                                 | Begin creating a data story that aligns with the research question. The narrative will communicate key insights. |
|       | **Website Development**                                                                                  | Start working on the webpage layout and design for presenting the findings in an engaging way. |
| Week 14 | **Finalizing Data Story Webpage**                                                                         | Complete the webpage, ensuring it is cohesive and interactive with the data story clearly presented. |
|       | **Visualizations**                                                                                        | Host interactive visualizations that clearly present the results of the analysis. |
|       | **Final Review**                                                                                         | Perform final review and testing of the webpage to ensure functionality and engagement. |


## Organization within the team

**Mattia**: Further exploration to uncover correlations between features and affinity.  

**Federico**: Analysis of the actual drug discovery and comparison with the past one.

**Simone**: Analysis of the drug discovery in the past, retrieving interesting features to make comparisons.

**Leonor**: Creativity ideation of the webpage and formatting of the repository for its creation.

**Elisa**: Creativity ideation of the webpage format and recovery of protein sequences and features from UniProt or PDB.  

