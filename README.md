# All you need is love🫂 … and a good chemical affinity🧪

## Abstract
Humans, as social beings, often form meaningful romantic relationships. However, such bonds come with risks, including the transmission of infections. Sexually transmitted diseases (STDs) remain a significant global health issue, with the World Health Organization recognizing their eradication as a key objective in its 2030 agenda. Among these diseases, HIV stands out due to its enduring impact.
Interestingly, the concept of relationships extends beyond humans to the chemical realm, where compounds are designed to bond with their “target proteins”. This project aims to explore how these chemical “couples” have contributed to combating STDs, particularly HIV. By tracing affinity levels, we aim to analyze changes in drug affinity for their targets due to different features. Additionally, our analysis will expand to ligands to better understand how drug development has progressed, focusing on ligands with varying properties. 

References
[1] https://www.who.int/publications/i/item/9789240053779


## Research Questions
- Which are the main different types of target proteins in the fight against HIV?
- Is there any target with significant better affinity score?
- Which features do the ligands with high affinty have in common?
- 

## Proposed dataset

We will not be using any additional datasets for our analysis. We consider that BindingDB provides us with the necessary information to carry out our project proposal.


## Methods

**Part 1: Preparing and cleaning the dataset** 

- Step 1: After inspection of the original BindingDB dataset, we build the master dataset selecting only rows (ligand-target interactions) that are linked to STDs. We manually inspect the target source organisms and selected those related to STDs. 

- Step 2: Cleaning the dataset: This step is crucial for obtaining interpretable results in the next parts. In order to do so, we plotted the availability of binding affinity metrics, and seeing that IC50 was the most abundant one, we selected only rows with IC50 available values. Then we removed rows were NaN values were too abundant.

- Step 3: General exploration of this dataset and selection of HIV, as this immunodeficient disease proved to be the one on which we had the most data available.

**Part 2: Focusing on HIV: ligand-target couples**

- Step 1: Analysis of target proteins in HIV: these are the elements that we use to attack this virus. Plot of the different types we found and their frequencies. Three are the most present: Reverse Transcriptase, Gag-Pol polyprotein, and Integrase.
    - Step 1.1: Analysis of IC50 to see the relationship between this value and the different targets.
    - Step 1.2: Statitical test to study the distribution of the data for subsequent analysis: Kolmogorov-Smirnov test
    - Step 1.3: Two different statistical test to prove how the median varies among the targets: Kruskal-Wallis test and post-hoc Dunn’s test. The latter shows that at the standard significance threshold of 0.05, the Gag-Pol polyprotein emerges as a standout.

- Step 2: Focus on ligands: 

- Step 3: Study the structure of ligands in detail, thanks to the Python Rdkit library (after an additional cleaning). Extraction of molecular features from the SMILES string permitted us to study the correlation between these features and binding affinity. For now, we have not uncovered any significant relationships. This might lead to further exploration of other features for P3 and other Rdkit functionalities. Furthermore, we performed K-means clustering on the similarity matrix of fingerprints.


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
| Week 12 |**Data Story Creation**                                                                                 | Begin creating a data story that aligns with the research question. The narrative will communicate key insights. |
|       | **Exploration of Protein-Ligand Binding**                                                                  | Explore amino acid sequences of target proteins and identify relevant structures for predicting binding affinities. |
| Week 13 | 
|       | **Website Development**                                                                                  | Start working on the webpage layout and design for presenting the findings in an engaging way. |
| Week 14 | **Finalizing Data Story Webpage**                                                                         | Complete the webpage, ensuring it is cohesive and interactive with the data story clearly presented. |
|       | **Visualizations**                                                                                        | Host interactive visualizations that clearly present the results of the analysis. |
|       | **Final Review**                                                                                         | Perform final review and testing of the webpage to ensure functionality and engagement. |


## Organization within the team

**Mattia**: Further exploration to uncover correlations between features and affinity.  

**Federico**: Creativity ideation of the webpage format. 

**Simone**: Analysis of the drug discovery retrieving interesting features to make comparisons.

**Leonor**: Creativity ideation of the webpage and formatting of the repository for its creation.

**Elisa**: Description of the datastory, enlargement of the analysis about proteins.  

