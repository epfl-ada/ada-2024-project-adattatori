# All you need is love🫂 … and a good chemical affinity🧪

## URL to the data story

Here is the URL to our data story: https://federicorossi498.github.io/adattatori-web/

## Abstract
Humans, as social beings, often form meaningful romantic relationships. However, such bonds come with risks, including the transmission of infections. Sexually transmitted diseases (STDs) remain a significant global health issue, with the World Health Organization recognizing their eradication as a key objective in its 2030 agenda [1]. Among these diseases, HIV-1 stands out due to its enduring impact.
Interestingly, the concept of relationships extends beyond humans to the chemical realm, where compounds are designed to bond with their “target proteins”. This project aims to explore how these chemical “couples” contribute to the fight against HIV-1. By studying how the affinity of these drug-target pairs is influenced by both the target protein and the ligand's features, in our data story we try to shed a light on what makes a "strong" chemical couple.

References
[1] https://www.who.int/publications/i/item/9789240053779


## Research Questions
- Which are the main different types of target proteins in the fight against HIV?
- Is there any target with significant better affinity score?
- Can we find features of the ligand that are linked to higher affinities?

## Proposed additional datasets

We will not be using any additional datasets for our analysis. We consider that BindingDB provides us with the necessary information to carry out our project proposal.


## Methods

**Part 1: Preparing and cleaning the dataset** 

- Step 1: After inspection of the original BindingDB dataset, we build the master STDs dataset selecting only rows (ligand-target interactions) that are linked to STDs. We manually inspected the list of target source organisms and selected only viruses related to STDs. 

- Step 2: Cleaning the dataset: This step is crucial for obtaining interpretable results in the next parts. In order to do so, we plotted the availability of binding affinity metrics, and seeing that IC50 was the most abundant one, we selected only rows with IC50 available values. Then we removed rows were NaN values were too abundant (> 50 %).

- Step 3: General exploration of this dataset and selection of HIV, as this organism proved to be the one on which we had the most data available.

**Part 2: Focusing on HIV: ligand-target couples**

- Step 1: Analysis of target proteins in HIV: these are the elements where we "land our attacks" on the virus. Plot of the different types we found and their frequencies. Three are the most present: Reverse Transcriptase, Gag-Pol polyprotein, and Integrase.
    - Step 1.1: Analysis of logarithmic IC50 (LogIC50, logarithmic transform is employed since the metric's distribution is very skewed) to see the relationship between this value and the different target proteins.
    - Step 1.2: Statitical test to study the normality of the distribution of the data for each target for subsequent analysis: Kolmogorov-Smirnov test is employed. Almost all target's LogIC50 is confirmed to be not normally distributed.
    - Step 1.3: Two different statistical test to prove how the median varies among the targets: Kruskal-Wallis test and post-hoc Dunn’s test. We use these tests and the median metric because the distribution are not normal. The latter shows that at the standard significance threshold of 0.05, the Gag-Pol polyprotein emerges as a standout, with significantly lower values of LogIC50 than other targets.

- Step 2: Focus on ligands: these are our weapons against the virus, and more specifically against the target that seems to show an overall better affinity, that is the Gag-Pol polyprotein.
    - Step 2.1: Expansion of the dataset with RDKit obtained molecular features
    - Step 2.2: PCA analysis:
        - Step 2.2.0: test to prove linearity: the results confirm that the relationship between IC50 and the features is not linear.
        - Step 2.2.1: Obtain the absolute molecular features contributions to the first five principal components.
        - Step 2.2.2: Compute a total contribution score by weighing the features contributions on the percentage of variance explained by each principal component.
        - Step 2.2.3: Extraction of the most relevant feature, that was found to be the `number of sp3 hybridized carbons`
    - Step 2.3: Study of the relationship between the IC50 metric and different groups of values of the selected feature.
    - Step 2.4: Normality test (Kolmogorov-Smirnov) to assess groups' distributions normality. Almost all distribution are not normal.
    - Step 2.5: Kruskal-Wallis test and post-hoc Dunn's test to assess differences in the IC50 medians of the groups: an intermediate number of sp3 hybridized carbons seems to be associated with a better binding affinity.
 
- Step 3: Manual inspection of findings.

- Extra step: To validate our results, we perform the analysis step1-step3 but with the dataset divided into 80% for training and 20% for testing. This approach was used to evaluate potential imbalances in the datas and to be sure our findings were valid. All these analyses are grouped in the train_test_nb.ipynb notebook.

## Organization of the repository
The structure of our repository is the following:
```
├── data                        <- BindingSTD dataset
│
├── src                         <- Source code
│   ├── data                            <- Data directory
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
In the `utils` folder there are 2 files: `data_utils.py` contains the functions that can be applied to the dataset to format it in a usable shape; `evaluation_utils.py` contains the functions used to plot the results and to perform statistical tests.
In the `tests` folder we added all the analysis and explorations that we performed.  
The `result.ipynb` main file consists for now of the exploratory data analysis that has been performed in the scope of milestone 2.
In `environment.yml` file there are all the python dependicies needed to run our code. In the `data` folder there are two files: `dataloader.py` contains the data loading functions; `std_extraction.ipynb` contains the workflow to obtain the STDs dataset that we use also in the principal notebook. The `scripts` folder contains the `rdkit_extraction.py` file, that contains some useful methods to leverage the RDKit library for our analyses

## Followed timeline 

| Week  | Objective/Task                                                                                           | Description/Details                                                                                      |
|-------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Week 12 |**Data Story Creation**                                                                                 | Begin creating a data story that aligns with the research question. The narrative will communicate key insights. |
|       | **Exploration of Protein Targets**                                                                  | Explore target proteins and identify relevant structures for predicting binding affinities. |
| Week 13 | **Website Development**                                                                                  | Start working on the webpage layout and design for presenting the findings in an engaging way. |
|       | **Exploration of Ligand features**                                                                  | Explore ligands and undercover relevant features influencing affinity. |
| Week 14 | **Finalizing Data Story Webpage**                                                                         | Complete the webpage, ensuring it is cohesive and interactive with the data story clearly presented. |
|       | **Visualizations**                                                                                        | Host interactive visualizations that clearly present the results of the analysis. |
|       | **Final Review**                                                                                         | Perform final review and testing of the webpage to ensure functionality and engagement. |

## Contributions of all group members

**Leonor**: Graphs plotting, website development and data story writing, targets explorations.

**Elisa**: Website development, data story and README writing, ligands explorations.

**Mattia**: Ligands' features analysis, explorations with RDKit, final notebook writing.

**Federico**: Website development, final notebook writing, code quality control.

**Simone**: Whole dataset explorations, protein targets analysis, final notebook writing.


