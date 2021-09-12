# MADSmilestone2
**SIADS 694/695 alzheimers and dementia project**

**Authors:**
- Stacey Rivet Beck starbeck@umich.edu
- Ian Byrne ianbyrne@umich.edu

[Project Inspiration](https://www.aicrowd.com/challenges/addi-alzheimers-detection-challenge)

Data information:
- [NHATs data access page](https://nhats.org/researcher/data-access)
- [NHATs Crosswalk](https://www.nhats.org/sites/default/files/2021-07/NHATS_R10_Final_Crosswalk_between_Instruments_and_Codebook.pdf)
- [Data user guide](https://www.nhats.org/sites/default/files/2021-07/NHATS_User_Guide_R10_Final_Release.pdf)

## **AI Crowd Labeling Method**

We received inspiration from the AI Crowd's Alzheimer's Clock Challenge and reached out to their data team inquiring about labeling the clock images for that challenge. Ankit < insert last name > explained that they use the values in the variable 'hc1disescn9' to help label the images. The variable 'hc1disescn9' contains values that indicate whether someone has dementia/Alzheimer's or not. A response of '1 YES' indicates that they have a confirmed diagnosis of Alzheimer's or Dementia and '2 NO' indicates no diagnosis has been given. A value of '7' indicates that a response of '1 YES' has already been recorded in a previous round. AI Crowd labeled each image as either:
  - 0 - Pre-Alzheimers
  - 1 - Post-Alzheimers 
  - 2 - Normal

If a subject reports '1 YES' in the variable 'hc1disescn9' in a later round, all the previous rounds receive the label '0 - Pre-Alzheimers'. All 'hc1disescn9' reports of '1 YES' and '7' receive the label '1 - Post-Alzheimers' for the current and subsequent rounds.  All else receive the label '2 - Normal.' 

We provide code to produce labels using this algorithm.  
<br><br/>
## **NHATs Study Criteria used to help make Labels**

In addition, we also feel that this way of labeling can lead to false negatives in terms of whether someone might have dementia/Alzheimer's disease. There may be participants whose diagnosis has gone undetected by medical professionals and who are able to complete the NHATs survey, thus leading to a label of '2 - Normal' for their drawings. 

In order to provide a potentially more robust set of label parameters, we will also implement a label strategy that mimicks the NHAT study found in this report. https://www.nhats.org/sites/default/files/inline-files/DementiaTechnicalPaperJuly_2_4_2013_10_23_15.pdf (page 2)


We will use these specific labels to identify if someone has:
- 0 - 'Possible Dementia'
- 1 - 'Probable Dementia' (Likely Dementia)
- 2 - 'No Dementia' 

**Variables used to help classify '0' for "Possible Dementia":**
- One cognitive test score with cut off <= 1.5 SD below the mean. 

**Variables used to help classify '1' for 'Probable Dementia' include:**
-  'hc1disescn9' for diagnosis ('1 YES', '2 NO', '7') and if this is not provided we will look at:
  - 'cp1dad8dem' which provides a diagnosing like score through the use of a proxy. 
- Two cognitive test scores with cut offs <= 1.5 SD below the mean. 

**All else will be labeled as '2' for 'No Dementia."**
<br><br/>
The cognitive tests are based in three domains:
- ***Orientation*** 
    - President and VP First and Last names: **'cg1presidna1', 'cg1presidna3', 'cg1vpname1', 'cg1vpname3'**
    - Date, Month, Year, Day of the Week: **'cg1todaydat1' (Month), 'cg1todaydat2' (Day), 'cg1todaydat3' (Year), 'cg1todaydat4' (Day of the Week)**
    - Each correct answer gets a point; Total points out of 8.
    - Score cut point for <= 1.5 SD below the mean is <= 3 across all variables
- ***Memory*** 
    - Delayed Word Recall: **'cg1dwrdimmrc' (total Score)**
    - Immediate Word Recall: **'cg1dwrddlyrc' (total Score)** 
    - Total points out of 20
    - Score cut off is <= 3 across all variables
- ***Executive Functioning*** 
    - Clock Drawing Battery: **'cg1dclkdraw'**
    - Total points out of 5
    - Score cut off is <= 1 

***Each variable is used for all 9 years of this study and change only by the number value located within the variable name.**

## Repository Structure

### Data
- Holds the .txt and .csv files containing data from the NHATS rounds. 
- .txt files contain dictionaries in the structure of key = round and values = tuple of ( id #, label ).
### Labeling
- The Cleaning_Lableing notebook contains an explanation of the labeling strategy employed by NHATS and AI Crowd.
- THe python files contain the functions used with the notebook.
### Loading
- Contains the initial functions used to load images from S3 as well as the tabular data that corresponds with the images.

## Models
### Predicting Dementia directly from the images.
- For this model this idea is to pass in the images drawn by the patient to a CNN and try to determine whether or not the patient should be diagnosed with dementia.
### Predicting images score and passing to another model with other dementia test scores.
- For this model the idea is to pass the image to a CNN to determine a score of 1-5 for the image and then along with the scores of other dementia tests, pass that data to a second model to try to determine whether or not the patient should be diagnosed with dementia. 
