# Comma Separated Files
The files contained here were created along the exploration phase of this project. Below explains each file and relevance to the project goals.
  - <b>round_data.csv</b>: We initially created a csv file of just the rounds, SPIDs and Healthscore and clock drawing scores. We realized that using this data was insufficient for the purpose of labeling since we adapted our labeling process to include the AI crowd method of labeling as well as the NHATs dementia classification. Combining the two into a hybrid label process to meet our goals. 
  - <b>hats_round_data.csv</b>: This csv includes all the features used to proceed with creating our data labels using the NHATs dementia classification:
    - spid
    - cg1dclkdraw
    - hc1disescn9 
    - cg1presidna1 
    - cg1presidna3 
    - cg1vpname1 
    - cg1vpname3 
    - cg1todaydat1 
    - cg1todaydat2 
    - cg1todaydat3 
    - cg1todaydat4 
    - cg1dwrdimmrc 
    - cg1dwrddlyrc 
    - round 
    - cp1dad8dem

  - <b>cleaned_nhat_data.csv</b>: is a cleaned version of the hats_round_data.csv data file sans labels that contains SPIDs of type 'string' and all the rest of the features are of type int or float. 

# Dictionaries
  - We separated our CSV files from the dictionaries for organization purposes
