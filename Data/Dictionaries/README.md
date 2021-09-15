# Dementia Label Dict folder 
- Contains all the dictionaries that we created to map the image data to the labels for 0 - possible dementia, 1 - probable dementia, 2 - no dementia. 
- There are 5 txt dictionary files in this folder:
  - train_dict_bal, val_dict_nobal, test_dict_bal are all dictionaries containing the hybrid AI Crowd and NHATs labeling system containing rounds 1-10 SPIDs and the labels. The data was randomly split into train, val, test at a 90-5-5 split. These dictionaries contained a train set that is balanced to 8000 items per class. The validation and test sets are not balanced.  
    - train_dict_bal: 24000 items
    - val_dict_nobal: ~2500 items
    - test_dict_nobal: ~2500 items


# Score Dict folder
- Contains a dictionary of the clock SPIDs for rounds 0-10 and the clock score labels 0 - 5 from variable cg1dclkdraw. Score labels are: 
  - 0 - Not recognizable as a clock
  - 1 - Severely distorted depiction of a clock
  - 2 - Moderately distorted depiction of a clock
  - 3 - Mildly distorted depiction of a clock
  - 4 - Reasonably accurate depiction of a clock
  - 5 - Accurate depiction of a clock (circular or square)
