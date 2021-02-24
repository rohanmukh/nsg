****************** DATA EXTRACTION*********************

For data extraction, go to 
> cd data_extraction/tool_files/batch_dom_driver

RUN
> mvn package

This will create our java parser jar

once finished go to data_extraction/scripts/

RUN

>./drive_the_dom.sh $FILEPATH

$FILEPATH is the list of the locations of all JAVA files in your local machine

This will dump data folder in the current directory
*************************************************************
************************* TRAINING *****************************************

For training, go to trainer

RUN
> python3 -u train.py --config config.json --save save_train --data data_folder


