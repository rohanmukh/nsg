#!/bin/bash
filename="/home/ewen/data/nag/train_test_split/github-java-files-train.txt"
filename_parsed='parsed-files.txt'
files_handled_log='files-handled.log'
stderr_file='error.log'

#Delete the Last Log
rm -r $files_handled_log $stderr_file JSONFiles 
# Appends 'java_projects' to beginning of each file_ptr

sed 's/^\./\/home\/ewen\/data\/nag\/java_projects/g' $filename > $filename_parsed

mkdir JSONFiles
java -jar /home/ewen/code/nag/data_extraction/tool_files/batch_dom_driver/target/batch_dom_driver-1.0-jar-with-dependencies.jar $filename_parsed /home/ewen/code/nag/data_extraction/java_compiler/config-full.json > $files_handled_log 2>$stderr_file
# Need to stich the jsons after this step
#locate all json files easily by this

sed -i 's/\//_/g' $filename_parsed
sed -i 's/^/JSONFiles\//g' $filename_parsed
sed -i 's/$/\.json/g' $filename_parsed

# Use the prebuilt merge API
export PYTHONPATH="/home/ewen/code/nag"
#python -u data_extraction/scripts/merge_manipulate.py $filename_parsed --output_folder temp_data

#delete temp files

#rm -r $filename_parsed $files_handled_log $stderr_file #temp_data

