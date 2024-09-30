## Extract dataframes form rosbags and save them as pickle files

The first step is to extract the dataframes from the rosbags and save them as pickle files

```bash
cd social_evaluation
python3 bag_extraction.py -h # to see the help
```

The script `bag_extraction.py` extracts the dataframes from the rosbags and saves them as pickle files. The script takes as input the path to the rosbag file and the path to the output directory where the pickle files will be saved. The script also takes as input the name of the topics that will be extracted from the rosbag file. The topics are the following:

..