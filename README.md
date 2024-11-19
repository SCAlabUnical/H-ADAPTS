# Dynamic hashtag recommendation in social media with trend shift detection and adaptation

## How it works
**H-ADAPTS** *(Hashtag recommendAtion by Detecting and adAPting to Trend Shifts)* is a BERT-based hashtag recommendation methodology designed to operate in dynamic contexts characterized by the continuous evolution of social trends and hashtags over time.
</br>Specifically, H-ADAPTS leverages Apache Storm to detect shifts in the main trends and topics underlying social media conversation, and an ad-hoc re-training strategy to adapt the recommendation model to those shifts rapidly and effectively.

The methodology performs the following steps:
- **Model bootstrap**: all necessary components are initialized.
- **Trend shift detection**: the real-time stream of social posts is processed by Storm to detect a trend shift, i.e., a significant deviation of current online conversation from previous history compared to its past trends and topics.
- **Model adaptation**: upon detecting a trend shift, the current recommendation model is asynchronously updated, re-aligning it with the current trends and topics.
- **Hashtag recommendation**: the current recommendation model is used for recommending a set of hashtags for a query post provided by the user.

> ***H-ADAPTS effectively addresses the high dynamicity of social media conversation by identifying changes in the main trending hashtags and topics, as well as semantic shifts in the meaning of existing hashtags, enabling an accurate recommendation of hashtags in real-life scenarios.***

## Prerequisites
- Java 8
- Python 3.7
- Apache Maven

## Installation
Install Python requirements:
```bash
$ pip install -r requirements.txt
$ python -m spacy download en_core_web_lg
```
Build the project with Maven:
```bash
$ mvn package
```
Create the datasets directory:
```bash
$ mkdir datasets
```

## Dataset Format
The dataset must follow the standard format of the [read_json](https://pandas.pydata.org/pandas-docs/version/1.1.3/reference/api/pandas.read_json.html) function of the Pandas library (orient='columns'), with the following columns:
- **date (long)**: unix timestamp representing the date of each tweet.
- **text (string)**: text of the tweet.
- **is_retweet (bool)**: flag that is True only for retweets.

## How to run
Place the dataset to be processed in the ```datasets/``` directory and modify ```src/main/resources/constants.py``` accordingly:
- The **DATASET_NAME** parameter with the dataset file name.
- The **SPLIT_DATE** parameter with the timestamp for splitting the dataset into static and real-time portions.

It is recommended to use 3 shell sessions, open in the project directory, working in parallel (to execute and interrupt executions in a controlled manner).

**SHELL 1**
Start the Python component:
```bash
$ source run_python.sh
```

**SHELL 2**
Open the log file:
```bash
$ tail -f results/log.txt
```

**SHELL 3** (following the display of the message *"start storm topology"* in SHELL 1)
Start the java-storm component:
```bash
$ source run_java.sh
```

Upon completion of dataset processing by Storm, in **SHELL 3** the message **SPOUT EXHAUSTED** will be displayed.
To terminate the execution, simply interrupt the execution of **SHELL 3**.

The execution results are contained in the following files (located in the ```results``` directory):
- **results.txt**: contains the results of the top-k hashtags predictions.
- **evaluation_results.txt**: contains the expansion results.
- **log.txt**: contains the execution log.

# System configuration
The configuration parameters are all located in the file ```src/main/resources/constants.py```. Most important ones include:
- **DATASET_NAME**: the name of the dataset to be processed (in the ```datasets/``` directory).
- **THRESHOLD**: the occurrence threshold that a hashtag must exceed to be considered trending.
- **TUMBLING_WINDOW_LENGTH**: length of the tumbling window.
- **LOGIC_WINDOW_LENGTH**: length of the logic window.
- **PERC_TEST**: percentage of the validation set.
- **MAX_EPOCHS**: number of epochs for training.
- **DO_STATIC_TRAIN**: *True* if static training is desired, *False* if you want to start directly from the real-time part, loading weights from the directory ```src/main/resources/static_train_save_files```.
- **REALTIME**: *True* if you want to enable real-time trend shift detection and model adaptation, *False* otherwise.
- **RT_LR**: the learning rate of real-time training.
- **SPLIT_DATE**: the date for splitting the dataset into static and real-time portions.
