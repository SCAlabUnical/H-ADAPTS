# BERT-based hashtag recommendation in social media with real-time trend shift detection

## How it works
**H-ADAPTS** *(Hashtag recommendAtion by Detecting and adAPting to Trend Shifts)* is a BERT-based hashtag recommendation methodology designed to operate in dynamic contexts characterized by the continuous evolution of social trends and hashtags over time.
</br>Specifically, H-ADAPTS leverages Apache Storm to detect shifts in the main trends and topics underlying social media conversation, and an ad-hoc re-training strategy to adapt the recommendation model to those shifts rapidly and effectively.

The methodology performs the following steps:
- *Model bootstrap*: all necessary components are initialized.
- *Trend shift detection*: the real-time stream of social posts is processed by Storm to detect a trend shift, i.e., a significant deviation of current online conversation from previous history compared to its past trends and topics.
- *Model adaptation*: upon detecting a trend shift, the current recommendation model is asynchronously updated, re-aligning it with the current trends and topics.
- *Hashtag recommendation*: the current recommendation model is used for recommending a set of hashtags for a query post provided by the user.

H-ADAPTS effectively addresses the high dynamicity of social media conversation by identifying changes in the main trending hashtags and topics, as well as semantic shifts in the meaning of existing hashtags, enabling an accurate recommendation of hashtags in real-life scenarios.

## Reproducibility
This repository hosts all the code necessary for reproducibility purposes.

