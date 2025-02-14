#!/bin/bash
curl -L -o ./data/data.zip\
  https://www.kaggle.com/api/v1/datasets/download/kazanova/sentiment140
curl -L -o ./data/rusentitweet_full.csv\
  https://raw.githubusercontent.com/sismetanin/rusentitweet/main/rusentitweet_full.csv
