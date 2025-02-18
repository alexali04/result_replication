Result replication



# RNN

Data: 

```shell
curl -L -o ~/personal_proj/result_replication/RNN/data/the-bards-best-a-character-modeling-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/thedevastator/the-bards-best-a-character-modeling-dataset

unzip ~/personal_proj/result_replication/RNN/data/the-bards-best-a-character-modeling-dataset.zip
```

# RNN Commands

```shell
python fitting.py --train_path=./data/train.csv --epochs=1 --name=indeera
```

