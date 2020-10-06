# Event Entity Coreference

Event Entity Coreference with SpanBERT on ECB+.  
More details will be added later.

## Dataset
---
More details about the ECB+ dataset can be found [here](http://www.newsreader-project.eu/results/data/the-ecb-corpus/).  
To download the dataset, click [here](kyoto.let.vu.nl/repo/ECB+_LREC2014.zip).

## Requirements
---
Python 3.8.5  
Pytorch  
Transformers  
Exact versions and other details are mentioned in `requirements.txt`

## Files
---
  main.py - run this file with required options  
  src / model.py - has the pytorch model class  
  src / train.py - has methods for training, evaluation (will rename this later)  
  src / dataloader.py - has pytorch dataset class   
  src / utils.py - has all the misc functions used for data processing and other stuff.

## Usage 
---
`python main.py -h` for usage details  

More details regarding the repository will be added soon.
