## This project is about using machine learning to make images generate music 
### environment for this project
1. python >= 3.9, torch >= 2.0
2. need install clip, transformers, audiocraft


### Introduction to the project structure
1. data: about the dataset and vocab, in this project, we use LAION as the dataset, the link is: https://laion.ai/blog/laion-400-open-dataset/
2. dataProcess
   1. getVocab.py: create vocab
   2. precess.py: precess data 
   3. getImgTextData.py: create dataset about image and text features
   4. getTextFeature.py: create textfeature dataset
3. models: about models to upload in my github
4. module: about how to design models
   1. image2Text.py: create Image2Textfeature model
   2. textFeature2Text.py: create textFeature2Text model
   3. loadMusicGen.py: load musicGen model
5. image2music.py: main function that input a image then generate music

### update 
we will update this project in github if we have some updates, the link is:https://github.com/Rachel1111

