# Reproducing trained model
```bash
python train_model.py --model SmallVGG --batch_size <batchsize> --n_epoch <nepoch> --learning_rate <lr>
```

If you ommit batch_size it will default to 8, n_epoch defaults to 10 and learning_rate defaults to 0.001.

This will create a folder **date-time** in the models directory containing:
* A model called **model_earlystop.pt**. This is the model used for submission and is the model saved when 
the training error becomes 5% lower than validation error (to avoid overfitting).
* A model called **model.pt** which is the model obtained at the end of training.
* Loss and classification error images for training and validation.
* A binary file (pkl) with model parameters. (FYI : It also accidentally, but lazily unfixed, contains the actual model.)

## Running eval.py
In order to get the submission file, you need to run eval.py by changing the model's path by hand 
(it was not put into a script, sorry). So you need to run eval.py in a dev environment. 

# Getting feature maps
Getting feature maps has only been implemented for SmallVGG and SmallVGG_5K. But to get it : 

```bash
python feature_maps.py <model_folder> <model_name> <model_file_name>

ex:
python feature_maps.py models/<model_folder> SmallVGG model.pt
```
