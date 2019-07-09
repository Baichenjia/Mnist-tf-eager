# Mnist-tf-eager
Mnist classification with tensorflow eager execution

## train.py
正常训练 mnist

## train_scope.py
add a new function "_vars" in model, enable to compute gradient with a part of trainable_variables, by using "model._vars(scope)"
