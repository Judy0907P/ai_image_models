 # 04 Experiment Tracking

To improve models, we need to run experiments. It is important to record:
- which code we used to train
- what where the parameters/hyper-paramters
- what data the model was trained on
as well as:
- how the training went: loss over time, system metrics (e.g gpu RAM)
- test results, in our case FID

If we dont track those things, it's super easy to get confused and not attribute results to the correct causes.
Also super easy to write bugs, and not realize.

sweep we ran in class: https://wandb.ai/armandpl/ai_image_models-04_experiment_tracking/sweeps/9fc3lvdd

Homework:
- train a neural net on your dataset, with experiment tracking
- vary a hyper-parameter of your choice, compare two runs with the FID metric, conclude which hyper-parameter value is the best
- if using weights and biaes, make your project public and send me the link. if not using wandb (e.g tensorboard), we'll take a look at it in class :)
- (optional) track additionnal experiment variables, e.g dataset version
