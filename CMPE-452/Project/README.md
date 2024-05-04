# EuclidMLP
### What is this repo?
Alternative implementaion of EuclidNets (https://arxiv.org/abs/2212.11803) using a simple MLP to classify MNIST data as well as an MLP autoencoder to reconstruct MNIST digits.

### How does it work?
Instead of using a teacher-student network and an operation homotopy as is used in the EuclidNet paper, we implemented EuclidNet in the same style as AdderNet (https://arxiv.org/abs/1912.13200) with backpropogation. We created custom torch.Autograd.Function forward and backward methods for training. The code for these methods can be found in the mnist_model.py file. The rest of the code is fairly standard machine learning workflow.

### How can I train and test one of these models?
1. First run python ./requirements.txt to ensure you have all the right dependencies
2. Run one of the scripts from the /scripts folder, either train_autoencoder.txt or train_classifier.txt
3. If you are unsure what a certain flag does then pass the help argument to one of the flags in the training scripts for a description
4. If you want to change a parameter other than the ones listed in the ./src/functions.py parse_args() function, then you will have to change them in the mnist_autoencoder.py or mnist_classifier.py file
5. To see the results of the autoencoder run python ./src/mnist_reconstruct.py -l path_to_your_weight_file 

