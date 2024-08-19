# Deep Learning-Based Classification of Households for Domestic Consumption Balancing

The household sector stands at the crossroads of energy consumption, environmental impact, and societal well-being. As we navigate the complexities of a rapidly changing world, understanding how households utilize energy becomes paramount. Household electricity consumption is a significant contributor to overall energy demand, and optimizing its usage is crucial for both economic and environmental reasons.

Energy disaggregation is a technique used to break down a household’s total electricity consumption into individual appliance-level usage. This allows for more detailed insights into energy consumption patterns, enabling better management and efficiency improvements. Among the various approaches to energy disaggregation, sequence-to-point (seq2point) learning has emerged as a powerful method for non-intrusive load monitoring (NILM).

Seq2point learning is a generic and simple framework for NILM [1-2]. It learns a mapping from the mains window Y to the midpoint x of the corresponding appliance. From a probabilistic perspective, seq2point learns a conditional distribution p(x|Y) (see details in [2]). The seq2seq learning proposed in [2] extends this approach by learning a mapping from sequence to sequence.

Seq2point learning is flexible, allowing you to choose any architecture, including CNN, RNN, and AutoEncoders, or even other models like logistic regression, SVM, and Gaussian Process regression.

## Requirements

* To set up the environment for running this project, you need to create a virtual environment using Python 3.5-3.8. The required libraries, such as TensorFlow and Keras, are specified in the environment.yml file. To create the environment, simply run:

```bash
conda env create -f environment.yml
```

* GPU support is highly recommended for training the models due to the computationally intensive nature of deep learning tasks. Ensure that your system is equipped with a compatible GPU and the necessary drivers.

## Directory Tree

This project allows you to use the Sequence to Point network, enabling you to prepare datasets from commonly used sources in NILM, train the network, and test it. The target appliances considered are kettle, microwave, fridge, dishwasher, and washing machine.

Directory tree:
```bash
Deep-Learning-Based-Classification-of-Households-for-Domestic-Consumption-Balancing/
├── appliance_data.py
├── data_feeder.py
├── dataset_management/
│   ├── functions.py
│   ├── refit/
│   │   ├── dataset_generator
│   │   ├── appliance_data
│   └── ukdale/
│       ├── create_trainset_ukdale
│       └── ukdale_parameters.py
├── environment.yml
├── images/
│   ├── model.png
│   ├── s2p.png
├── model.png
├── model_structure.py
├── README.md
├── remove_space.py
├── saved_models/
├── seq2point_test.py
├── seq2point_train.py
├── test_main.py
└── train_main.py
```
## How to Run the Code

To use the code:

* Prepare the Dataset: You can create your dataset using REFIT, UK-DALE, or REDD data. Download the raw data from the original sources

* Train the Model: Run the train_main.py script to start training the seq2point model. Adjust parameters in the script as needed.

* Test the Model: Once training is complete, use the test_main.py script to evaluate the model's performance on the test dataset.

## References

[1] DIncecco, Michele, Stefano Squartini, and Mingjun Zhong. "Transfer Learning for Non-Intrusive Load Monitoring." arXiv preprint arXiv:1902.08835 (2019).

[2] Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, and Charles Sutton. "Sequence-to-point learning with neural networks for nonintrusive load monitoring." Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), Feb. 2-7, 2018.
