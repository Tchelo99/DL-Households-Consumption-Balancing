import argparse
from remove_space import remove_space
from seq2point_test import Tester

# Allows a model to be tested from the terminal.

# You need to input your test data directory
test_directory = "G:/kettle/kettle_test_.csv"

def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(
        description="Test a pre-trained neural network for energy disaggregation."
    )

    parser.add_argument(
        "--appliance_name",
        type=remove_space,
        default="kettle",
        help="The name of the appliance to perform disaggregation with. Default is kettle. Available are: kettle, fridge, dishwasher, microwave.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="The batch size to use when testing the network. Default is 1000.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=1000000,
        help="The number of rows of the dataset to take testing data from. Default is 10000.",
    )
    parser.add_argument(
        "--algorithm",
        type=remove_space,
        default="seq2point",
        help="The pruning algorithm of the model to test. Default is none.",
    )
    parser.add_argument(
        "--network_type",
        type=remove_space,
        default="default",
        help="The seq2point architecture to use. Available are: default, dropout, reduced, and reduced_dropout.",
    )
    parser.add_argument(
        "--input_window_length",
        type=int,
        default=599,
        help="Number of input data points to network. Default is 599.",
    )
    parser.add_argument(
        "--test_directory",
        type=str,
        default=test_directory,
        help="The directory for testing data.",
    )

    args, unknown = parser.parse_known_args()

    # Access the parsed arguments
    saved_model_dir = f"saved_models/{args.appliance_name}_seq2point_model.h5"

    # The logs including results will be recorded to this log file
    log_file_dir = f"saved_models/{args.appliance_name}_seq2point_{args.network_type}.log.h5"


    tester = Tester(
        args.appliance_name,
        args.algorithm,
        args.crop,
        args.batch_size,
        args.network_type,
        args.test_directory,
        saved_model_dir,
        log_file_dir,
        args.input_window_length,
    )
    tester.test_model()

if __name__ == "__main__":
    main()
