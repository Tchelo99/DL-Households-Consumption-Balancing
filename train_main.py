import argparse
import os
from remove_space import remove_space
from seq2point_train import Trainer
import pandas as pd

training_directory = "G:/ukdale_processed/washingmachine/washingmachine_training.csv"
validation_directory = "G:/ukdale_processed/washingmachine/washingmachine_validation.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Train sequence-to-point learning for energy disaggregation."
    )

    parser.add_argument(
        "--appliance_name",
        type=remove_space,
        default="washingmachine",
        help="The name of the appliance to train the network with. Default is kettle. Available are: kettle, fridge, washing machine, dishwasher, and microwave.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="The batch size to use when training the network. Default is 1000.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=10000,
        help="The number of rows of the dataset to take training data from. Default is 10000.",
    )
    parser.add_argument(
        "--network_type",
        type=remove_space,
        default="seq2point",
        help="The seq2point architecture to use.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs. Default is 10."
    )
    parser.add_argument(
        "--input_window_length",
        type=int,
        default=599,
        help="Number of input data points to network. Default is 599.",
    )
    parser.add_argument(
        "--validation_frequency",
        type=int,
        default=1,
        help="How often to validate model. Default is 2.",
    )
    parser.add_argument(
        "--training_directory",
        type=str,
        default=training_directory,
        help="The dir for training data.",
    )
    parser.add_argument(
        "--validation_directory",
        type=str,
        default=validation_directory,
        help="The dir for validation data.",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="saved_models/washingmachine_uk_seq2point_model.h5",
        help="The path to save the trained model.",
    )

    args, unknown = parser.parse_known_args()

    print(f"Model will be saved to: {args.save_model_path}")

    # Ensure directories exist
    os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)

    # Check if files exist
    if not os.path.exists(args.training_directory):
        raise FileNotFoundError(f"Training file not found: {args.training_directory}")
    if not os.path.exists(args.validation_directory):
        raise FileNotFoundError(
            f"Validation file not found: {args.validation_directory}"
        )

    # Load and inspect data
    try:
        print(f"Reading training file: {args.training_directory}")
        training_data = pd.read_csv(args.training_directory)
        print(f"Training data sample:\n{training_data.head()}")

        print(f"Reading validation file: {args.validation_directory}")
        validation_data = pd.read_csv(args.validation_directory)
        print(f"Validation data sample:\n{validation_data.head()}")
    except pd.errors.EmptyDataError:
        raise ValueError("One of the CSV files is empty or improperly formatted.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing one of the CSV files: {e}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {e}")

    print(
        f"Training data loaded successfully with {training_data.shape[0]} rows and {training_data.shape[1]} columns."
    )
    print(
        f"Validation data loaded successfully with {validation_data.shape[0]} rows and {validation_data.shape[1]} columns."
    )

    # Optionally print the first few rows to verify
    print("First few rows of training data:")
    print(training_data.head())
    print("First few rows of validation data:")
    print(validation_data.head())

    trainer = Trainer(
        args.appliance_name,
        args.batch_size,
        args.crop,
        args.network_type,
        args.training_directory,
        args.validation_directory,
        args.save_model_path,
        epochs=args.epochs,
        input_window_length=args.input_window_length,
        validation_frequency=args.validation_frequency,
    )
    trainer.train_model()


if __name__ == "__main__":
    main()
