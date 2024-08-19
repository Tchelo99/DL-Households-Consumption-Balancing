from dataset_management.ukdale.ukdale_parameter import *
import pandas as pd
import time
import argparse
from dataset_management.ukdale.functions import load_dataframe

DATA_DIRECTORY = "G:/ukdale/"
SAVE_PATH = "G:/ukdale_processed/kettle/"
AGG_MEAN = 522
AGG_STD = 814


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except NameError:
        return False


def get_arguments():
    parser = argparse.ArgumentParser(
        description="sequence to point learning example for NILM"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIRECTORY,
        help="The directory containing the UKDALE data",
    )
    parser.add_argument(
        "--appliance_name",
        type=str,
        default="kettle",
        help="which appliance you want to train: kettle, microwave, fridge, dishwasher, washing machine",
    )
    parser.add_argument(
        "--aggregate_mean",
        type=int,
        default=AGG_MEAN,
        help="Mean value of aggregated reading (mains)",
    )
    parser.add_argument(
        "--aggregate_std",
        type=int,
        default=AGG_STD,
        help="Std value of aggregated reading (mains)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=SAVE_PATH,
        help="The directory to store the training data",
    )
    return parser.parse_args()


if is_notebook():

    class Args:
        data_dir = DATA_DIRECTORY
        appliance_name = "kettle"
        aggregate_mean = AGG_MEAN
        aggregate_std = AGG_STD
        save_path = SAVE_PATH

    args = Args()
else:
    args = get_arguments()

appliance_name = args.appliance_name
print(appliance_name)


def main():
    start_time = time.time()
    sample_seconds = 8
    validation_percent = 13
    nrows = None
    debug = False

    combined_train_val = pd.DataFrame()
    combined_test = pd.DataFrame()

    for h in params_appliance[appliance_name]["houses"]:
        print(f"Processing house {h}")

        mains_df = load_dataframe(args.data_dir, h, 1)
        app_df = load_dataframe(
            args.data_dir,
            h,
            params_appliance[appliance_name]["channels"][
                params_appliance[appliance_name]["houses"].index(h)
            ],
            col_names=["time", appliance_name],
        )

        mains_df["time"] = pd.to_datetime(mains_df["time"], unit="s")
        mains_df.set_index("time", inplace=True)
        mains_df.columns = ["aggregate"]

        app_df["time"] = pd.to_datetime(app_df["time"], unit="s")
        app_df.set_index("time", inplace=True)

        df_align = (
            mains_df.join(app_df, how="outer")
            .resample(str(sample_seconds) + "S")
            .mean()
            .fillna(method="backfill", limit=1)
        )
        df_align = df_align.dropna()

        df_align.reset_index(inplace=True)
        del df_align["time"]

        # Normalization
        mean = params_appliance[appliance_name]["mean"]
        std = params_appliance[appliance_name]["std"]

        df_align["aggregate"] = (
            df_align["aggregate"] - args.aggregate_mean
        ) / args.aggregate_std
        df_align[appliance_name] = (df_align[appliance_name] - mean) / std

        if h == params_appliance[appliance_name]["test_build"]:
            combined_test = pd.concat([combined_test, df_align], ignore_index=True)
        else:
            combined_train_val = pd.concat(
                [combined_train_val, df_align], ignore_index=True
            )

        del df_align

    # Split into train and validation
    val_len = int((len(combined_train_val) / 100) * validation_percent)
    combined_val = combined_train_val.tail(val_len).reset_index(drop=True)
    combined_train = combined_train_val.head(-val_len).reset_index(drop=True)

    # Save datasets
    combined_test.to_csv(args.save_path + appliance_name + "_test.csv", index=False)
    combined_val.to_csv(
        args.save_path + appliance_name + "_validation.csv", index=False
    )
    combined_train.to_csv(
        args.save_path + appliance_name + "_training.csv", index=False
    )

    print(f"Size of total training set is {len(combined_train) / 10**6:.4f} M rows.")
    print(f"Size of total validation set is {len(combined_val) / 10**6:.4f} M rows.")
    print(f"Size of total test set is {len(combined_test) / 10**6:.4f} M rows.")

    print("\nPlease find files in: " + args.save_path)
    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))


if __name__ == "__main__":
    main()
