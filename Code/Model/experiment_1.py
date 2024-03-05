import pickle
from pathlib import Path
import os


stored_folder = Path(os.path.abspath('')).parent.parent / "data" / "processed" / "cleaned_df.pkl"
input_file = open(stored_folder, "rb")
cleaned_data = pickle.load(input_file)


if __name__ == "__main__":
    print(cleaned_data)