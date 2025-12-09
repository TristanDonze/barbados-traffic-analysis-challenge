import json
import pickle


def save_data_to_json(data, output_path):
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving JSON: {e}")


def load_data_from_json(input_path):
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []


def save_data_to_pickle(data, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    # print(f"Data saved to {output_path}")


def load_data_from_pickle(input_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    return data
