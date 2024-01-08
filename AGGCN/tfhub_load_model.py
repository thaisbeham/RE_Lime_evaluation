import tensorflow_hub as hub
import tensorflow as tf
import os

def save_tfhub_model(module_url, save_path):
    # Load the TF Hub model
    model = hub.load(module_url)
    print(f"Module {module_url} loaded")

    # Save the model to the specified path
    tf.saved_model.save(model, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Set the TF Hub module URL and the desired save path
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    save_path = "../tf_hub_model/"

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the TF Hub model
    save_tfhub_model(module_url, save_path)
