#internal imports
import config
config.create_data_generators = True
from suae import sUAE

def train():
    suae = sUAE()
    suae.model_summary()
    suae.train()
    # suae.load()
    # suae.evaluate_testing_losses()
    suae.save()

if __name__ == "__main__":
    print(config.train_dataset_folder)
    train()