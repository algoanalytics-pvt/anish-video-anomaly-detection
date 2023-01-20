#internal imports
from suae import sUAE
import config

if __name__ == "__main__":
    suae = sUAE()
    # suae.model_summary()
    suae.load()
    # suae.generate_losses_sequential()
    suae.read_json_labels()
    suae.evaluate_sequential()
    suae.show_evaluation_results()
