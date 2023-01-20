#internal imports
from suae import sUAE
import config

if __name__ == "__main__":
    suae = sUAE()
    suae.load()
    # suae.generate_losses()
    suae.evaluate()
    suae.show_evaluation_results()
    suae.generate_evaluation_plot()