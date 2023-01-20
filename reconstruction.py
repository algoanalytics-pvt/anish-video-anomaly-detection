# internal imports
from suae import sUAE
import config

input_filename = ""
input_reshaped_filename = ""
output_filename = ""


def reconstruct():
    suae = sUAE()
    suae.load()
    suae.generate_reconsrtuction(
        input_filename, input_reshaped_filename, output_filename
    )


if __name__ == "__main__":
    reconstruct()
