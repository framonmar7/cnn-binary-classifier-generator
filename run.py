import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

from core.pipeline import run

if __name__ == "__main__":
    run()
    