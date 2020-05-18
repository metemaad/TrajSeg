import pandas as pd
from Trajlib2.SegmentationAlgorithms.SWS.sws import SWS
import warnings
warnings.filterwarnings("ignore")


ows=SWS()
res=ows.experiment_window_size2()
print(pd.DataFrame(res))