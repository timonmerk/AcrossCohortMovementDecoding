import mne_bids
import mne
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from bids import BIDSLayout
from py_neuromodulation import nm_across_patient_decoding

PATH_BIDS =  "C:\Users\ICN_admin\Documents\Datasets\Beijing_new"
layout = BIDSLayout(PATH_BIDS)
subjects = layout.get_subjects()
run_files = []

for sub in subjects:
    if sub != "FOG013":
        try:
            run_files.append(layout.get(subject=sub, task='ButtonPress', extension='.vhdr')[0])
        except:
            pass

nm_across_patient_decoding.AcrossPatientRunner()