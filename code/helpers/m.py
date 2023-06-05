import torch
import numpy as np

MANUALLY_LABLED_DATA_PATH = 'C:\\Users\\espebh\\Documents\\Thesis\\data\\labeled data\\manually annotated data'
ALB_LABLED_DATA_PATH = 'C:\\Users\\espebh\\Documents\\Thesis\\data\\labeled data\\augmented data'
DEVICE = torch.device('cpu')
FRAMES_PER_SECOND = 60
IMG_SHAPE = (1080, 1920, 3)
BATCH_SIZE = 32

LABEL_NAMES = ['ljaw', 'ujaw', 'eye', 'rjaw', 'rpec', 'head_body_intercept', 'dfin']
LABEL_MAP = dict([(y,x) for x,y in enumerate((LABEL_NAMES))])
NUM_KEYPOINTS = len(LABEL_NAMES)

CLASSES_T1 = ['background', 'Abijah_sl', 'Abijah_sr', 'Baara_sl', 'Baara_sr', 'Cephas_sl', 'Cephas_sr', 'Dan_sl', 'Dan_sr', 'Elam_sl', 'Elam_sr', 'Gabriel_sl', 'Gabriel_sr', 'Habazzianiah_sl', 'Habazzianiah_sr']
CLASSES_MAP_T1 = dict([(x,y) for x,y in enumerate((CLASSES_T1))])
NUM_CLASSES_T1 = len(CLASSES_T1)
FISH_NAMES_T1 = ['Abijah', 'Baara', 'Cephas', 'Dan', 'Elam', 'Gabriel', 'Habazzianiah']
CLASSES_TO_FISH_MAP_T1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])


CLASSES_T2 = ['background', 'Abagatha_sl', 'Abagatha_sr', 'Benhadad_sl', 'Benhadad_sr', 'Chenaniah_sl', 'Chenaniah_sr', 'Darius_sl', 'Darius_sr', 'Eliab_sl', 'Eliab_sr', 'Gemalli_sl', 'Gemalli_sr', 'Hammelech_sl', 'Hammelech_sr']
CLASSES_MAP_T2 = dict([(x,y) for x,y in enumerate((CLASSES_T2))])
NUM_CLASSES_T2 = len(CLASSES_T2)
FISH_NAMES_T2 = ['Abagatha', 'Benhadad', 'Chenaniah', 'Darius', 'Eliab', 'Gemalli', 'Hammelech']
CLASSES_TO_FISH_MAP_T2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])

CLASSES_T3 = ['background', 'Abiezer_sl', 'Abiezer_sr', 'Berachiah_sl', 'Berachiah_sr', 'Chimham_sl', 'Chimham_sr', 'Dathan_sl', 'Dathan_sr', 'Eliah_sl', 'Eliah_sr', 'Gemariah_sl', 'Gemariah_sr', 'Habakkuk_sl', 'Habakkuk_sr']
CLASSES_MAP_T3 = dict([(x,y) for x,y in enumerate((CLASSES_T3))])
NUM_CLASSES_T3 = len(CLASSES_T3)
FISH_NAMES_T3 = ['Abiezer', 'Berachiah', 'Chimham', 'Dathan', 'Eliah', 'Gemariah', 'Habakkuk']
CLASSES_TO_FISH_MAP_T3 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])

CLASSES_T4 = ['background', 'Abishur_sl', 'Abishur_sr', 'Benjamin_sl', 'Benjamin_sr', 'Chuza_sl', 'Chuza_sr', 'Deborah_sl', 'Deborah_sr', 'Enan_sl', 'Enan_sr', 'Gog_sl', 'Gog_sr', 'Hezron_sl', 'Hezron_sr']
CLASSES_MAP_T4 = dict([(x,y) for x,y in enumerate((CLASSES_T4))])
NUM_CLASSES_T4 = len(CLASSES_T4)
FISH_NAMES_T4 = ['Abishur', 'Benjamin', 'Chuza', 'Deborah', 'Enan', 'Gog', 'Hezron']
CLASSES_TO_FISH_MAP_T4 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])

CLASSES_T5 = ['background', 'Abel_sl', 'Abel_sr', 'Barabbas_sl', 'Barabbas_sr', 'Careah_sl', 'Careah_sr', 'Didymus_sl', 'Didymus_sr', 'Eleazar_sl', 'Eleazar_sr', 'Gether_sl', 'Gether_sr', 'Hadadezer_sl', 'Hadadezer_sr']
CLASSES_MAP_T5 = dict([(x,y) for x,y in enumerate((CLASSES_T5))])
NUM_CLASSES_T5 = len(CLASSES_T5)
FISH_NAMES_T5 = ['Abel', 'Barabbas', 'Careah', 'Didymus', 'Eleazar', 'Gether', 'Hadadezer']
CLASSES_TO_FISH_MAP_T5 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])

CLASSES_T6 = ['background', 'Abimelech_sl', 'Abimelech_sr', 'Bernice_sl', 'Bernice_sr', 'Cornelius_sl', 'Cornelius_sr', 'Doeg_sl', 'Doeg_sr', 'Ephaphras_sl', 'Ephaphras_sr', 'Guni_sl', 'Guni_sr', 'Hodaiah_sl', 'Hodaiah_sr']
CLASSES_MAP_T6 = dict([(x,y) for x,y in enumerate((CLASSES_T6))])
NUM_CLASSES_T6 = len(CLASSES_T6)
FISH_NAMES_T6 = ['Abimelech', 'Bernice', 'Cornelius', 'Doeg', 'Ephaphras', 'Guni', 'Hodaiah']
CLASSES_TO_FISH_MAP_T6 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])

CLASSES_T7 = ['background', 'Achsah_sl', 'Achsah_sr', 'Bezaleel_sl', 'Bezaleel_sr', 'Chilion_sl', 'Chilion_sr', 'Dorcas_sl', 'Dorcas_sr', 'Esarhaddon_sl', 'Esarhaddon_sr', 'Ginath_sl', 'Ginath_sr', 'Huppim_sl', 'Huppim_sr']
CLASSES_MAP_T7 = dict([(x,y) for x,y in enumerate((CLASSES_T7))])
NUM_CLASSES_T7 = len(CLASSES_T7)
FISH_NAMES_T7 = ['Achsah', 'Bezaleel', 'Chilion', 'Dorcas', 'Esarhaddon', 'Ginath', 'Huppim']
CLASSES_TO_FISH_MAP_T7 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])

CLASSES_T8 = ['background', 'Aedias_sl', 'Aedias_sr', 'Bunah_sl', 'Bunah_sr', 'Chelluh_sl', 'Chelluh_sr', 'Deuel_sl', 'Deuel_sr', 'Evi_sl', 'Evi_sr', 'Gershon_sl', 'Gershon_sr', 'Hymeneus_sl', 'Hymeneus_sr']
CLASSES_MAP_T8 = dict([(x,y) for x,y in enumerate((CLASSES_T8))])
NUM_CLASSES_T8 = len(CLASSES_T8)
FISH_NAMES_T8 = ['Aedias', 'Bunah', 'Chelluh', 'Deuel', 'Evi', 'Gershon', 'Hymeneus']
CLASSES_TO_FISH_MAP_T8 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])

CLASSES_T9 = ['background', 'Aaron_sl', 'Baasha_sr', 'Caiphas_sl', 'Daniel_sr', 'Ebed-Melech_sl', 'Aaron_sr', 'Daniel_sl', 'Ebed-Melech_sr', 'Gehasi_sr', 'Caiphas_sr', 'Hosea_sl', 'Baasha_sl', 'front', 'occluded', 'Hosea_sr', 'Gehasi_sl']
CLASSES_MAP_T9 = dict([(x,y) for x,y in enumerate((CLASSES_T9))])
NUM_CLASSES_T9 = len(CLASSES_T9)
FISH_NAMES_T9 = ['Aaron', 'Baasha', 'Caiphas', 'Daniel', 'Ebed-Melech', 'Hosea', 'Gehasi']
CLASSES_TO_FISH_MAP_T9 = np.array([[1, 6], [12, 2], [3, 10], [7, 4], [5, 8], [11, 15], [16, 9]])

exp_dat = { '280922_t1': [14.0292, 36.600385, 90.030031, 300], 
            '051022_t1': [15.5924, 39.933990, 94.223844, 333],
            '280922_t2': [14.0156, 40.273676, 91.225978, 301], 
            '051022_t2': [15.5679, 43.603615, 90.923769, 344],
            '280922_t3': [13.9932, 17.670506, 76.095653, 311], 
            '051022_t3': [15.5649, 38.462245, 86.869107, 319],
            '290922_t4': [12.2851, 45.803954, 90.870657, 363],
            '061022_t4': [13.4351, 38.865835, 88.609558, 365],
            '290922_t5': [12.3212, 38.056640, 90.431858, 371],
            '061022_t5': [13.4442, 38.908660, 88.658724, 334],
            '290922_t6': [12.3295, 34.190574, 88.977146, 335],
            '061022_t6': [13.4536, 38.581455, 89.823750, 329],
            '280922_t7': [15.8902, 44.868576, 90.211965, 218],
            '051022_t7': [17.7105, 48.610588, 88.855440, 215],
            '280922_t8': [15.9239, 42.851896, 87.953265, 204],
            '051022_t8': [17.7525, 47.286791, 88.154413, 231],
            '280922_t9': [15.8689, 44.738884, 93.572209, 213],
            '051022_t9': [17.7342, 48.750417, 94.023570, 213]}