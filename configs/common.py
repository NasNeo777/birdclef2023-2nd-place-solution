from types import SimpleNamespace
import glob
import torch
import warnings

cfg = SimpleNamespace(**{})
cfg.WANDB_API_KEY = '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'
cfg.infer_duration = 5
cfg.valid_duration = 60
cfg.valid_ratio = 0.1
cfg.val_matmul_precision = "high"
cfg.max_valid_batch_size = 8
cfg.num_workers = 4
cfg.val_num_workers = 2
cfg.num_sanity_val_steps = 0
cfg.accumulate_grad_batches = 1
cfg.cuda_alloc_conf = "expandable_segments:True"
cfg.gradient_clip_val = 1.0
cfg.gradient_clip_algorithm = "norm"
cfg.label_smoothing = 0.1
cfg.weight_decay = 1e-3
cfg.SR = 32000
cfg.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.primary_label_col = 'primary_label'
cfg.secondary_labels_col = 'secondary_labels'
cfg.background_duration_thre = 60
cfg.pseudo_label_path = 'inputs/pseudo_label'
cfg.hand_label_path = 'inputs/hand_label'

cfg.train_data = '/media/nasneo/AI/datas/bird/train.csv'
cfg.train_dir = '/media/nasneo/AI/datas/bird/train_audio'
cfg.test_dir = '/media/nasneo/AI/datas/bird/test_audio'

cfg.birdclef2021_nocall = glob.glob("inputs/background_noise/birdclef2021_nocall/*")
cfg.birdclef2020_nocall = glob.glob("inputs/background_noise/birdclef2020_nocall/*")
cfg.freefield = glob.glob("inputs/background_noise/freefield/*")
cfg.warblrb = glob.glob("inputs/background_noise/warblrb/*")
cfg.birdvox = glob.glob("inputs/background_noise/birdvox/*")
cfg.rainforest = glob.glob("inputs/background_noise/rainforest/*")
cfg.environment = glob.glob("inputs/background_noise/environment/*")
cfg.background_noise_cache_size = 256

warnings.filterwarnings(
    "ignore",
    message=r".*had to be resampled from .* This hurt execution time\.",
)

cfg.bird_cols_train = ['1161364', '116570', '1176823', '1595929', '209233', '22930', '22956', '22961', '22967', '22973', '22983', '22985', '23150', '23154', '23158', '23176', '23724', '24279', '24285', '24287', '24321', '244024', '25092', '25214', '326272', '41970', '43435', '47144', '476521', '516975', '555123', '555145', '555146', '64898', '65377', '65380', '66971', '67107', '67252', '70711', '738183', '74113', '74580', '760266', 'ashgre1', 'astcra1', 'bafcur1', 'baffal1', 'banana', 'barant1', 'batbel1', 'baymac', 'bbwduc', 'bcwfin2', 'bkcdon', 'bkhpar', 'blchaw1', 'blheag1', 'blttit1', 'bncfly', 'bobfly1', 'brcmar1', 'brnowl', 'bucmot4', 'bucpar', 'bufpar', 'bunibi1', 'burowl', 'camfli1', 'chacha1', 'chbmoc1', 'chobla1', 'chvcon1', 'cibspi1', 'coffal1', 'compau', 'compot1', 'crbthr1', 'crebec1', 'dwatin1', 'epaori4', 'eulfly1', 'fabwre1', 'fepowl', 'ficman1', 'flawar1', 'fotfly', 'fusfly1', 'gilhum1', 'giwrai1', 'glteme1', 'grasal3', 'greani1', 'greant1', 'greela', 'grekis', 'grepot1', 'gretho2', 'greyel', 'grfdov1', 'grhtan1', 'gycwor1', 'horscr1', 'houspa', 'hyamac1', 'larela1', 'lesela1', 'lesgrf1', 'limpki', 'linwoo1', 'litcuc2', 'litnig1', 'mabpar', 'magant1', 'magtan2', 'masgna1', 'nacnig1', 'ocecra1', 'oliwoo1', 'orbtro3', 'orwpar', 'osprey', 'pabspi1', 'palhor3', 'paltan1', 'phecuc1', 'picpig2', 'pirfly1', 'plasla1', 'platyr1', 'plcjay1', 'pluibi1', 'purjay1', 'pvttyr1', 'ragmac1', 'rebscy1', 'recfin1', 'redjun', 'relser1', 'rinkin1', 'rivwar1', 'roahaw', 'rubthr1', 'rufcac2', 'rufcas2', 'rufgna3', 'rufhor2', 'rufnig1', 'ruftho1', 'ruftof1', 'rumfly1', 'ruther1', 'rutjac1', 'sabspa1', 'saffin', 'saytan1', 'scadov1', 'schpar1', 'scther1', 'shcfly1', 'shshaw', 'shtnig1', 'sibtan2', 'smbani', 'smbtin1', 'sobcac1', 'sobtyr1', 'socfly1', 'sofspi1', 'souant1', 'soulap1', 'souscr1', 'spbant3', 'spispi1', 'sptnig1', 'squcuc1', 'stbwoo2', 'strcuc1', 'strher2', 'strowl1', 'swthum1', 'swtman1', 'tattin1', 'thlwre1', 'toctou1', 'trokin', 'trsowl', 'undtin1', 'varant1', 'watjac1', 'wesfie1', 'wfwduc1', 'whbant2', 'whbwar2', 'whiwoo1', 'whlspi1', 'whnjay1', 'whtdov', 'whwpic1', 'y00678', 'yebcar', 'yebela1', 'yecmac', 'yecpar', 'yehcar1', 'yeofly1']

cfg.bird_cols_pretrain = cfg.bird_cols_train

common_cfg = cfg
