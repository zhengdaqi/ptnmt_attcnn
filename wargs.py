dataset = 'S' # S for 40k, M for 1.2M, L for wmt en-de

# Maximal sequence length in training data
#max_seq_len = 10000000
max_seq_len = 50

'''
Embedding layer
'''
# Size of word embedding of source word and target word
src_wemb_size = 512
trg_wemb_size = 512

'''
Encoder layer
'''
# Size of hidden units in encoder
enc_hid_size = 512

'''
Attention layer
'''
# Size of alignment vector
align_size = 512

'''
Decoder layer
'''
# Size of hidden units in decoder
dec_hid_size = 512
# Size of the output vector
out_size = 512

# Directory to save model, test output and validation output
dir_model = 'wmodel'
dir_valid = 'wvalid'
dir_tests = 'wtests'

# Validation data
val_shuffle = True
# Training data
train_shuffle = True
batch_size = 80
sort_k_batches = 20

# Data path
dir_data = 'data/'
train_prefix = 'train'
train_src_suffix = 'src'
train_trg_suffix = 'trg'
dev_max_seq_len = 10000000

# Dictionary
word_piece = False
src_dict_size = 30000
trg_dict_size = 30000
src_dict = dir_data + 'src.dict.tcf'
trg_dict = dir_data + 'trg.dict.tcf'

inputs_data = dir_data + 'inputs.pt'

with_bpe = False
with_postproc = False
copy_trg_emb = False
# Training
max_epochs = 20
epoch_shuffle = False
epoch_shuffle_minibatch = 1

small = False
eval_small = False
epoch_eval = False
final_test = False
char = False

if dataset == 'S':
    src_wemb_size = 256
    trg_wemb_size = 256
    enc_hid_size = 256
    align_size = 256
    dec_hid_size = 256
    out_size = 256
    val_tst_dir = './data/'
    val_prefix = 'devset1_2.lc'
    dev_prefix = 'devset1_2.lc'
    val_src_suffix = 'zh'
    val_ref_suffix = 'en'
    tests_prefix = ['devset3.lc']
    #val_tst_dir = '/home5/wen/2.data/iwslt14-de-en/'
    #val_tst_dir = '/home/wen/3.corpus/mt/iwslt14-de-en/'
    #val_prefix = 'valid.de-en'
    #val_src_suffix = 'de'
    #val_ref_suffix = 'en'
    ref_cnt = 16
    #tests_prefix = ['test.de-en']
    #ref_cnt = 1
    batch_size = 40
    max_epochs = 200
    #src_dict_size = 32009
    #trg_dict_size = 22822
    epoch_eval = True
    small = True
    use_multi_bleu = False
    #eval_small = True
    with_bpe = False
    cased = False
elif dataset == 'M':
    src_wemb_size = 512
    trg_wemb_size = 512
    enc_hid_size = 512
    align_size = 512
    dec_hid_size = 512
    out_size = 512
    val_tst_dir = '/home5/wen/2.data/mt/nist_data_stanseg/'
    #val_tst_dir = '/home/wen/3.corpus/mt/nist_data_stanseg/'
    #val_tst_dir = '/home5/wen/2.data/mt/uy_zh_300w/devtst/'
    #val_tst_dir = '/home/wen/3.corpus/mt/uy_zh_300w/devtst/'
    val_prefix = 'nist02'
    dev_prefix = 'nist02'
    #val_prefix = 'dev700'
    #dev_prefix = 'dev700'
    #val_src_suffix = '8kbpe.src'
    #val_src_suffix = 'uy.src'
    #val_src_suffix = 'uy.32kbpe.src'
    val_src_suffix = 'src'
    val_ref_suffix = 'ref.plain_'
    src_dict_size = 30000
    trg_dict_size = 30000
    ref_cnt = 4
    tests_prefix = ['nist03', 'nist04', 'nist05', 'nist06', 'nist08', '900']
    #tests_prefix = ['tst861']
    with_bpe = False
    with_postproc = False
    use_multi_bleu = False
    cased = False
    #char = True
elif dataset == 'L':
    #src_wemb_size = 500
    #trg_wemb_size = 500
    #enc_hid_size = 1024
    #align_size = 1024
    #dec_hid_size = 1024
    #out_size = 512
    #val_tst_dir = '/home/wen/3.corpus/wmt16/rsennrich/devtst/'
    #val_tst_dir = '/home/wen/3.corpus/wmt14/en-de-Luong/'
    val_tst_dir = '/home/wen/3.corpus/wmt2017/de-en/'
    val_prefix = 'newstest2014'
    #val_prefix = 'newstest2014.tc'
    use_multi_bleu = True
    val_src_suffix = 'en.16kbpe'
    val_ref_suffix = 'tc.de' if use_multi_bleu is True else 'ori.de'
    ref_cnt = 1
    tests_prefix = ['newstest2014.2737', 'newstest2015', 'newstest2016', 'newstest2017']
    #tests_prefix = ['newstest2009', 'newstest2010', 'newstest2011', 'newstest2012', 'newstest2014', 'newstest2015', 'newstest2016', 'newstest2017']
    #drop_rate = 0.2
    src_dict_size = 50000
    trg_dict_size = 50000
    with_bpe = True
    cased = True    # False: Case-insensitive BLEU  True: Case-sensitive BLEU
    #small = True
    #eval_small = True

display_freq = 10 if small else 1000
sampling_freq = 100 if small else 5000
sample_size = 5
if_fixed_sampling = False
eval_valid_from = 500 if eval_small else 100000
eval_valid_freq = 100 if eval_small else 20000

save_one_model = True
start_epoch = 1

model_prefix = dir_model + '/model'
best_model = dir_valid + '/best.model.pt' if dir_valid else 'best.model.pt'
# pretrained model
pre_train = None
#pre_train = best_model
fix_pre_params = False

# decoder hype-parameters
search_mode = 1
with_batch=1
ori_search=0
beam_size = 4
vocab_norm = 1  # softmax
len_norm = 2    # 0: no noraml, 1: length normal, 2: alpha-beta
with_mv = 0
merge_way = 'Y'
avg_att = 0
m_threshold = 100.
ngram = 3
alpha_len_norm = 0.6
beta_cover_penalty = 0.

'''
Starting learning rate. If adagrad/adadelta/adam is used, then this is the global learning rate.
Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001
'''
#opt_mode = 'adadelta'
#learning_rate = 1.0
#rho = 0.95

opt_mode = 'adam'
learning_rate = 1e-3
beta_1 = 0.9
beta_2 = 0.98

#opt_mode = 'sgd'
#learning_rate = 1.

max_grad_norm = 1.0

# Start decaying every epoch after and including this epoch
start_decay_from = None
learning_rate_decay = 0.5
last_valid_bleu = 0.

snip_size = 10
file_tran_dir = 'wexp-gpu-nist03'
laynorm = False
segments = False
seg_val_tst_dir = 'orule_1.7'

# 0: groundhog, 1: rnnsearch
model = 8 # 8 for transfomer

# convolutional layer
#fltr_windows = [1, 3, 5]   # windows size
#d_fltr_feats = [32, 64, 96]
fltr_windows = [1]
d_fltr_feats = [256]
d_mlp = 256

print_att = True

# free parameter for self-normalization
# 0 is equivalent to the standard neural network objective function.
#self_norm_alpha = 0.5
self_norm_alpha = None
#dec_gpu_id = None
gpu_id = [3]
#gpu_id = None

# Transfomer
proj_share_weight=True
embs_share_weight=False
d_k=64  # d_v == d_model // n_head
d_v=64
d_model=512     # == n_head*d_v
d_word_vec=512
d_inner_hid=1024
n_layers=1
n_head=8
warmup_steps=8000
drop_rate = 0.1 if model == 8 else 0.5
use_attcnn=True


