# wpynmt

pytorch nmt containing rnnsearch and transfomer

need to improve ...

# translate some samples
python wtrans.py -m model_file

# translate file
python wtrans.py -m model_file -i test_file_prefix

# evaluate alignment

score-alignments.py -d path/900 -s zh -t en -g wa -i force_decoding_alignment

# feat: add attcnn
