# Impact Resume Generation

The code is written in python and requires Pytorch 3.1.


* We trained according to the following command
```
python train_cpgn_length.py --data data/parsed_data.h5 --model paraphrasing_inpattern_length_v3.pt
```

The output of the training can be found in the slurm_output files.
The configuration used for training can be found in the gpu_train.job file

The final model file(paraphrasing_inpattern_length_v3.pt) can be found at: https://drive.google.com/file/d/11R7BPAizdAy45iNZ7bMyImQVduQO-8P0/view?usp=share_link

* To **generate** paraphrases run generate_paraphrases_length.py or generate_paraphrases_inpattern_length.py

## Reference:
```
@inproceedings{dehghani2021controllable,
  title={Controllable Paraphrase Generationwith Multiple Types of Constraints},
  author={Dehghani, Nazanin and Hajipoor, Hassan and Chevelu, Jonathan and Lecorvé, Gwénolé},
  booktitle={Proceedings of the Controllable Generative Modeling in Language and Vision Workshop at NeurIPS 2021 (CtrlGen)},
  year={2021}
}
```
