# Impact Resume Generation

The code is written in python and requires Pytorch 3.1.


* We trained according to the following command
```
python train_cpgn_length.py --data data/parsed_data.h5 --model CPGN_length.pt --gpu 0
```

The output of the training can be found in the slurm_output files.
The configuration used for training can be found in the gpu_train.job file


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
