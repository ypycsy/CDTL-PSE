# CDTL-PSE
## Cross Domain Transfer Learning to Recognize Professioal Skill Entity (CDTL-PSE)

## Requirements
- [Tensorflow=1.2.0](https://github.com/tensorflow/tensorflow)
- [jieba=0.37](https://github.com/fxsjy/jieba)
- [sklearn=0.18.1]

## Model
The CDTL-PSE model is improved based on Bi-LSTM-CRF model，which four layers are included. In this model, the source corpus is decomposed into three source domains according to its different NER tasks, and the labels of these three tasks are mapped into the same label set .
The target task is to recognize PSEs from recruitment texts in Chinese. The embedded layer and the Bi-LSTM layer are shared between the source and target domains for transfer learning from each source domain to the target domain, but task specific domain adaptation layer and CRF layer are constructed for specific task.
Hence, the framework of CDTL-PSE consists of three sub-networks.

## Datasets
Target Doamain Data: target_data/All_skill.train
Source Domain Data: source_data/loc;source_data/per;source_data/org;
Model Test Data: test_data/fashion design.txt;test_data/mechanical manufacturing.txt;

### Default parameters:
- batch size: 20
- gradient clip: 5
- embedding size: 100
- optimizer: Adam
- dropout rate: 0.5
- learning rate: 0.001

Word vectors are trained with CBOW version of word2vec on Chinese WiKi corpus, provided by [Chuanhai Dong](https://github.com/sea2603).

### Train the model with default parameters:
```shell
$ python3 main.py --train=True --clean=True
if domain adapation layer choose add new Bi-lstm method, --Add_Bilstm=True
else --Add_Bilstm=False
```
### Online Model Test:
```shell
$ python3 main.py --train=False --clean=False
```

## Suggested readings:
1. Lample G, Ballesteros M, Subramanian S, et al. Neural Architectures for Named Entity Recognition [C]// Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL), ACL, 2016:260–270.
2. Huang Z, Xu W, Yu, K. Bidirectional LSTM-CRF Models for Sequence Tagging[J]. arXiv preprint arXiv:1508.01991, 2015.
3. Pan S. J., Yang Q. A Survey on Transfer Learning[J]. IEEE Transactions on Knowledge & Data Engineering, 2010, 22(10):1345-1359..
4. Yang Z, Salakhutdinov R, Cohen W W. Transfer learning for sequence tagging with hierarchical recurrent networks[C]// Proceedings of 5th International Conference on Learning Representations (ICLR), IEEE, 2017.
5. Lee J Y, Dernoncourt F, Szolovits P. Transfer learning for named-entity recognition with neural networks[J]. arXiv preprint arXiv:1705.06273, 2017.
6. Peng N, Dredze M. Multi-task multi-domain representation learning for sequence tagging[J]. CoRR, abs/1608.02689, 2016.
7. Daumé III H. Frustratingly easy domain adaptation[C]// Proceedings of the 45th Annual Meeting of the Association for Computational Linguistics, ACL, 2007.
8. Lin B Y, Lu W. Neural adaptation layers for cross-domain named entity recognition[C]// Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), ACL, 2018:2012-2022.
9. Devlin, J, Chang, M, Lee, k, et al. Bert: Pre-training of Deep Bidirectional Transformers for Language Understanding[C]// Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), ACL, 2019: 4171-4186.

