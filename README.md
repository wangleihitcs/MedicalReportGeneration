### Medical Report Generation
A Project for Medical Report Generation, it is a base model.

### Config
- python 2.7/tensorflow 1.8.0
- extra package: nltk, json, PIL, numpy

### DataDownload
- IU X-Ray Dataset
    * The raw data is from [openi](https://openi.nlm.nih.gov/), it has many public datasets.
    * The proccessed data is on [](), you should unzip it to dir 'data/NLMCXR_png_pairs/', got 3011 image pairs.
- PreTrained InceptionV3 model
    * The raw model is from [tensorflow slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)
    * The proccessed data is on [](), you shold unzip it to dir 'data/pretrain_model/'

### Train
#### First, get post proccess data(I have done it)
- get 'data/data_entry.json', it is the report sentences.
- get 'data/train_split.json' and 'data/test_split.json', it is the ids for train/val/test.
- get 'data/vocabulary.json', it is the vocabulary extracted from report.

#### Second, get TFRecord files
- get 'data/train.tfrecord' and 'data/test.tfrecord'
    ```shell
    $ python datasets.py
    ```
    e.g. if you get tfrecord files, you must annotate the code for func 'get_train_tfrecord()'
#### Third, go train
    ```shell
    $ python train.py
    ```

### Framework
#### Core Framework
![example](data/experments/framework.png)

e.g.Yuan Xue et.al-**Multimodal Recurrent Model with Attention for Automated Radiology Report Generation**, MICCAI 2018

### Experments
#### Metrics Results
|  | BLEU_1 | BLEU_2 | BLEU_3 | BLEU_4 | METEOR | ROUGE | CIDEr |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CNN-RNN<sup>[10]</sup> | 0.6687 | 0.4879 | 0.3421 | 0.2364 | 0.2096 | 0.4838 | 0.6972 |
| CNN-RNN-Att<sup>[11]</sup> | 0.6687 | 0.4879 | 0.3421 | 0.2364 | 0.2096 | 0.4838 | 0.6972 |
| Hier-RNN<sup>[9]</sup> | 0.3508 | 0.2385 | 0.1642 | 0.1127 | 0.1607 | 0.3252 | 0.2612 |
| MRNA<sup>[6]</sup> | 0.6687 | 0.4879 | 0.3421 | 0.2364 | 0.2096 | 0.4838 | 0.6972 |
| Vis-RNN | 0.6687 | 0.4879 | 0.3421 | 0.2364 | 0.2096 | 0.4838 | 0.6972 |
| Sem-RNN | 0.6687 | 0.4879 | 0.3421 | 0.2364 | 0.2096 | 0.4838 | 0.6972 |

#### Details
I split train/test dataset as 2811/300, use Adam with initial learning rate is 1e-4 with 5 epoch for decay 0.9.Then I set 
generate max 10 sentence with max 40 words for a sentence. The word embedding size is 512 and RNN units is 512. The more details is on
config.py


### Summary
#### Problems
There are many challenges for this task, I refer to some points of <sup>**[1]**</sup>.
- **Very Small Medical Data**, most medical datasets only with images and nearly without bounding boxes and reports.
- **Very Uncertainty Report Descriptions**, because different doctors have different style description for diagnosis report.
- **More-Like Dense Caption Task not Story Generation**, we should ground the description sentence with relevant region.
- **Unsuitable Metrics**, the BLEU for machine translation and CIDEr for captioning and so on are not suitable for this task.
- **Impractical**, up to now, there are 4-5 papers <sup>**[5][6][7][8]**</sup>. public for this task, but to be honest, they are only for papers.

#### Little Advice
If you want to research medical report generation, you could get more data, and you could focus on the **Semantic Information** not **Visual Information** when data is small.
In VQA task, someones found that Language is more useful than Image.

### References
- [1][医学诊断报告生成论文综述](https://blog.csdn.net/wl1710582732/article/details/85345285)
- [2][Tensorflow Model released im2text](https://github.com/tensorflow/models/tree/master/research/im2txt)
- [3][MS COCO Caption Evaluation Tookit](https://github.com/tylin/coco-caption)
- [4]**TieNet Text-Image Embedding Network for Common Thorax Disease Classification and Reporting in Chest X-rays**, Xiaosong Wang et at, CVPR 2018, NIH
- [5]**On the Automatic Generation of Medical Imaging Reports**, Baoyu Jing et al, ACL 2018, CMU
- [6]**Multimodal Recurrent Model with Attention for Automated Radiology Report Generation**, Yuan Xue, MICCAI 2018, PSU
- [7]**Hybrid Retrieval-Generation Reinforced Agent for Medical Image Report Generation**, Christy Y. Li et al, NIPS 2018, CMU
- [8]**Knowledge-Driven Encode, Retrieve, Paraphrase for Medical Image Report Generation**, Christy Y. Li et al, AAAI 2019, DU
- [9]**A Hierarchical Approach for Generating Descriptive Image Paragraphs, Jonathan Krause** et al, CVPR 2017, Stanford
- [10]**Show and Tell: A Neural Image Caption Generator**, Oriol Vinyals et al, CVPR 2015, Google
- [11]**Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**, Kelvin Xu et at, ICML 2015