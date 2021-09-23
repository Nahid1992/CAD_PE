![](https://github.com/Nahid1992/CAD_PE/blob/main/figures/Tittle_Logo.png)
<!-- # Seeking an Optimal Approach for Computer-aided Pulmonary Embolism Detection -->
Pulmonary embolism (PE) represents a blood clot that travels to the blood vessels in the lung, causing vascular obstruction, and in some patients, death. CT pulmonary angiography (CTPA), is the most common type of medical imaging to evaluate patients with suspected PE. These CT scans consist of hundreds of images that require detailed review to identify clots within the pulmonary arteries. Recent research in deep learning across academia and industry produced numerous architectures, various model initialization, and distinct learning paradigms. It has resulted in many competing approaches to CAD implementation in medical imaging and produced great confusion in the CAD community. we have conducted extensive experiments with various deep learning methods applicable for PE diagnosis at both slice and exam levels using the [RSNA PE dataset](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/overview).
 
 ![](https://github.com/Nahid1992/CAD_PE/blob/main/figures/Workflow.png)
 
## Publication
**Seeking an Optimal Approach for Computer-aided Pulmonary Embolism Detection** <br/><br/>
Nahid Ul Islam<sup>1</sup>, Shiv Gehlot<sup>1</sup>, Zongwei Zhou<sup>1</sup>, Michael B Gotway<sup>2</sup>, and Jianming Liang<sup>1</sup><br/>
<sup>1</sup>Arizona State University, <sup>2</sup>Mayo Clinic<br/>
International Conference on Medical Image Computing and Computer Assisted Intervention ([MICCAI 2021](https://www.miccai2021.org/)); Machine Learning in Medical Imaging ([MLMI2021](https://sites.google.com/view/mlmi2021/))<br/>
[Paper](https://arxiv.org/pdf/2109.07029.pdf) | [Graphical Abstract](https://github.com/Nahid1992/CAD_PE/blob/main/figures/MLMI2021_Graphical_abstract_v7.pdf) | [Poster](https://github.com/Nahid1992/CAD_PE/blob/main/figures/MLMI2021_PE_Poster_v6.pdf) | [Code](https://github.com/jlianglab/CAD_PE_Detection) | [Presentation](https://github.com/Nahid1992/CAD_PE/blob/main/figures/MLMI2021_Presentation__.pdf)

## Major results from our work
**1. Transfer learning improves the performance despite modality difference between datasets**
<p align="left">
<img src="https://github.com/Nahid1992/CAD_PE/blob/main/figures/Backbone_Random_vs_ImageNet.jpg" width=70% height=70%>
</p>

**2. Squeeze & excitation blocks enhance CNN performance**
<p align="left">
<img src="https://github.com/Nahid1992/CAD_PE/blob/main/figures/Backbone_w_o_SEblock.jpg" width=70% height=70%>
</p>
 
**3. Transfer learning with self-supervised methods performs better than supervised model**
<p align="left">
<img src="https://github.com/Nahid1992/CAD_PE/blob/main/figures/Backbone_ResNet50_SSL.jpg" width=70% height=70%>
</p>

**4. Vision transformer performs inferiorly compared with CNN**
<p align="left">
<img src="https://github.com/Nahid1992/CAD_PE/blob/main/figures/ViT-B.jpg" width=70% height=70%>
</p>
 
**5. Conventional classification (CC)  marginally outperforms the Multiple instance learning (MIL) for exam-level data**
<p align="left">
<img src="https://github.com/Nahid1992/CAD_PE/blob/main/figures/2ndStage_PE_Exam_CCvsMIL.jpg" width=70% height=70%>
</p>

## Data organization and preprocessing steps
This framework was built on top of the first place solution. The necessary data organization as well as the preprocessing steps can be found [here](https://github.com/GuanshuoXu/RSNA-STR-Pulmonary-Embolism-Detection).

## SeXception pre-trained weights
Insprired from SeResNet50 and SeResNext50, we added squeeze and excitation (SE) block to Xception architecture and pre-trained it on ImageNet dataset. The pre-trained SeXception model can be found [here](https://drive.google.com/file/d/128gkASApEH3mbBJ1ohwU9CRlw1oH9cHO/view?usp=sharing).

## Citation
If you use this code or use our pre-trained weights for your research, please cite our paper:

```
@misc{islam2021seeking,
  title={Seeking an Optimal Approach for Computer-Aided Pulmonary Embolism Detection},
  author={Islam, Nahid Ul and Gehlot, Shiv and Zhou, Zongwei and Gotway, Michael B and Liang, Jianming},
  journal={arXiv preprint arXiv:2109.07029},
  year={2021}
}
```

## Acknowledgement
This research has been supported partially by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, and partially by the NIH under Award Number R01HL128785.  The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH. This work has utilized the GPUs provided partially by the ASU Research Computing and partially by the Extreme Science and Engineering Discovery Environment (XSEDE) funded by the National Science Foundation (NSF) under grant number ACI-1548562. We thank Ruibin Feng for aggregating 14 self-supervised pre-trained models. The content of this paper is covered by patents pending.

## License
Released under the [ASU GitHub Project License](https://github.com/jlianglab/CAD_PE_Detection/blob/main/LICENSE)

