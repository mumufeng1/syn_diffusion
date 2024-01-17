# A diffusion-based generative network for de novo synthetic promoter design
=====================================

Computer-aided promoter design is a major development trend in synthetic promoter engineering. Various deep learning models have been used to evaluate or screen synthetic promoters, but there have been few works for de novo promoter design. To explore the potential ability of generative models in promoter design, we established a diffusion-based generative model for promoter design in Escherichia coli. The model was completely driven by sequence data and could study essential characteristics of natural promoters, thus generating synthetic promoters similar to natural promoters in structure and component. We also improved the calculation method of FID indicator, using convolution layer to extract feature matrix of promoter sequence instead. As a result, we got a FID equal to 1.37%, which meant synthetic promoters have a similar distribution to natural ones. Our work provided a fresh approach to de novo promoter design, indicating that a completely data-driven generative model was feasible for promoter design.
<p align="center">
  <img width="800" height="1000" src="https://github.com/mumufeng1/syn_diffusion/blob/main/fig/fig.png">
</p>


## Prerequisites

- Python, NumPy, TensorFlow, SciPy, Matplotlib, Seabron
- A recent NVIDIA GPU
- pytorch == 1.13.1
- Cuda == 11.7
- Python == 3.7.0

## Dataset

Three datasets were utilized in the experimental design for synthesizing promoter sequences. 
### Wang(promoters_1)
The first dataset, published by Wang et al[1], comprised 11,884 regulatory sequences extracted from E. coli genomes. Each promoter sequence was precisely defined as the 50 base pairs upstream of the Transcription Start Site (TSS). 
### Johns(promoters_2)
The second dataset, published by Johns et al.[2], contained a total of 29,249 regulatory sequences from 184 prokaryotic genomes. The regulatory sequence was defined as 165 base pairs upstream of the gene start codon. The regulatory library was cloned into a p15A vector and transformed into E. coli MG1655. 
### Lubliner(promoters_3)
The third dataset, published by Lubliner et al[3], included 7,536 core promoter sequences obtained from Saccharomyces cerevisiae. The regulatory sequence was defined as 165 base pairs upstream of the gene start codon. The regulatory library constructed from this dataset was transformed into yeast cells (Y8205 strain). These three datasets will be individually employed to train three diffusion models for subsequent experiments.

## Training Diffusion Models for Learning Natural Promoters

model training: diffusion_model_training.ipynb

synthetic promoter generation: syn_seq_generate.ipynb

## Training WGAN Models for Learning Natural Promoters

model training & synthetic promoter generation: gan_torch.py

## Synthetic Promoter Validation

GC content: GC_content.ipynb

FID: FID_indicator.ipynb

IS: IS_indicator.ipynb

R2 of k-mers frequency: k_mer_freq.ipynb

plotting: draw_plot.ipynb

## Citation
Wang Y, Wang H, Liu L, et al. Synthetic Promoter Design in Escherichia coli based on Generative Adversarial Network[J]. BioRxiv, 2019: 563775.

## References
[1]  Wang, Y.; Wang, H.; Wei, L.; Li, S.; Liu, L.; Wang, X. Synthetic promoter design in
Escherichia coli based on a deep generative network. Nucleic Acids Research 2020,
48, 6403¨C6412

[2]  Johns N I, Gomes A L C, Yim S S, et al. Metagenomic mining of regulatory elements enables programmable species-selective gene expression[J]. Nature methods, 2018, 15(5): 323-329.

[3] Lubliner S, Regev I, Lotan-Pompan M, et al. Core promoter sequence in yeast is a major determinant of expression level[J]. Genome research, 2015, 25(7): 1008-1017.
