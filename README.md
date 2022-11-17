# pacbayes-adaptation-UAI2022

This is the code repository for [PAC-Bayesian Domain Adaptation Bounds for Multiclass Learners](https://openreview.net/pdf?id=S0lx6I8j9xq) to appear in [UAI 2022](https://www.auai.org/uai2022/).

This research was conducted in conjunction with [The Change that Matters in Discourse Parsing: Estimating the Impact of Domain Shift on Parser Error](https://arxiv.org/abs/2203.11317) to appear in Findings of [ACL 2022](https://www.2022.aclweb.org). 

Some of the code is shared across repositories as detailed below.

## Shared Code
Code for running DANN experiments and summarizing DANN results is available in this repository. Code for running and summarizing additional experiments is available in the shared code repository [here](https://github.com/anthonysicilia/multiclass-domain-divergence). Further, a python package designed to compute key terms in the bounds we propose is available [here](https://github.com/anthonysicilia/classifier-divergence).

Please consider citing one or both papers if you use this code.

## Relevant Links
OpenReview (UAI 2022): https://openreview.net/pdf?id=S0lx6I8j9xq

arXiv (ACL 2022): https://arxiv.org/abs/2203.11317

shared code: https://github.com/anthonysicilia/multiclass-domain-divergence

ACL code: https://github.com/anthonysicilia/change-that-matters-ACL2022

package: https://github.com/anthonysicilia/classifier-divergence

## Running the code
For ease of use, we have created a python script ```make_scripts.py``` to generate example bash scripts. In many cases, these may be exactly identical to the scripts used to generate results in the paper. The bash scripts interfrace with the ```experiments``` module to create the raw results for DANN experiments. Following this, code for summarizing raw results can be run using ```results.py```. Feel free to contact us with any questions (e.g., by raising an issue here or using the contact information available in the accompanying papers).

### Notable Versions
Code was run using the following versions:
 - python==3.7.4
 - matplotlib==3.5.0
 - numpy==1.21.2
 - pandas==1.3.5
 - scipy==1.7.3
 - seaborn==0.12.1
 - torch==1.10.2 (build py3.7_cuda10.2_cudnn7.6.5_0)
 - tqdm==4.45.0
 
  ## More Papers
 This paper is one of a series from our lab using learning theory to study understanding and generation in NLP. Check out some of our other papers here:
  - [Modeling Non-Cooperative Dialogue: Theoretical and Empirical Insights](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00507/113020/Modeling-Non-Cooperative-Dialogue-Theoretical-and)
  - [LEATHER: A Framework for Learning to Generate Human-like Text in Dialogue](https://arxiv.org/abs/2210.07777)
