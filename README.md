# pacbayes-adaptation-UAI2022

This is the code repository for “PAC-Bayesian Domain Adaptation Bounds for Multiclass Learners” to appear in [UAI 2022](https://www.auai.org/uai2022/).

This research was conducted in conjunction with [The Change that Matters in Discourse Parsing: Estimating the Impact of Domain Shift on Parser Error](https://arxiv.org/abs/2203.11317) to appear in Findings of [ACL 2022](https://www.2022.aclweb.org). 

Some of the code is shared across repositories as detailed below.

## Shared Code
Code for running DANN experiments and summarizing DANN results is available in this repository. Code for running and summarizing additional experiments is available in the shared code repository [here](https://github.com/anthonysicilia/multiclass-domain-divergence). Further, a python package designed to compute key terms in the bounds we propose is available [here](https://github.com/anthonysicilia/classifier-divergence).

Please consider citing one or both papers if you use this code.

## Relevant Links
arXiv (UAI 2022): Forthcoming

arXiv (ACL 2022): https://arxiv.org/abs/2203.11317

shared code: https://github.com/anthonysicilia/multiclass-domain-divergence

ACL code: https://github.com/anthonysicilia/change-that-matters-ACL2022

package: https://github.com/anthonysicilia/classifier-divergence

## Running the code
For ease of use, we have created a python script ```make_scripts.py``` to generate example bash scripts. In many cases, these may be exactly identical to the scripts used to generate results in the paper. The bash scripts interfrace with the ```experiments``` module to create the raw results for DANN experiments. Following this, code for summarizing raw results can be run using ```results.py```. Feel free to contact us with any questions (e.g., by raising an issue here or using the contact information available in the accompanying papers).
