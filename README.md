# Prediction of stroke and COVID-19 death using deep learning and sequential medical histories in a nationwide atrial fibrillation cohort

## Project description

Recent advances in artificial intelligence can provide the basis for more personalised medical recommendations. In particular, advances in modelling large sequences of text using deep learning (DL) has opened up the possibility of using long term patient trajectories held in electronic health records. The performance of these DL models on selected disease prediction tasks is impressive but there is limited comparison to existing clinical prediction tools or exploration of how they could practically improve clinical practice.  

Anticoagulant prescribing decisions in atrial fibrillation (AF) offer a use case where the benchmark stroke risk prediction tool (CHA2DS2-VASc) used regularly in clinical practice could be meaningfully improved. AF is a disturbance of heart rhythm affecting 37.5 million people globally and significantly increases stroke risk. Anticoagulants reduce the risk of stroke and are recommended for people with AF and a high risk of stroke, broadly defined as a CHA2DS2-VASc >=2 based on the National Institute for Health and Care Excellence (NICE) threshold. However, NICE’s own evidence review highlights the need for improved stroke risk assessment and shows that whilst CHA2DS2-VASc is good for identifying people potentially at risk of stroke (high sensitivity) it is poor at identifying people who may not have a stroke (low specificity). Whilst evidence is limited, the predictive performance of CHA2DS2-VASc appears even worse at assessing the risk of first stroke for people with AF, with discriminatory statistics (e.g. c-index, area under the curve) below 0.60.  

COVID-19 has presented another risk factor for people with AF, who are at increased risk of poor outcomes if they become infected. Recent research from our group and others has observed that pre-existing use of antithrombotics, particularly anticoagulants, is associated with lower odds of people with AF dying from COVID-19. A prediction model that could also identify which people with AF were at greatest risk of COVID-19 death and could further inform anticoagulant prescribing decisions.  

This study, therefore, aims to develop and test a DL model that uses an individual’s medical history (represented as a sequence of codes) to predict first stroke in people with AF, and as a secondary outcome, COVID-19 death. Results will be directly compared against more simplistic machine learning (ML) methods and CHA2DS2-VASc to support translation to clinical practice. As the first DL model to be developed using the nationwide linked electronic health record (EHR) data in the NHS Digital Trusted Research Environment (TRE) for England, this study will also provide a framework for other researchers and clinical use cases.  

## How to cite this work
> Citation details to follow

## Code

Click [here](https://github.com/BHFDSC/CCU004_02/tree/main/code) to view the analysis code.

Phenotyping algorithms and codelists are available [here](https://github.com/BHFDSC/CCU004_02/tree/main/phenotypes)

## Project approval

This is a sub-project of [project CCU004](https://github.com/BHFDSC/CCU004) approved by the CVD-COVID-UK / COVID-IMPACT Approvals & Oversight Board (sub-project: CCU004_02).

## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this software except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
