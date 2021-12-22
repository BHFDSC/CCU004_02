# A nationwide deep learning pipeline to predict stroke and COVID-19 death in atrial fibrillation 

## Project description

Recent advances in artificial intelligence can provide the basis for improving medical predictions. In particular, advances in modelling large sequences of text using deep learning (DL) and natural language processing has opened up the possibility of harnessing long-term patient trajectories held as medical codes in electronic health records (EHR). Unlike conventional statistical and machine learning (ML) models, DL models can learn representations by directly taking long, individual sequences of medical codes stored in EHRs as inputs and could potentially identify complex, long-term dependencies between medical events. To date, the improved performance of these DL models on their selected prediction tasks is promising but there has been limited comparison against prediction tools used routinely in clinical practice with comparisons typically made to other DL or ML methods. A direct comparison is important to demonstrate clearly where and by how much DL and ML can offer improvements and to help in integrating these methods (where appropriate) into routine clinical practice.

Anticoagulant prescribing decisions in atrial fibrillation (AF) offer a use case where the benchmark stroke risk prediction tool (CHA2DS2-VASc7) used routinely in clinical practice could be meaningfully improved by including more information from a patient’s medical history. AF is a disturbance of heart rhythm affecting 37.5 million people globally and significantly increases ischaemic stroke risk. Anticoagulants reduce the risk of stroke and are recommended for people with AF and a high risk of stroke, broadly defined as a CHA2DS2-VASc >=2 based on the National Institute for Health and Care Excellence (NICE) threshold. The CHA2DS2-VASc score benefits from being easy to calculate and interpret, however, it only measures 7 variables (age, sex, history of congestive heart failure, hypertension, stroke/TIA/thromboembolism, vascular disease and diabetes) and NICE’s own evidence review highlights the need for improved stroke risk assessment. It shows that whilst CHA2DS2-VASc is good for identifying people potentially at risk of stroke (high sensitivity) it is poor at identifying people who may not have a stroke (low specificity). The ability of CHA2DS2-VASc to discriminate an individual’s future stroke risk is also only moderate (pooled area under the receiver operating characteristics curve (AUC) of 0.67 across 27 studies) and potentially lower for predicting first ever stroke based on information at the point of AF diagnosis where available evidence is significantly limited. Recent research has also observed that pre-existing use of antithrombotics, particularly anticoagulants, is associated with lower odds of people with AF dying from COVID-19. A model that could improve prediction of first stroke in people with AF and also identify those at greatest risk of COVID-19 death would be a potentially valuable new tool to inform anticoagulant prescribing decisions.  

In this study, we design and build the first DL and ML pipeline that uses the routinely updated, linked EHR data for 56 million people in England accessed via NHS Digital’s Trusted Research Environment (TRE). We use this pipeline to predict first ischaemic stroke in people with AF (mean follow-up time 7.2 years), and as a secondary outcome, COVID-19 death, using individual sequences of medical codes from the entire primary and secondary care record.
We compare the performance of our DL and ML pipeline directly against the CHA2DS2-VASc score to support translation to clinical practice and demonstrate a 17% improvement on predicting first stroke in AF.
The code for our pipeline is generalisable, opensource and designed to provide a proof-of-concept framework that other researchers and developers can build on.


## How to cite this work  

Pre-print available [here](https://www.medrxiv.org/content/10.1101/2021.12.20.21268113v1)

## Code

Click [here](https://github.com/BHFDSC/CCU004_02/tree/main/code) to view the analysis code.

Phenotyping algorithms and codelists are available [here](https://github.com/BHFDSC/CCU004_02/tree/main/phenotypes)

## Project approval

This is a sub-project of [project CCU004](https://github.com/BHFDSC/CCU004) approved by the CVD-COVID-UK / COVID-IMPACT Approvals & Oversight Board (sub-project: CCU004_02).

## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this software except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
