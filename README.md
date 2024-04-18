# SciCite

This github repository presents a project done for a CS4248 project assignment by group 19. We implement a multitude of models and evaluate each of its performances against different variations of text, categorized based on their semantic changes. Each model can be found in their respective folders, named based on the model's name.

The below serves as a guide and notes to run and test the models.

1. Run baseline model (MNB and LR) in `/initial-evaluation/MNB_LR_category.ipynb`
2. Run other models in `<model_name>.ipynb`
3. `paraphrased.jsonl` and `synonymized.jsonl` is necessary if you want to run the category dataset.
4. All models run in the order of each Jupyter Notebook's code chunks
5. Data preprocessing done in each model file
6. Packages needed are listed in each coding file. Installation of these packages can be done through the following command:

```
pip install -r <model_name>/requirements.txt
```
With this project, we hope to be able to obtain deeper insights into the nuances of semantic representation within each model and contribute positively to the growing world of NLP.

## Accuracy Evaluation
| Test Data	| MNB TFIDF	| MNB BOW	| LR TFIDF | LR BOW	| GloVE LSTM	| GloVE CNN	| RoBERTa | ALBERT	| DistilBERT |	BERT-Base, Uncased	| GPT-2|
|-----------|---------|----------|----------|---------|-------------|-----------|---------|---------|-------------|---------------------|------|
| Default Test Data	| 0.75873	| 0.74906	| 0.77539	| 0.77700	| 0.73778	| 0.64804	| 0.83665	| 0.85008	| 0.85277	| 0.84148 | 0.85921 |
| Long Data	| 0.75234	| 0.74358	| 0.77235	| 0.77235	| 0.73358	| 0.64290	| 0.80175	| 0.84615	| 0.85178	| 0.80550 | 0.85428 |
| Short Data|	0.79770 |	0.78244 |	0.79389 |	0.80534 |	0.76336 |	0.67939	| 0.80916	| 0.87405	| 0.85878	| 0.83969 | 0.88931 |
| Paragraph Data | 0.78450 | 0.74818 |	0.78692	| 0.80145	| 0.74818	| 0.64165	| 0.79903	| 0.84998	| 0.85956	| 0.81598 | 0.85714 |
| Typo Data	| 0.75282	| 0.73455	| 0.75658	| 0.75228	| 0.71359	| 0.64105	| 0.79097	| 0.81354	| 0.82751	| 0.79044 | 0.83396 |
| Synonym Data	| 0.68995	| 0.69854	| 0.66953	| 0.66469	| 0.66093	| 0.58517	| 0.76679	| 0.74960	| 0.78076	| 0.75121 | 0.80978 |
| Paraphrased data	| 0.75389	| 0.74153	| 0.76034	| 0.76034	| 0.70500	| 0.63138	| 0.75819	| 0.82805	| 0.82160	| 0.76088 | 0.81730 |

## F1 Evaluation
| Test Data	| MNB TFIDF	| MNB BOW	| LR TFIDF | LR BOW	| GloVE LSTM	| GloVE CNN	| RoBERTa | ALBERT	| DistilBERT |	BERT-Base, Uncased	| GPT-2|
|-----------|---------|----------|----------|---------|-------------|-----------|---------|---------|-------------|---------------------|------|
| Default Test Data	| 0.72620	| 0.70441	| 0.74979	| 0.74725	| 0.70809	| 0.70731	| 0.80087	| 0.83034	| 0.83778	| 0.79672	| 0.84320 |
| Long Data	| 0.71778	| 0.69694	| 0.74631	| 0.74165	| 0.70430	| 0.71177	| 0.75766	| 0.82584	| 0.83607	| 0.75217	| 0.83814 |
| Short Data	| 0.77622	| 0.74811	| 0.77121	| 0.72426	| 0.73319	| 0.66362	| 0.76056	| 0.86011	| 0.85259	| 0.77279	| 0.87797 |
| Paragraph Data	| 0.75257	| 0.70082	| 0.76611	| 0.77945	| 0.72449	| 0.75132	| 0.78419	| 0.82996	| 0.85115	| 0.77961	| 0.84117 |
| Typo Data	| 0.71383	| 0.68237	| 0.72523	| 0.71715	| 0.67565	| 0.67345	| 0.73502	| 0.77306	| 0.80061	| 0.73635	| 0.81445 |
| Synonym Data	| 0.59607	| 0.59748	| 0.55024	| 0.54610	| 0.56793	| 0.46597	| 0.69969	| 0.63794	| 0.70871	| 0.66497	| 0.76672 |
| Paraphrased data	| 0.71912	| 0.69669	| 0.73212	| 0.72892	| 0.67462	| 0.69057	| 0.71023	| 0.80503	| 0.80437	| 0.70852	| 0.79924 |
