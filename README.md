# SciCite

This github repository presents a project done for a CS4248 project assignment by group 19. We implement a multitude of models and evaluate each of its performances against different variations of text, categorized based on their semantic changes. Each model can be found in their respective folders, named based on the model's name.

The below serves as a guide and notes to run and test the models.

1. Run baseline model, MNB, and LR in `/initial-evaluation/MNB_LR_category.ipynb`
2. Run other models in `<model_name>.ipynb`
3. `paraphrased.jsonl` and `synonymized.jsonl` is necessary if you want to run the category dataset.
4. All models run in the order of each Jupyter Notebook's code chunks
5. Data preprocessing done in each model file
6. Packages needed are listed in each coding file. Installation of these packages can be done through the following command:

```
pip install -r <model_name>/requirements.txt
```

With this project, we hope to be able to obtain deeper insights into the nuances of semantic representation within each model and contribute positively to the growing world of NLP.
