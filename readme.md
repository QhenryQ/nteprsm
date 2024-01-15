## Running the Notebook (Linux)

Download the github repository onto the local machine.

Follow https://python-poetry.org/docs/ to install Poetry.

Run the following terminal commands in the directory of installation:
  -> poetry install
  -> poetry run jupyter notebook --allow-root

Open the notebook in a browser by copying the notebook link in the terminal
run src/model_v8_sharedparams_fit.ipynb to fit the model
run src/model_v8_sharedparams_analyze.ipynb to load and analyze the model

## Relevant publications

If you use auto-sklearn in scientific publications, we would appreciate citations.

**A latent scale model to minimize subjectivity in the analysis of visual rating data for the National Turfgrass Evaluation Program**
*Yuanshuo Qu, Len Kne, Steve Graham, Eric Watkins, and Kevin Morris*
Front Plant Sci 2023 Jul 6;14:1135918

[Link](https://www.frontiersin.org/articles/10.3389/fpls.2023.1135918/full) to publication.
```
@article{qu2023latent,
   title     = {A latent scale model to minimize subjectivity in the analysis of visual rating data for the National Turfgrass Evaluation Program},
   author    = {Qu, Yuanshuo and Kne, Len and Graham, Steve and Watkins, Eric and Morris, Kevin},
   journal   = {Frontiers in Plant Science},
   volume    = {14},
   year      = {2023},
   publisher = {Frontiers Media SA}
}
```
