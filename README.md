## Introduction
This project seeks to enhance the understanding and evaluation of turfgrass by developing a model that utilizes visual rating data collected by the National Turfgrass Evaluation Program (NTEP). By incorporating methodologies from item response theory and Gaussian Processes, and further refining the model with Hilbert Space Approximation, the project aims to accurately compare turfgrass cultivars across time and space. This approach addresses the challenges posed by the subjective nature of visual ratings and the variability in rating standards over time and across different locations. This model promises to elevate the scientific rigor of turfgrass research and management, supporting the NTEP's mission in evaluating turfgrass varieties across North America.


## Relevant publications

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

## Installation Guide for `nteprsm` on Mac

### Prerequisites

Before you begin, ensure you have the following installed on your Mac:

- [Homebrew](https://brew.sh/)
- [Git](https://git-scm.com/)
- [Python 3.8+](https://www.python.org/)
- [Poetry](https://python-poetry.org/)

### Steps

1. **Install Homebrew** (if not already installed):
    ```sh
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2. **Install Git** (if not already installed):
    ```sh
    brew install git
    ```

3. **Install Python 3** (if not already installed):
    ```sh
    brew install python
    ```

4. **Install Poetry** (if not already installed):
    ```sh
    brew install poetry
    ```

5. **Clone the Repository**:
    ```sh
    git clone https://github.com/QhenryQ/nteprsm.git
    cd nteprsm
    ```

6. **Install Dependencies with Poetry**:
    ```sh
    poetry install
    ```

7. **Open VS Code**:
    - Launch Visual Studio Code (VS Code).
    - Open the `nteprsm` project folder in VS Code.

8. **Open Terminal in VS Code and Run the Environment**:
    - Open the terminal in VS Code (`View > Terminal`).
    - Activate the Poetry environment by running:
        ```sh
        poetry shell
        ```
    - Ensure the environment is active.

9. **Run the Model**:
    ```sh
    python nteprsm/model.py config/nteprsm_in1kbg07.yml
    ```

10. **Wait for the Iterations to Complete**:
    - The process will start, and you will need to wait until the iterations are done.

11. **Retrieve the CSV Files**:
    - After the iterations are complete, the CSV files will be generated.
    - These CSV files can later be used in Jupyter notebooks.

### Troubleshooting

If you encounter any issues, ensure that all dependencies are installed correctly and that you have activated the Poetry environment.
