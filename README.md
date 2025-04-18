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

Before you begin, ensure you have the following installed on your System:
- [Git](https://git-scm.com/)

### Steps
1. **Clone the Repository**:
    ```sh
    git clone https://github.com/QhenryQ/nteprsm.git
    cd nteprsm
    ```

2. **Setup the Repository**:

    ```bash
    source tools/repo_setup.sh
    ```

    This script will perform the following tasks:

    - **Set up a data directory**:  
      We use Google Drive to share and sync data among collaborators. Please contact [Henry Qu](mailto:henry.yqu@gmail.com) to request access to the data.

    - **Install Python 3.12.x**:  
      Ensure that Python 3.12.x is installed on your system.

    - **Install Poetry 2.1.0**:  
      The script will also install Poetry version 2.1.0 for dependency management.
 
3. **Open VS Code**:
    - Launch Visual Studio Code (VS Code).
    - Open the `nteprsm` project folder in VS Code.

4. **Open Terminal in VS Code and Run the Environment**:
    - Open the terminal in VS Code (`View > Terminal`).
    - Activate the Poetry environment by running:
        ```sh
        poetry shell
        ```
    - Ensure the environment is active.

5. **Run the Model**:
    ```sh
    python nteprsm/model.py config/nteprsm_in1kbg07.yml
    ```

6. **Wait for the Iterations to Complete**:
    - The process will start, and you will need to wait until the sampling are done.

7. **Retrieve the CSV Files**:
    - After the iterations are complete, the CSV files will be generated.
    - These CSV files can later be used in Jupyter notebooks.

### Troubleshooting

If you encounter any issues, ensure that all dependencies are installed correctly and that you have activated the Poetry environment.
