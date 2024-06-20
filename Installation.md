# Installation Guide for `nteprsm` on Mac

## Prerequisites

Before you begin, ensure you have the following installed on your Mac:

- [Homebrew](https://brew.sh/)
- [Git](https://git-scm.com/)
- [Python 3.8+](https://www.python.org/)
- [Visual Studio Code](https://code.visualstudio.com/)

## Steps

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

4. **Clone the Repository**:
    ```sh
    git clone https://github.com/QhenryQ/nteprsm.git
    cd nteprsm
    ```

5. **Open VS Code and Open Terminal**:

    - Open Visual Studio Code.
    - Navigate to the cloned `nteprsm` repository folder.
    - Open a terminal within VS Code.

6. **Run the Model**:
    ```sh
    nteprsm/model.py config/nteprsm_in1kbg07.yml
    ```

    The process will start, and you will need to wait until iterations are done. After the process completes, you will have your CSV files ready, which can later be used in Jupyter Notebooks.

## Troubleshooting

If you encounter any issues, ensure that all dependencies are installed correctly and that you are in the correct directory in the terminal. For further assistance, please refer to the project's documentation.
