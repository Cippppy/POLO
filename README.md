# YOLO Multi-Object Color Attack (YMCA)
Created by Christian Cipolletta at Rowan University during the Spring 2024 Semester under Dr. Robi Polikar.

## Description

In this study, we focus on designing a method to enhance the robustness and safety of computer vision systems, with a specific emphasis on YOLO models. Our objective is to assess the susceptibility of YOLO models to artificial coloring attacks and quantify their impact on model performance. Through the development of a methodology, we aim to identify and analyze the vulnerabilities of YOLO models to color-based perturbations. By doing so, we seek to contribute to the advancement of more resilient and reliable computer vision systems. We hypothesize that YOLO predictions are influenced by color variations, thereby suggesting a potential avenue for improving model robustness against adversarial attacks.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [License](#license)
4. [Contact](#contact)
5. [Appendix](#appendix-additional-links)

## Installation

To get started with this project, follow these steps:

1. **Clone the Repository**: Begin by cloning this repository to your local machine using Git:

    ```bash
    git clone https://github.com/Cippppy/YMCA
    ```

2. **Navigate to the Project Directory**: Change into the project directory:

    ```bash
    cd YMCA
    ```

3. **Install Dependencies**: Install the required dependencies using pip. It's recommended to use a virtual environment to manage dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download Data**: Download the datasets required for this project from Google Drive. Follow the links below to access the datasets:
    - [Segment](https://drive.google.com/drive/folders/1aD2zYarCA-sAvx8O0bFugoG9NcBixxJY?usp=sharing)
    - [Detect](https://drive.google.com/drive/folders/1S9DrhHAzD7lbrIVZVIJdXGrYDtDC9qKn?usp=sharing)

   Once downloaded, extract the datasets and place them in the `datasets` folder within the project directory. The "_colored" image folders are supposed to be empty.

5. **Ready to Use**: You're all set! You can now proceed to use the project as described in the [Usage](#usage) section of this README.

If you encounter any issues during the installation process or have questions, feel free to reach out for assistance.


## Usage

To use this project like we did in the paper all you need to do is use the command below:

```bash
python main.py
```

The scripts to create the figures and results from the paper are in the "results" directory. "plot.py" will create the figures. "runtime.py" will output the total runtime of the model's predictions. The total runtime is longer because of the coloring step.

```bash
python results/plot.py
```

```bash
python results/runtime.py
```

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive open-source license that allows you to use, modify, and distribute the code for both commercial and non-commercial purposes. You are free to incorporate this project into your own work, subject to the terms and conditions outlined in the license file.

By using or contributing to this project, you agree to abide by the terms of the MIT License.


## Contact

Feel free to reach out to me with any questions, feedback, or support requests!

- LinkedIn: [Christian Cipolletta](https://www.linkedin.com/in/christian-cipolletta/)
- GitHub: [Cippppy](https://github.com/Cippppy)
- School Email: cipoll17@students.rowan.edu
- Personal Email: cjcipbiz@gmail.com


## Appendix: Additional Links

Here are some additional resources and links related to this project:

- [Project Repository](https://github.com/Cippppy/YMCA): The GitHub repository for this project.
- [Weights & Biases](https://wandb.ai/cippppy/YOLO%20Multiobject%20Color%20Attack%20(YMCA)): This Wandb repository holds all the data from the runs used in the results.
- [Project Paper](https://drive.google.com/file/d/1gzRx54RBFELdnS_dpAAmjLfFJoAHSzyB/view?usp=sharing): This is the paper written for this project.
- [Project Poster](https://drive.google.com/file/d/1bjkANTcSbMgCHXR9tZdOMMiBNuUZ8Tg3/view?usp=sharing): This is the poster created for this project.

Feel free to explore these links for more information about the project.
