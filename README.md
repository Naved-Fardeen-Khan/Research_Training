### Code for Naved Fardeen Khan (2026). "Simulating Human Developmental Speech: Fine-Tuning LLMs for Age-Appropriate Parental Dialogue", Research Training Project.

Program code for fine-tuning a Phi-3 language model (LM) from CHILDES transcripts using Low-Rank Adaptation (LoRA) and generating/evaluating age-appropriate child-directed speech.

### Contents:

* **LoRA.py**: The main script for fine-tuning the `microsoft/Phi-3-mini-4k-instruct` model using LoRA on the Puhti HPC cluster.
* **perplexity.py**: Script for calculating unbiased generative perplexity scores for the generated texts and original CHILDES texts using an independent GPT-2 judge.
* **triple_plot.py**: Plotting the results based on the corpus analyses (TTR, MLU, and Perplexity comparisons) using Matplotlib and Seaborn.
* **requirements.txt**: List of Python dependencies required to run the pipeline.

### Main dependencies:

* Python
* For LM training and generation:
    * `torch`
    * `transformers`
    * `peft`
    * `accelerate`
    * `bitsandbytes`
* For evaluation of the generated data and plotting:
    * `pandas`
    * `numpy`
    * `matplotlib`
    * `seaborn`
    * `tqdm`

### Instructions:

1.  **Setup the environment:** Install the required Python packages using the provided requirements file.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Preparation:** Prepare the extracted CHILDES transcripts as CSV files categorized by age bins (3 to 84 months). Ensure the paths to these files are correctly set inside the training script.

3.  **Train the Model:** Run `LoRA.py` to fine-tune the Phi-3 model using prepared datasets. *(Note: If running on CSC Puhti, submit this script via SLURM batch file).*
    ```bash
    python LoRA.py
    ```

4.  **Evaluate Output:** Run `perplexity.py` to calculate the GPT-2 evaluation metrics on the generated transcripts versus the human baseline texts.
    ```bash
    python perplexity.py
    ```

5.  **Visualize Results:** Run `triple_plot.py` to generate the feature comparison graphs (Lexical Diversity, Utterance Length, Generative Naturalness) of the generated transcripts against original transcripts.
    ```bash
    python triple_plot.py
    ```
