### Code for Naved Fardeen Khan (2026). "Simulating Human Developmental Speech: Fine-Tuning LLMs for Age-Appropriate Parental Dialogue", Research Training Project.

Program code for fine-tuning a Phi-3 language model (LM) from CHILDES transcripts using Low-Rank Adaptation (LoRA) and generating/evaluating age-appropriate child-directed speech.

### Contents:

* **LoRA.py**: The main script for fine-tuning the `microsoft/Phi-3-mini-4k-instruct` model using LoRA on the Puhti HPC cluster.
* **CHILDES_dataset_test.py**: Script to extract and sample real-world data from the CHILDES database into age-specific bins, calculating human baseline Lexical Diversity (TTR) and exporting reference CSVs.
* **zero_shot_test.py**: Script to evaluate the base (untrained) Phi-3 model. Generates baseline child-directed speech across target age bins, calculates initial TTR, and exports outputs.
* **test_model.py**: Script to load the fine-tuned LoRA adapter, generate age-appropriate transcripts, and calculate the post-training TTR metrics.
* **combine_csv.py**: Script to aggregate multiple generated CSV files into three unified datasets for analysis.
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
4.  **Generate Data for Evaluation:**
    * Run `zero_shot_test.py` to generate the generic AI baseline transcripts.
    * Run `test_model.py` to generate the fine-tuned, age-adapted transcripts.
    * Run `CHILDES_dataset_test.py` to sample the real-world dataset.
    ```bash
    python zero_shot_test.py
    python test_model.py
    python CHILDES_dataset_test.py
    ```
    Run multiple times.

5.  **Combine CSV files:** Run `combine_csv.py` to combine multiple generated CSV files into three unified datasets before running the final evaluations. 
    ```bash
    python combine_csv.py
    ```
6.  **Evaluate Output:** Run `perplexity.py` to calculate the GPT-2 evaluation metrics on the generated transcripts versus the human baseline texts.
    ```bash
    python perplexity.py
    ```

7.  **Visualize Results:** Run `triple_plot.py` to generate the feature comparison graphs (Lexical Diversity, Utterance Length, Generative Naturalness) of the generated transcripts against original transcripts.
    ```bash
    python triple_plot.py
    ```
