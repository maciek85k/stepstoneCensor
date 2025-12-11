#!/usr/bin/env Rscript
# ------------------------------------------------------------
# anonymise_data.R (macOS optimized - Token Classification)
#
# PURPOSE
#   Censor (anonymise) the text stored in the column `info_1`
#   of a data.table by calling the Hugging‑Face model
#   ai4privacy/llama-ai4privacy-multilingual-anonymiser-openpii
#   This model uses TOKEN CLASSIFICATION to detect and mask PII.
#
# USAGE
#   Rscript anonymise_data.R <input.csv> <output.csv>
#
# DEPENDENCIES
#   • R (>= 4.0) with packages: data.table, reticulate
#   • Homebrew Python 3.12 at /opt/homebrew/bin/python3.12
#
# ------------------------------------------------------------

# ------------------------------------------------------------
# 0. Command‑line arguments
# ------------------------------------------------------------
# args <- commandArgs(trailingOnly = TRUE)
# if (length(args) < 2) {
#   stop(
#     "Usage: Rscript anonymise_data.R <input.csv> <output.csv>\n",
#     "  <input.csv>  path to the raw data file\n",
#     "  <output.csv> path where the censored data will be saved"
#   )
# }

input_path  <- "~/Documents/wigeo-inarbeit/Stepstone/test/KI_Klassifizierer_DK/Stepstone_Combined.json"
output_path <- "~/Documents/wigeo-inarbeit/Stepstone/test/KI_Klassifizierer_DK/stepstone_cen.json"


# ------------------------------------------------------------
# 1. Load R packages (quietly)
# ------------------------------------------------------------
suppressPackageStartupMessages({
  library(data.table)   # fast data handling
  library(reticulate)   # R ↔ Python bridge
})

# ------------------------------------------------------------
# 2. Set up a Python virtual‑environment (macOS + Homebrew Python)
# ------------------------------------------------------------
python_bin <- "/opt/homebrew/bin/python3.12"

if (!file.exists(python_bin)) {
  stop("Cannot find the python interpreter at ", python_bin,
       ". Please verify the path.")
}

# Name and path of the virtual‑env
venv_name <- "privacy_venv"
venv_path <- file.path(getwd(), venv_name)

# Helper: does the venv already exist and is it valid?
venv_exists <- dir.exists(venv_path) && file.exists(file.path(venv_path, "bin", "python"))

if (!venv_exists) {
  message("[1/4] Creating a new virtual‑env '", venv_name, "' ...")
  
  # Remove old incomplete venv if it exists
  if (dir.exists(venv_path)) {
    message("   Removing incomplete venv...")
    unlink(venv_path, recursive = TRUE)
  }
  
  # Create venv using system command (more reliable on macOS)
  cmd <- paste(python_bin, "-m venv", venv_path)
  result <- system(cmd, intern = FALSE)
  
  if (result != 0) {
    stop("Failed to create virtual environment. Try running in terminal:\n",
         cmd)
  }
  
  # Verify it was created
  if (!file.exists(file.path(venv_path, "bin", "python"))) {
    stop("Virtual environment created but python binary not found!")
  }
  
  message("   Virtual environment created successfully!")
} else {
  message("[1/4] Re‑using existing virtual‑env '", venv_name, "'.")
}

# ------------------------------------------------------------
# 3. Install Python packages (macOS-friendly approach)
# ------------------------------------------------------------
message("[2/4] Installing Python packages in the virtual‑env ...")
message("      This may take several minutes on first run...")

# Get pip and python paths from the venv
pip_path <- file.path(venv_path, "bin", "pip")
venv_python <- file.path(venv_path, "bin", "python")

# First, upgrade pip itself
message("   Upgrading pip...")
cmd_pip_upgrade <- paste(venv_python, "-m pip install --upgrade pip")
system(cmd_pip_upgrade, intern = FALSE)

# Install packages one by one
required_pkgs <- c(
  "torch",
  "transformers",
  "sentencepiece",
  "accelerate",
  "huggingface_hub"
)

for (pkg in required_pkgs) {
  message("   Installing ", pkg, "...")
  
  # Use the venv's python to install packages
  cmd <- paste(venv_python, "-m pip install", pkg, "--no-cache-dir")
  result <- system(cmd, intern = FALSE, ignore.stderr = FALSE)
  
  if (result != 0) {
    warning("Failed to install ", pkg, ". Trying without --no-cache-dir...")
    cmd_retry <- paste(venv_python, "-m pip install", pkg)
    result_retry <- system(cmd_retry, intern = FALSE)
    
    if (result_retry != 0) {
      stop("Could not install ", pkg, ". Error code: ", result_retry)
    }
  }
}

message("   All packages installed successfully!")

# ------------------------------------------------------------
# 4. Tell reticulate to use this venv
# ------------------------------------------------------------
reticulate::use_virtualenv(venv_path, required = TRUE)

# Verify Python is working
py_config <- reticulate::py_config()
message("   Using Python: ", py_config$python)

# ------------------------------------------------------------
# 5. Import Python modules & build the anonymisation pipeline
# ------------------------------------------------------------
message("[3/4] Loading the anonymisation model...")
message("      (This is a TOKEN CLASSIFICATION model)")

# Set environment variables to prevent threading issues
Sys.setenv(OMP_NUM_THREADS = "1")
Sys.setenv(MKL_NUM_THREADS = "1")
Sys.setenv(OPENBLAS_NUM_THREADS = "1")

tryCatch({
  py_run_string("
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Prevent threading issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# Model identifier on the Hugging‑Face Hub
model_name = 'ai4privacy/llama-ai4privacy-multilingual-anonymiser-openpii'

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name)

print('Loading model (this may take a moment)...')
# NOTE: This is a TOKEN CLASSIFICATION model, not Seq2Seq!
model = AutoModelForTokenClassification.from_pretrained(model_name)

# macOS: use CPU
device = -1  # CPU

print('Creating NER pipeline...')
# Use token-classification pipeline for PII detection
ner_pipe = pipeline(
    'token-classification',
    model=model,
    tokenizer=tokenizer,
    device=device,
    aggregation_strategy='simple'  # Merge subword tokens
)

def anonymise_batch(texts):
    '''
    Anonymise a list (or a single string) of texts by detecting
    PII entities and replacing them with [MASKED] tokens.
    
    Parameters
    ----------
    texts : list[str] or str
        Raw text(s) that need to be censored.
    
    Returns
    -------
    tuple: (list[str], list[bool])
        - Censored version(s) of the input
        - Boolean flags indicating if censoring occurred
    '''
    # Ensure we always feed a list
    if not isinstance(texts, list):
        texts = [texts]
    
    # Filter out None and empty strings
    texts_clean = [str(t) if t is not None else '' for t in texts]
    
    results = []
    censored_flags = []
    
    for text in texts_clean:
        if not text or text.strip() == '':
            results.append('')
            censored_flags.append(False)
            continue
        
        try:
            # Get PII entities from the model
            entities = ner_pipe(text)
            
            # Check if any PII was found
            has_pii = len(entities) > 0
            censored_flags.append(has_pii)
            
            if not has_pii:
                results.append(text)
                continue
            
            # Sort entities by start position (in reverse) to replace from end to start
            # This prevents index shifting issues
            entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)
            
            # Replace each PII entity with [MASKED]
            masked_text = text
            for entity in entities_sorted:
                start = entity['start']
                end = entity['end']
                entity_type = entity['entity_group']
                
                # Replace the PII with [MASKED_ENTITY_TYPE]
                masked_text = masked_text[:start] + f'[MASKED_{entity_type}]' + masked_text[end:]
            
            results.append(masked_text)
        except Exception as e:
            print(f'Error processing text: {str(e)[:100]}')
            # Return original text on error
            results.append(text)
            censored_flags.append(False)
    
    return (results, censored_flags)

print('Model loaded successfully!')
")
}, error = function(e) {
  stop("Failed to load the anonymisation model: ", e$message,
       "\n\nMake sure all packages are installed. You can test manually:\n",
       "  ", venv_python, " -c 'import torch; import transformers'")
})

# Export the Python helper
anonymise_batch <- py$anonymise_batch

# ------------------------------------------------------------
# 6. Load the data (supports CSV and JSON)
# ------------------------------------------------------------
message("[4/4] Reading data from: ", input_path)

tryCatch({
  # Check file extension
  file_ext <- tolower(tools::file_ext(input_path))
  
  if (file_ext == "json") {
    # Load JSON file
    message("   Detected JSON file, loading with jsonlite...")
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
      stop("Package 'jsonlite' is required for JSON files. Install with: install.packages('jsonlite')")
    }
    json_data <- jsonlite::fromJSON(input_path, flatten = TRUE)
    
    # Convert to data.table
    if (is.data.frame(json_data)) {
      dt <- as.data.table(json_data)
    } else if (is.list(json_data)) {
      dt <- as.data.table(json_data)
    } else {
      stop("JSON file format not recognized. Expected a data frame or list structure.")
    }
  } else {
    # Assume CSV or other fread-compatible format
    dt <- fread(input_path)
  }
}, error = function(e) {
  stop("Failed to read input file: ", e$message)
})

# Sanity check
if (!"info_1" %in% names(dt)) {
  stop("Column 'info_1' not found in the input data.")
}

# ------------------------------------------------------------
# 7. Anonymise `info_1` in batches
# ------------------------------------------------------------
batch_size <- 8  # SMALLER batch for RStudio stability
n_rows     <- nrow(dt)

# Pre‑allocate
censored_vec <- vector("character", n_rows)
censored_flag_vec <- vector("logical", n_rows)

# Progress bar
pb <- txtProgressBar(min = 0, max = n_rows, style = 3)

message("Running the anonymiser (batch size = ", batch_size, ") ...")
message("Depending on your data size, this may take a while...")
message("NOTE: Processing one text at a time to prevent RStudio crashes")

for (start_idx in seq(1, n_rows, by = batch_size)) {
  end_idx <- min(start_idx + batch_size - 1, n_rows)
  idx     <- start_idx:end_idx
  
  # Extract the raw texts
  raw_batch <- dt$info_1[idx]
  
  # Handle NAs and empty strings
  raw_batch[is.na(raw_batch)] <- ""
  
  # Call Python function with error handling
  # Process one at a time to prevent crashes
  for (i in seq_along(idx)) {
    tryCatch({
      single_text <- raw_batch[i]
      result <- anonymise_batch(single_text)
      censored_vec[idx[i]] <- result[[1]][1]
      censored_flag_vec[idx[i]] <- result[[2]][1]
    }, error = function(e) {
      warning("Error at row ", idx[i], ": ", e$message)
      # Keep original text if anonymisation fails
      censored_vec[idx[i]] <- raw_batch[i]
      censored_flag_vec[idx[i]] <- FALSE
    })
  }
  
  # Update progress
  setTxtProgressBar(pb, end_idx)
  
  # Force garbage collection every few batches
  if (start_idx %% 50 == 0) {
    gc(verbose = FALSE)
  }
}
close(pb)

# ------------------------------------------------------------
# 8. Add the censored column and flag back
# ------------------------------------------------------------
dt[, info_1_censored := censored_vec]
dt[, was_censored := censored_flag_vec]

# ------------------------------------------------------------
# 9. Write the censored table to disk (as JSON)
# ------------------------------------------------------------
message("Saving censored data to: ", output_path)

tryCatch({
  # Check output format
  file_ext <- tolower(tools::file_ext(output_path))
  
  # Clean up list columns before saving
  # Convert any list columns to JSON strings or handle NULLs
  for (col in names(dt)) {
    if (is.list(dt[[col]])) {
      message("   Converting list column '", col, "' to character...")
      dt[[col]] <- sapply(dt[[col]], function(x) {
        if (is.null(x)) {
          return(NA_character_)
        } else if (length(x) == 0) {
          return(NA_character_)
        } else if (length(x) == 1 && is.atomic(x)) {
          return(as.character(x))
        } else {
          # Convert complex lists to JSON strings
          return(jsonlite::toJSON(x, auto_unbox = TRUE))
        }
      })
    }
  }
  
  if (file_ext == "json") {
    # Save as JSON
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
      stop("Package 'jsonlite' is required for JSON output. Install with: install.packages('jsonlite')")
    }
    jsonlite::write_json(dt, output_path, pretty = TRUE, auto_unbox = TRUE, na = "null")
    message("✅ Finished! Censored table written to ", output_path, " (JSON format)")
  } else {
    # Save as CSV
    fwrite(dt, output_path, na = "")
    message("✅ Finished! Censored table written to ", output_path, " (CSV format)")
  }
  
  # Summary statistics
  n_censored <- sum(censored_flag_vec, na.rm = TRUE)
  pct_censored <- round(100 * n_censored / n_rows, 1)
  message("Summary: ", n_censored, " out of ", n_rows, " rows (", pct_censored, "%) contained PII and were censored")
  message("New columns added:")
  message("  - info_1_censored: The anonymised text")
  message("  - was_censored: TRUE if PII was detected and masked")
}, error = function(e) {
  stop("Failed to write output file: ", e$message)
})