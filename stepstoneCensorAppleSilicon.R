#!/usr/bin/env Rscript
# ------------------------------------------------------------
# stepstoneCensorAppleSiliconV2.R (macOS, Apple Silicon)
#
# PURPOSE
#   Censor (anonymise) the text in column `info_1`.
#   V2 fixes the gap found in the manual review of V1: the
#   ai4privacy model masked ONLY numeric PII (phone numbers,
#   reference numbers) and missed ALL person names and e-mails.
#
#   V2 = three layers, implemented in censorV2.py:
#     1) regex      -> e-mails (incl. scrape-broken forms
#                      like 'name surname@firma com')
#     2) German NER -> person names (PER entities)
#     3) ai4privacy -> phone/reference numbers (as in V1)
#   ORG and LOC are deliberately NOT masked - company names and
#   places are the analytical content of the data.
#
#   RStudio-safe: no reticulate. The models run in a separate
#   Python process; a torch/MPS crash cannot take down R.
#
# USAGE
#   Adjust input_path/output_path, run from the stepstoneCensor
#   folder (censorV2.py must be next to this script).
#   First test on the sample: testV2Sample.R
# ------------------------------------------------------------

input_path  <- "Stepstone_Combined.json"
output_path <- "
stepstone_cen_v2.json"

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
})

# ---- 1. Python venv -----------------------------------------------------------
python_bin  <- "/opt/homebrew/bin/python3.12"
venv_path   <- file.path(getwd(), "privacy_venv")
venv_python <- file.path(venv_path, "bin", "python")

if (!file.exists(venv_python)) {
  message("[1/4] Creating venv ...")
  if (system(paste(python_bin, "-m venv", shQuote(venv_path))) != 0)
    stop("Failed to create venv")
} else {
  message("[1/4] Re-using venv.")
}
message("[2/4] Ensuring Python packages ...")
system(paste(shQuote(venv_python),
             "-m pip install --quiet torch transformers sentencepiece accelerate huggingface_hub"))

# ---- 2. Load data (CSV or JSON) ------------------------------------------------
message("[3/4] Reading data from: ", input_path)
file_ext <- tolower(tools::file_ext(input_path))
if (file_ext == "json") {
  dt <- as.data.table(fromJSON(input_path, flatten = TRUE))
} else {
  dt <- fread(input_path)
}
if (!"info_1" %in% names(dt)) stop("Column 'info_1' not found.")
n_rows <- nrow(dt)

# ---- 3. Censor via subprocess --------------------------------------------------
# info_1 may be a list column after fromJSON; flatten to character
texts <- dt$info_1
if (is.list(texts)) {
  texts <- sapply(texts, function(x)
    if (is.null(x) || length(x) == 0) "" else paste(unlist(x), collapse = " "))
}
texts[is.na(texts)] <- ""

tmp_in  <- tempfile(fileext = ".json")
tmp_out <- tempfile(fileext = ".json")
write_json(texts, tmp_in, auto_unbox = FALSE)

message("[4/4] Censoring ", n_rows, " rows in a Python subprocess ...")
message("      (progress printed by the subprocess)")
status <- system2(venv_python, c("censorV2.py", shQuote(tmp_in), shQuote(tmp_out)))
if (status != 0) stop("censorV2.py failed with exit code ", status,
                      "\nIf MPS is the problem, force CPU:  PII_DEVICE=cpu ",
                      venv_python, " censorV2.py <in> <out>")

res <- fromJSON(tmp_out)
file.remove(tmp_in, tmp_out)

# Replace info_1 by NAME (V1 renamed by position - fragile)
dt[, info_1       := res$censored]
dt[, was_censored := res$flags]

# ---- 4. Write output ------------------------------------------------------------
message("Saving censored data to: ", output_path)
for (col in names(dt)) {
  if (is.list(dt[[col]])) {
    dt[[col]] <- sapply(dt[[col]], function(x) {
      if (is.null(x) || length(x) == 0) return(NA_character_)
      if (length(x) == 1 && is.atomic(x)) return(as.character(x))
      jsonlite::toJSON(x, auto_unbox = TRUE)
    })
  }
}
if (tolower(tools::file_ext(output_path)) == "json") {
  write_json(dt, output_path, pretty = TRUE, auto_unbox = TRUE, na = "null")
} else {
  fwrite(dt, output_path, na = "")
}

n_censored <- sum(res$flags, na.rm = TRUE)
message("Done. ", n_censored, " of ", n_rows, " rows (",
        round(100 * n_censored / n_rows, 1), "%) contained PII.")
