# PII Anonymisation Tool

Automatic detection and anonymisation of Personally Identifiable Information (PII) in text data using AI.

## 📋 Overview

This R script uses the Hugging Face model [`ai4privacy/llama-ai4privacy-multilingual-anonymiser-openpii`](https://huggingface.co/ai4privacy/llama-ai4privacy-multilingual-anonymiser-openpii) to automatically detect and mask personal information in text.

**Supported Languages:** German, English, French, Italian, Spanish, Dutch, Hindi, Telugu

**Detected PII Types:**
- Names (First name, Last name)
- Email addresses
- Phone numbers
- Addresses
- Dates of birth
- Social security numbers
- Passport and driver's license numbers
- And 20+ additional categories

## 🚀 Features

- ✅ **Automatic Detection:** Uses state-of-the-art NLP models for PII recognition
- ✅ **Multilingual:** Works with 8 different languages
- ✅ **Apple Silicon Support:** Automatically uses GPU on M1/M2/M3 Macs
- ✅ **Flexible Formats:** Reads and writes CSV and JSON
- ✅ **Tracking:** Marks rows where PII was found
- ✅ **Batch Processing:** Efficient processing of large datasets

## 📦 Installation

### Prerequisites

- macOS with Homebrew
- R (>= 4.0)
- Python 3.12 (via Homebrew)

### Setup

1. **Install Python:**
```bash
brew install python@3.12
```

2. **Install R packages:**
```r
install.packages(c("data.table", "reticulate", "jsonlite"))
```

3. **Make script executable:**
```bash
chmod +x anonymise_data.R
```

The script will automatically create a Python virtual environment and install all required packages (torch, transformers, etc.) on first run.

**Note:** Both script versions (`stepstoneCensor.R` and `stepstoneCensorAppleSilicon.R`) share the same virtual environment and dependencies.

## 💻 Usage

### Choose the Right Script

**Two versions are available:**

- **`stepstoneCensor.R`** - For Intel Macs (CPU-only)
- **`stepstoneCensorAppleSilicon.R`** - For Apple Silicon Macs (M1/M2/M3 with GPU support)

**How to check which Mac you have:**
```bash
sysctl -n machdep.cpu.brand_string
```
- Contains "Intel" → Use `stepstoneCensor.R`
- Contains "Apple" → Use `stepstoneCensorAppleSilicon.R`

### Basic Usage

**Intel Mac:**
```bash
Rscript stepstoneCensor.R input.json output.json
```

**Apple Silicon Mac:**
```bash
Rscript stepstoneCensorAppleSilicon.R input.json output.json
```

### With Default Paths

Both scripts have built-in default paths. Simply run without arguments:

```bash
# Intel Mac
Rscript stepstoneCensor.R

# Apple Silicon Mac
Rscript stepstoneCensorAppleSilicon.R
```

### In RStudio

Open the appropriate script in RStudio and run it. **Note:** For large files, the terminal is more stable.

## 📊 Input/Output Format

### Input

The input file must contain a column named `info_1`:

**JSON Example:**
```json
[
  {
    "id": 1,
    "info_1": "Contact me at john.doe@email.com"
  },
  {
    "id": 2,
    "info_1": "The weather is nice today"
  }
]
```

**CSV Example:**
```csv
id,info_1
1,"Contact me at john.doe@email.com"
2,"The weather is nice today"
```

### Output

The output contains the original data plus two new columns:

- **`info_1_censored`**: The anonymised text with `[MASKED_*]` placeholders
- **`was_censored`**: Boolean flag (TRUE if PII was found)

**Output Example:**
```json
[
  {
    "id": 1,
    "info_1": "Contact me at john.doe@email.com",
    "info_1_censored": "Contact me at [MASKED_EMAIL]",
    "was_censored": true
  },
  {
    "id": 2,
    "info_1": "The weather is nice today",
    "info_1_censored": "The weather is nice today",
    "was_censored": false
  }
]
```

## 🎯 Masking Examples

| Original | Anonymised |
|----------|------------|
| `John Doe` | `[MASKED_GIVENNAME] [MASKED_SURNAME]` |
| `john@example.com` | `[MASKED_EMAIL]` |
| `+1 555 123-4567` | `[MASKED_PHONENUMBER]` |
| `123 Main St, Boston` | `[MASKED_BUILDINGNUMBER] [MASKED_STREET], [MASKED_CITY]` |
| `03/12/1990` | `[MASKED_DATE]` |

## ⚙️ Configuration

### Adjust Batch Size

In the script (line ~290):
```r
batch_size <- 8  # Smaller value = more stable, larger value = faster
```

### Change Python Path

In the script (line ~47):
```r
python_bin <- "/opt/homebrew/bin/python3.12"
```

### GPU/CPU Selection

The script automatically detects Apple Silicon and uses the GPU if available. To force CPU:

In the Python code (line ~160):
```python
device = -1  # -1 = CPU, 'mps' = Apple Silicon GPU
```

## 🐛 Troubleshooting

### RStudio Crashes

→ Run the script in Terminal instead of RStudio:
```bash
# Intel Mac
Rscript stepstoneCensor.R input.json output.json

# Apple Silicon Mac
Rscript stepstoneCensorAppleSilicon.R input.json output.json
```

### "Cannot find python interpreter"

→ Make sure Python 3.12 is installed:
```bash
brew install python@3.12
which python3.12
```

### "Column 'info_1' not found"

→ Your input file must have a column named `info_1`.

### Installation Fails

→ Delete the virtual environment and try again:
```bash
rm -rf privacy_venv
# Then run your script again
Rscript stepstoneCensor.R input.json output.json
# or
Rscript stepstoneCensorAppleSilicon.R input.json output.json
```

### Wrong Script for Your Mac

→ Using Intel script on Apple Silicon (or vice versa):
- Check your Mac type: `sysctl -n machdep.cpu.brand_string`
- Use the correct script for your hardware
- Apple Silicon script will still work on Intel Macs (but slower)

### Very Slow

→ Make sure you're using the right script for your Mac:
- **Apple Silicon (M1/M2/M3)?** → Use `stepstoneCensorAppleSilicon.R`
- **Intel Mac?** → Use `stepstoneCensor.R`

→ Check if GPU is being used (Apple Silicon only):
- Look for "Apple Silicon (MPS) detected!" at startup
- If you see "using CPU" on Apple Silicon → Something went wrong
- GPU version should be 2-3x faster than CPU

## 📈 Performance

**Typical Processing Speed:**

| Mac Type | Speed | Example (1,000 texts) |
|----------|-------|----------------------|
| Apple Silicon (M1/M2/M3) with GPU | ~5-10 texts/sec | 2-3 minutes |
| Intel Mac (CPU-only) | ~2-4 texts/sec | 4-8 minutes |

**Processing Large Datasets:**
- 10,000 texts on Apple Silicon ≈ 20-30 minutes
- 10,000 texts on Intel ≈ 40-80 minutes

**💡 Tip:** Use `stepstoneCensorAppleSilicon.R` on M-series Macs for 2-3x faster processing!

## 🔒 Privacy

- ✅ All processing runs **locally** on your Mac
- ✅ No data is sent to external servers
- ✅ The model is downloaded once and cached locally
- ✅ Original data is preserved (new columns are added)

## 📝 License

This project uses the [ai4privacy model](https://huggingface.co/ai4privacy/llama-ai4privacy-multilingual-anonymiser-openpii) under the MIT License.

## 🤝 Credits

- **Model:** [ai4privacy](https://www.ai4privacy.com/) - Multilingual PII Anonymiser
- **Base Model:** ModernBERT by Answer.AI
- **Framework:** Hugging Face Transformers

## 📧 Support

For questions or issues:
1. Check the Troubleshooting section above
2. See the model documentation: https://huggingface.co/ai4privacy
3. Contact the ai4privacy team: support@ai4privacy.com

---

**Version:** 1.0  
**Last Update:** December 2024  
**Compatible with:** macOS (Apple Silicon & Intel)
