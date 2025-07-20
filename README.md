# LATUC Corpus

This corpus includes 36 Ottoman Turkish poem books (_Dîvân_), written between the 15th and 19th centuries. The books were transliterated by domain experts and publicly shared on the Internet. The books in the corpus were automatically structured via a rule‑based approach and manually checked.

See the interface for free: https://koras0ff.github.io/latuc/

## Corpus Overview

The corpus has more than 1 million words from 36 authors who wrote between 15th and 19th centuries. 

## Data Files

### Work‑level metadata (`LATUC_metadata.csv`)
Each Dîvân work is accompanied by:

- `file_name`  
- `work_name` (title of the Dîvân)  
- `pen_name` (mahlas)  
- `real_name`  
- `viaf`  
- `century`  
- `gender`  
- `rank`  
  - e.g. “Sultan,” “Judiciary & Religious Office,” “High Bureaucracy/Military,”  
    “Scholars & Sufi Orders,” “Civil Bureaucracy,” “Lay/Non‑official”

### Poem‑level data (`LATUC.json`)
Each individual poem includes:

- `poem_id`  
- `title` (if available)  
- `meter` (in aruz notation)  
- `text` (line‑by‑line Latin transliteration)  
- `tags` (optional part‑of‑speech sequences)

## Usage Notes

- The corpus can be utilized for **diachronic studies**; Yılandiloğlu (forthcoming) demonstrated that poets adhered more accurately to the aruz meter over the centuries, reflected in rising conformity rates.  
- Metadata allows you to focus on specific ranks (e.g. “Sultan”) or gender.  
- Current work is focused on standardizing transliteration to the IJMES system and expanding the corpus further.  
