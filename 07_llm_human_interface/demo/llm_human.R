###############################################################################
# LLM-Assisted Data Extraction (Human-in-the-Loop)
# R Version (Using httr2 + local API or ellmer/tidyllm)
# Structured Extraction from Messy Text (R)
# Author: Jared Edgerton
# Date: Sys.Date()
#
# Teaching version:
# - Explicit, sequential steps
# - No user-defined functions
#
# Install (if needed):
#   install.packages(c("tidyverse", "jsonlite", "httr2"))
#
# NOTE: This script demonstrates the extraction pipeline structure using
# regex-based parsing as a self-contained fallback. For actual LLM inference
# in R, students can use the ellmer or tidyllm packages with a local model
# endpoint (e.g., Ollama running Qwen2.5-1.5B-Instruct).
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
library(tidyverse)
library(jsonlite)

set.seed(123)

# -----------------------------------------------------------------------------
# Part 1: Define a Schema (What fields do we want?)
# -----------------------------------------------------------------------------
# We want to extract from each text:
#   - event_type: protest, election, policy_change, violence, disaster, other
#   - event_date_iso: YYYY-MM-DD or NA
#   - date_is_approximate: TRUE/FALSE
#   - country: character or NA
#   - admin1_or_state: character or NA
#   - city_or_local: character or NA
#   - geo_precision: country_only, admin1_or_state, city_or_local, unknown
#   - actors: comma-separated list
#   - outcome_summary: one sentence
#   - extraction_confidence: 0 to 1
#   - uncertainty_flags: list of issues

# Allowed values
valid_event_types  <- c("protest", "election", "policy_change",
                        "violence", "disaster", "other")
valid_geo_precisions <- c("country_only", "admin1_or_state",
                          "city_or_local", "unknown")

# -----------------------------------------------------------------------------
# Part 2: Create Messy Text Inputs (Mini Corpus)
# -----------------------------------------------------------------------------
docs <- tibble(
  doc_id = paste0("doc_", sprintf("%03d", 1:15)),
  text = c(
    "Breaking: Thousands rallied in Santiago on 2026-03-14 demanding pension reform. Police reported minor clashes; 12 were arrested.",
    "On March 2nd, lawmakers passed the 'Clean Air Act' amendment in the national assembly. Environmental groups praised the vote.",
    "Election officials said voting will take place next Sunday. Turnout is expected to be high in the capital.",
    "A 6.2 magnitude earthquake struck near the coastal city overnight, damaging dozens of homes and cutting power to 40,000 residents.",
    "Witnesses described gunfire outside a nightclub late Friday; at least two people were injured, but details remain unclear.",
    "The governor announced a new curfew order effective immediately. Critics called it an overreach.",
    "Early April saw renewed demonstrations in the northern province after fuel prices rose again.",
    "Floodwaters inundated low-lying neighborhoods; emergency shelters opened at local schools, officials said.",
    "Opposition leaders met with international observers in Brussels to discuss election monitoring.",
    "Police said the suspect was arrested after a stabbing in downtown; the mayor urged calm.",
    "Parliament reversed the prior ban on rideshare apps, citing labor market flexibility.",
    "A protest was planned for tomorrow, but organizers postponed it due to severe weather warnings.",
    "Following a landslide, the ministry declared a state of emergency in two districts.",
    "The court ruling sparked demonstrations across the city center; human rights groups condemned the decision.",
    "The article mentions reforms and elections in passing but gives no clear time or place."
  )
)

cat("\n------------------------------\n")
cat("Input corpus (first 5 docs)\n")
cat("------------------------------\n")
print(head(docs, 5))
cat("docs shape:", nrow(docs), "x", ncol(docs), "\n")

# -----------------------------------------------------------------------------
# Part 3: Prompt Design (Schema + Guardrails)
# -----------------------------------------------------------------------------
# This is the instruction template we would send to a local LLM.
# In R, we store it as a character string.

system_instructions <- paste(
  "Task: Extract ONE event record from the text.",
  "Return EXACTLY the following 9 lines, one per line, in the format key: value",
  "Use empty value if unknown.",
  "",
  "event_type: protest|election|policy_change|violence|disaster|other",
  "event_date_iso: YYYY-MM-DD",
  "date_is_approximate: true|false",
  "country:",
  "admin1_or_state:",
  "city_or_local:",
  "geo_precision: country_only|admin1_or_state|city_or_local|unknown",
  "actors: comma-separated list",
  "outcome_summary: one sentence",
  "",
  "Do not output anything else.",
  sep = "\n"
)

cat("\n------------------------------\n")
cat("System instructions:\n")
cat("------------------------------\n")
cat(system_instructions, "\n")

# -----------------------------------------------------------------------------
# Part 4: Extraction via Local LLM API (or regex fallback)
# -----------------------------------------------------------------------------
# Teaching note:
# In a real workflow, you would call a local LLM (e.g., via Ollama) like:
#
#   library(httr2)
#   resp <- request("http://localhost:11434/api/generate") |>
#     req_body_json(list(model = "qwen2.5:1.5b", prompt = full_prompt)) |>
#     req_perform()
#
# Here we use keyword-based extraction as a deterministic fallback,
# so the pipeline structure is fully visible without requiring a running model.

# Keyword patterns for event type detection
event_keywords <- list(
  protest       = c("rally", "rallied", "protest", "demonstration", "march"),
  election      = c("election", "voting", "vote", "ballot"),
  policy_change = c("passed", "amendment", "ban", "curfew", "order", "reversed",
                     "parliament", "lawmakers"),
  violence      = c("gunfire", "stabbing", "shooting", "attack", "injured"),
  disaster      = c("earthquake", "flood", "landslide", "hurricane", "storm")
)

# Date pattern
date_pattern <- "\\d{4}-\\d{2}-\\d{2}"

# Storage
extractions <- vector("list", nrow(docs))

cat("\n------------------------------\n")
cat("Running extraction (keyword-based fallback)\n")
cat("------------------------------\n")

for (i in seq_len(nrow(docs))) {
  doc_id <- docs$doc_id[i]
  txt    <- docs$text[i]
  txt_lower <- tolower(txt)

  # Detect event type
  event_type <- "other"
  for (etype in names(event_keywords)) {
    keywords <- event_keywords[[etype]]
    if (any(str_detect(txt_lower, fixed(keywords)))) {
      event_type <- etype
      break
    }
  }

  # Extract date
  date_match <- str_extract(txt, date_pattern)
  event_date_iso <- ifelse(is.na(date_match), NA_character_, date_match)
  date_is_approximate <- is.na(event_date_iso)

  # Location (simple: check for known city/country keywords)
  country <- NA_character_
  city_or_local <- NA_character_
  geo_precision <- "unknown"

  if (str_detect(txt_lower, "santiago")) {
    city_or_local <- "Santiago"
    country <- "Chile"
    geo_precision <- "city_or_local"
  } else if (str_detect(txt_lower, "brussels")) {
    city_or_local <- "Brussels"
    country <- "Belgium"
    geo_precision <- "city_or_local"
  }

  # Actors (simple: extract capitalized multi-word sequences)
  actors_raw <- str_extract_all(txt, "[A-Z][a-z]+(\\s[A-Z][a-z]+)*")[[1]]
  actors_raw <- actors_raw[nchar(actors_raw) > 3]
  actors <- paste(unique(actors_raw), collapse = ", ")

  # Outcome summary (use first sentence as proxy)
  outcome_summary <- str_extract(txt, "^[^.]+\\.")

  # Confidence
  parse_ok <- TRUE
  parse_flags <- character(0)

  if (event_type == "other") parse_flags <- c(parse_flags, "ambiguous_event_type")
  if (is.na(event_date_iso)) parse_flags <- c(parse_flags, "missing_date")
  if (is.na(country)) parse_flags <- c(parse_flags, "missing_country")

  confidence <- 0.6
  if (length(parse_flags) > 0) confidence <- 0.35

  extractions[[i]] <- tibble(
    doc_id               = doc_id,
    event_type           = event_type,
    event_date_iso       = event_date_iso,
    date_is_approximate  = date_is_approximate,
    country              = country,
    admin1_or_state      = NA_character_,
    city_or_local        = city_or_local,
    geo_precision        = geo_precision,
    actors               = actors,
    outcome_summary      = outcome_summary,
    extraction_confidence = confidence,
    uncertainty_flags    = paste(parse_flags, collapse = "; "),
    raw_text             = txt,
    parse_ok             = parse_ok
  )
}

extractions_df <- bind_rows(extractions)

cat("\n------------------------------\n")
cat("Extracted records (first 5 rows)\n")
cat("------------------------------\n")
print(head(extractions_df %>% select(doc_id, event_type, event_date_iso, country), 5))
cat("extractions_df shape:", nrow(extractions_df), "x", ncol(extractions_df), "\n")

dir.create("outputs", recursive = TRUE, showWarnings = FALSE)
write_csv(extractions_df, "outputs/extractions_raw.csv")

# -----------------------------------------------------------------------------
# Part 5: Uncertainty Checks (Automatic Flags for Human Review)
# -----------------------------------------------------------------------------
extractions_df <- extractions_df %>%
  mutate(
    flag_low_confidence  = extraction_confidence < 0.70,
    flag_missing_date    = is.na(event_date_iso),
    flag_missing_country = is.na(country),
    flag_geo_unknown     = geo_precision %in% c("unknown", "country_only"),
    needs_human_review   = flag_low_confidence | flag_missing_date |
                           flag_missing_country | flag_geo_unknown
  )

cat("\n------------------------------\n")
cat("Review flag counts\n")
cat("------------------------------\n")
cat("flag_low_confidence: ", sum(extractions_df$flag_low_confidence), "\n")
cat("flag_missing_date:   ", sum(extractions_df$flag_missing_date), "\n")
cat("flag_missing_country:", sum(extractions_df$flag_missing_country), "\n")
cat("flag_geo_unknown:    ", sum(extractions_df$flag_geo_unknown), "\n")
cat("needs_human_review:  ", sum(extractions_df$needs_human_review), "\n")

write_csv(extractions_df, "outputs/extractions_with_flags.csv")

# -----------------------------------------------------------------------------
# Part 6: Human Validation / Spot-Audits (Create an Audit Sheet)
# -----------------------------------------------------------------------------
audit_random <- extractions_df %>% slice_sample(n = 5)
audit_flagged <- extractions_df %>% filter(needs_human_review)

audit_sheet <- bind_rows(audit_random, audit_flagged) %>%
  distinct(doc_id, .keep_all = TRUE) %>%
  arrange(doc_id) %>%
  mutate(
    human_is_correct         = "",
    human_correct_event_type = "",
    human_correct_date_iso   = "",
    human_correct_location   = "",
    failure_mode             = "",
    reviewer_notes           = ""
  )

write_csv(audit_sheet, "outputs/human_audit_sheet.csv")

cat("\n------------------------------\n")
cat("Wrote outputs/human_audit_sheet.csv\n")
cat("------------------------------\n")

# -----------------------------------------------------------------------------
# Part 7: Evaluation Patterns (Precision/Recall + Auditing)
# -----------------------------------------------------------------------------
gold <- tibble(
  doc_id          = paste0("doc_", sprintf("%03d", 1:8)),
  event_type_gold = c("protest", "policy_change", "election", "disaster",
                       "violence", "policy_change", "protest", "disaster")
)

eval_df <- gold %>%
  left_join(extractions_df %>% select(doc_id, event_type), by = "doc_id") %>%
  rename(event_type_pred = event_type) %>%
  mutate(event_type_pred = replace_na(event_type_pred, "MISSING"))

cat("\n------------------------------\n")
cat("Evaluation table (gold vs predicted)\n")
cat("------------------------------\n")
print(eval_df)

# Accuracy
accuracy <- mean(eval_df$event_type_gold == eval_df$event_type_pred)
cat("\nOverall accuracy:", round(accuracy, 3), "\n")

# Confusion matrix
cat("\n------------------------------\n")
cat("Confusion matrix (event_type)\n")
cat("------------------------------\n")
print(table(True = eval_df$event_type_gold, Predicted = eval_df$event_type_pred))
