# CTS environmental prediction for human-only PPCP parents
# Run from SC_CLM repo root

library(patRoon)
library(data.table)

# Load parents
parents_df <- read.csv("scratch/cts_input_human_only_parents.csv", stringsAsFactors = FALSE)
parents <- parents_df$SMILES
cat(sprintf("Loaded %d parent SMILES\n", length(parents)))

# Source the export utilities
source("scratch/patoon_tp_unify.R")

# Define tp_to_dt function (required by patoon_tp_unify.R)
tp_to_dt <- function(tp_result) {
  if (is.null(tp_result) || length(tp_result) == 0) return(data.table())
  dt <- as.data.table(tp_result, parents = TRUE)
  if (nrow(dt) == 0) return(dt)
  # Add likelihood from patRoon scoring if available
  if ("score" %in% names(dt)) {
    dt[, likelihood := fifelse(score >= 0.5, "LIKELY", "UNLIKELY")]
  }
  dt
}

timestamp <- format(Sys.time(), "%Y%m%d")
out_dir <- "results/benchmark/v5/patoon_cts_augmented"

cat("\n=== Running CTS: photolysis_ranked ===\n")
t_start <- Sys.time()
TPs_cts_PHr <- tryCatch({
  generateTPs(parents, type = "cts", 
              ctsPathways = "photolysis_ranked",
              skipInvalid = TRUE)
}, error = function(e) {
  cat("ERROR:", e$message, "\n")
  NULL
})
cat(sprintf("Time: %.1f min\n", difftime(Sys.time(), t_start, units = "mins")))
cat(sprintf("Results: %s\n", if(is.null(TPs_cts_PHr)) "NULL" else paste(class(TPs_cts_PHr), length(TPs_cts_PHr))))

cat("\n=== Running CTS: hydrolysis ===\n")
t_start <- Sys.time()
TPs_cts_HY <- tryCatch({
  generateTPs(parents, type = "cts",
              ctsPathways = "hydrolysis",
              skipInvalid = TRUE)
}, error = function(e) {
  cat("ERROR:", e$message, "\n")
  NULL
})
cat(sprintf("Time: %.1f min\n", difftime(Sys.time(), t_start, units = "mins")))

cat("\n=== Running CTS: abiotic_reduction ===\n")
t_start <- Sys.time()
TPs_cts_AR <- tryCatch({
  generateTPs(parents, type = "cts",
              ctsPathways = "abiotic_reduction",
              skipInvalid = TRUE)
}, error = function(e) {
  cat("ERROR:", e$message, "\n")
  NULL
})
cat(sprintf("Time: %.1f min\n", difftime(Sys.time(), t_start, units = "mins")))

# Export results
cat("\n=== Exporting results ===\n")
spec <- data.frame(
  name = c("TPs_cts_PHr", "TPs_cts_HY", "TPs_cts_AR"),
  type_label = c("photolysis_ranked", "hydrolysis", "abiotic_reduction"),
  source = c("cts", "cts", "cts"),
  pathway_id = c("cts_photolysis_ranked_aug", "cts_hydrolysis_aug", "cts_abiotic_reduction_aug"),
  stringsAsFactors = FALSE
)

tp_list <- list(
  TPs_cts_PHr = TPs_cts_PHr,
  TPs_cts_HY = TPs_cts_HY,
  TPs_cts_AR = TPs_cts_AR
)

paths <- export_all_patoon_pathways(tp_list, spec, out_dir, timestamp)
cat(sprintf("Results saved to: %s\n", out_dir))
cat("DONE\n")
