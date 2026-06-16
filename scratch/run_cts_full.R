library(patRoon)
library(data.table)

# Load all parents
parents_df <- read.csv("scratch/cts_input_human_only_parents.csv", stringsAsFactors = FALSE)
parents_df$name <- paste0("P", seq_len(nrow(parents_df)))
cat(sprintf("Loaded %d parents\n", nrow(parents_df)))

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- file.path("results", "cts_augmented")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

process_pathway <- function(parents_df, pathway, label) {
  cat(sprintf("\n=== CTS %s (%d parents) ===\n", label, nrow(parents_df)))
  t_start <- Sys.time()
  
  TPs <- tryCatch({
    generateTPsCTS(parents_df, transLibrary = pathway, skipInvalid = TRUE, generations = 1)
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", conditionMessage(e)))
    return(NULL)
  })
  
  elapsed <- difftime(Sys.time(), t_start, units = "mins")
  cat(sprintf("Time: %.1f min\n", elapsed))
  
  if (is.null(TPs)) {
    cat("No results (all failed)\n")
    return(data.table())
  }
  
  # Extract products
  prod_list <- products(TPs)
  
  # Combine all parent results
  all_prods <- rbindlist(prod_list, fill = TRUE, idcol = "parent_name")
  
  if (nrow(all_prods) == 0) {
    cat("No products generated\n")
    return(data.table())
  }
  
  cat(sprintf("Total candidates: %d\n", nrow(all_prods)))
  
  # Add pathway info
  all_prods[, cts_pathway := label]
  
  # Add parent SMILES by joining on parent_name
  all_prods[parents_df, parent_SMILES := i.SMILES, on = .(parent_name = name)]
  
  # Filter LIKELY, gen=1
  likely <- all_prods[likelihood == "LIKELY" & generation == 1]
  cat(sprintf("LIKELY + gen=1: %d\n", nrow(likely)))
  
  # Save full results
  fn_full <- file.path(out_dir, sprintf("cts_full_%s_%s.csv", label, timestamp))
  fwrite(all_prods, fn_full)
  cat(sprintf("Full results: %s\n", fn_full))
  
  # Save LIKELY only
  if (nrow(likely) > 0) {
    fn_likely <- file.path(out_dir, sprintf("cts_likely_%s_%s.csv", label, timestamp))
    fwrite(likely, fn_likely)
    cat(sprintf("LIKELY results: %s\n", fn_likely))
  }
  
  return(likely)
}

# Run all three pathways
all_likely <- list()

all_likely[["photolysis_ranked"]] <- process_pathway(
  parents_df, "photolysis_ranked", "photolysis_ranked"
)

all_likely[["hydrolysis"]] <- process_pathway(
  parents_df, "hydrolysis", "hydrolysis"
)

all_likely[["abiotic_reduction"]] <- process_pathway(
  parents_df, "abiotic_reduction", "abiotic_reduction"
)

# Combine all LIKELY results
combined <- rbindlist(all_likely, fill = TRUE, use.names = TRUE)

if (nrow(combined) > 0) {
  fn_combined <- file.path(out_dir, sprintf("cts_all_likely_%s.csv", timestamp))
  fwrite(combined, fn_combined)
  cat(sprintf("\n=== FINAL ===\n"))
  cat(sprintf("Total LIKELY gen=1 products: %d\n", nrow(combined)))
  cat(sprintf("Unique parents with products: %d\n", uniqueN(combined$parent_name)))
  cat(sprintf("Combined file: %s\n", fn_combined))
} else {
  cat("\nNo LIKELY products found across all pathways\n")
}

cat("\nDONE\n")