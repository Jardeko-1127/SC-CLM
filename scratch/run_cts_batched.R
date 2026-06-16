library(patRoon)
library(data.table)

parents_df <- read.csv("scratch/cts_input_human_only_parents.csv", stringsAsFactors = FALSE)
parents_df$name <- paste0("P", seq_len(nrow(parents_df)))
n_total <- nrow(parents_df)
batch_size <- 100
batches <- split(parents_df, ceiling(seq_len(n_total) / batch_size))
cat(sprintf("Total: %d parents, Batches: %d x %d\n", n_total, length(batches), batch_size))

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- file.path("results", "cts_augmented")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

pathways <- list(
  photolysis_ranked = "photolysis_ranked",
  hydrolysis = "hydrolysis", 
  abiotic_reduction = "abiotic_reduction"
)

all_results <- list()
total_candidates <- 0

for (pw_name in names(pathways)) {
  pw_label <- pathways[[pw_name]]
  for (bi in seq_along(batches)) {
    batch <- batches[[bi]]
    cat(sprintf("\n[%s] Batch %d/%d (%d parents)...\n", pw_label, bi, length(batches), nrow(batch)))
    t0 <- Sys.time()
    
    TPs <- tryCatch({
      generateTPsCTS(batch, transLibrary = pw_label, skipInvalid = TRUE, generations = 1)
    }, error = function(e) {
      cat(sprintf("  ERROR: %s\n", conditionMessage(e)))
      return(NULL)
    })
    
    if (is.null(TPs)) next
    
    prod_list <- products(TPs)
    prods <- rbindlist(prod_list, fill = TRUE, idcol = "parent_name")
    
    if (nrow(prods) == 0) {
      cat(sprintf("  No products\n"))
      next
    }
    
    # Add parent SMILES
    prods[batch, parent_SMILES := i.SMILES, on = .(parent_name = name)]
    
    # Save all for this batch+pathway
    fn <- file.path(out_dir, sprintf("cts_%s_batch%02d_%s.csv", pw_name, bi, timestamp))
    fwrite(prods, fn)
    
    # Filter LIKELY gen=1
    likely <- prods[likelihood == "LIKELY" & generation == 1]
    n_likely <- nrow(likely)
    total_candidates <- total_candidates + n_likely
    
    elapsed <- round(difftime(Sys.time(), t0, units = "secs"), 1)
    cat(sprintf("  Products: %d total, %d LIKELY gen=1, Time: %.1fs\n", 
                nrow(prods), n_likely, elapsed))
  }
}

cat(sprintf("\n=== DONE ===\n"))
cat(sprintf("Total LIKELY gen=1 candidates: %d\n", total_candidates))
cat(sprintf("Output dir: %s\n", out_dir))