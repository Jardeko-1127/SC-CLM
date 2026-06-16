library(patRoon)
library(data.table)

parents_df <- read.csv("scratch/cts_input_human_only_parents.csv", stringsAsFactors = FALSE)
test_df <- head(parents_df, 2)
test_df$name <- paste0("CP_", seq_len(nrow(test_df)))

TPs <- generateTPsCTS(test_df, transLibrary = "photolysis_ranked", skipInvalid = TRUE)

cat("Class:", class(TPs), "\n")
cat("Length:", length(TPs), "\n")
cat("Names:", names(TPs), "\n")

# Try to get products
prods <- products(TPs)
cat("\nproducts() class:", class(prods), "\n")
if (is.data.frame(prods)) {
  cat(sprintf("Rows: %d, Cols: %s\n", nrow(prods), paste(names(prods), collapse=", ")))
  print(head(prods, 3))
} else if (is.list(prods)) {
  for (i in seq_along(prods)) {
    cat(sprintf("\nParent %d:\n", i))
    if (!is.null(prods[[i]])) print(prods[[i]])
  }
}

# Also try as.data.table with different args
cat("\nTrying as.data.table without parents=...\n")
dt <- tryCatch(as.data.table(TPs), error = function(e) { cat("Error:", conditionMessage(e), "\n"); NULL })
if (!is.null(dt)) print(names(dt))