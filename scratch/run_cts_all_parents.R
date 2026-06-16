library(patRoon)
library(data.table)

# ============================================================
# CTS 全量母体预测脚本
# 对所有 3219 个 unique 母体 SMILES 运行三条转化路径
# ============================================================

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- file.path("results", "cts_augmented")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 加载全部母体
parents_df <- read.csv("scratch/cts_input_all_parents.csv", stringsAsFactors = FALSE)
parents_df$name <- paste0("P", seq_len(nrow(parents_df)))
n_total <- nrow(parents_df)
cat(sprintf("Total parents: %d\n", n_total))

# 三条转化路径
pathways <- list(
  photolysis_ranked = "photolysis_ranked",
  hydrolysis       = "hydrolysis",
  abiotic_reduction = "abiotic_reduction"
)

# 分批大小（每批100个母体，避免内存/超时问题）
BATCH_SIZE <- 100
batches <- split(parents_df, ceiling(seq_len(n_total) / BATCH_SIZE))
n_batches <- length(batches)
cat(sprintf("Batches: %d x %d parents\n", n_batches, BATCH_SIZE))

# ============================================================
# 逐路径、逐批次执行
# ============================================================
all_likely <- list()

for (pw_name in names(pathways)) {
  pw_label <- pathways[[pw_name]]
  cat(sprintf("\n========================================\n"))
  cat(sprintf("PATHWAY: %s\n", pw_label))
  cat(sprintf("========================================\n"))
  
  pw_results <- list()
  n_pw_candidates <- 0
  n_pw_parents_with_tp <- 0
  
  for (bi in seq_along(batches)) {
    batch <- batches[[bi]]
    cat(sprintf("\n[%s] Batch %d/%d (%d parents)...\n", pw_label, bi, n_batches, nrow(batch)))
    t0 <- Sys.time()
    
    # 容错运行
    TPs <- tryCatch({
      generateTPsCTS(batch, transLibrary = pw_label, skipInvalid = TRUE, generations = 1)
    }, error = function(e) {
      cat(sprintf("  ERROR: %s\n", conditionMessage(e)))
      return(NULL)
    })
    
    elapsed <- round(difftime(Sys.time(), t0, units = "secs"), 1)
    
    if (is.null(TPs)) {
      cat(sprintf("  Failed (%.1fs), skipping batch\n", elapsed))
      next
    }
    
    # 提取产物
    prod_list <- tryCatch({
      products(TPs)
    }, error = function(e) {
      cat(sprintf("  ERROR extracting products: %s\n", conditionMessage(e)))
      return(NULL)
    })
    
    if (is.null(prod_list)) next
    
    prods <- rbindlist(prod_list, fill = TRUE, idcol = "parent_name")
    
    if (nrow(prods) == 0) {
      cat(sprintf("  No products (%.1fs)\n", elapsed))
      next
    }
    
    # 添加母体SMILES
    prods[batch, parent_SMILES := i.SMILES, on = .(parent_name = name)]
    
    # 保存全部结果
    fn <- file.path(out_dir, sprintf("cts_%s_batch%03d_%s.csv", pw_name, bi, timestamp))
    fwrite(prods, fn)
    
    # 筛选 LIKELY + gen=1
    likely <- prods[likelihood == "LIKELY" & generation == 1]
    n_likely <- nrow(likely)
    
    cat(sprintf("  Done (%.1fs): %d candidates, %d LIKELY gen=1\n", 
                elapsed, nrow(prods), n_likely))
    
    if (n_likely > 0) {
      likely[, cts_pathway := pw_label]
      pw_results[[length(pw_results) + 1]] <- likely
      n_pw_candidates <- n_pw_candidates + n_likely
      n_pw_parents_with_tp <- n_pw_parents_with_tp + uniqueN(likely$parent_SMILES)
    }
  }
  
  # 合并当前路径的所有LIKELY
  if (length(pw_results) > 0) {
    pw_combined <- rbindlist(pw_results, fill = TRUE)
    fn_pw <- file.path(out_dir, sprintf("cts_likely_%s_%s.csv", pw_name, timestamp))
    fwrite(pw_combined, fn_pw)
    cat(sprintf("\n%s LIKELY total: %d products from %d unique parents\n", 
                pw_label, nrow(pw_combined), uniqueN(pw_combined$parent_SMILES)))
    all_likely[[pw_name]] <- pw_combined
  } else {
    cat(sprintf("\n%s: No LIKELY products found\n", pw_label))
    all_likely[[pw_name]] <- data.table()
  }
}

# ============================================================
# 合并所有路径的LIKELY结果
# ============================================================
combined <- rbindlist(all_likely, fill = TRUE, use.names = TRUE)

if (nrow(combined) > 0) {
  fn_combined <- file.path(out_dir, sprintf("cts_all_likely_%s.csv", timestamp))
  fwrite(combined, fn_combined)
  
  cat(sprintf("\n========================================\n"))
  cat(sprintf("FINAL SUMMARY\n"))
  cat(sprintf("========================================\n"))
  cat(sprintf("Total LIKELY gen=1 products: %d\n", nrow(combined)))
  cat(sprintf("Unique parents with >=1 LIKELY product: %d\n", uniqueN(combined$parent_SMILES)))
  cat(sprintf("Pathway breakdown:\n"))
  for (pw_name in names(pathways)) {
    sub <- combined[cts_pathway == pathways[[pw_name]]]
    if (nrow(sub) > 0) {
      cat(sprintf("  %s: %d products, %d parents\n", 
                  pw_name, nrow(sub), uniqueN(sub$parent_SMILES)))
    } else {
      cat(sprintf("  %s: 0 products\n", pw_name))
    }
  }
  cat(sprintf("Combined output: %s\n", fn_combined))
} else {
  cat("\nWARNING: No LIKELY products found across any pathway!\n")
}

cat("\nDONE\n")
