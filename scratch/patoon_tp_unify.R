# patRoon TP 统一格式 + 各途径导出 CSV
# 依赖：dplyr, data.table, patRoon；需先定义 tp_to_dt()（见你的 generateTPsBioTransformer_rowwise_safe 脚本）

#' 单个 TP 对象 → data.table（与 rowwise_safe 输出一致）
tp_result_to_unified_dt <- function(
    res,
    type_label,
    source = c("biotransformer", "cts", "library"),
    failed_parent_rows = integer(0)
) {
  source <- match.arg(source)
  dt <- if (is.null(res)) data.table() else tp_to_dt(res)

  if (nrow(dt) > 0) {
    dt[, `:=`(bt_type = type_label, tp_source = source)]
  }

  attr(dt, "bt_type") <- type_label
  attr(dt, "source") <- source
  attr(dt, "failed_parent_rows") <- failed_parent_rows
  dt
}

#' rowwise 结果已是 data.table 时补齐列与属性
as_unified_tp_dt <- function(
    x,
    type_label,
    source = c("biotransformer", "cts", "library")
) {
  source <- match.arg(source)
  if (!inherits(x, "data.table")) {
    return(tp_result_to_unified_dt(
      x, type_label, source,
      failed_parent_rows = attr(x, "failed_parent_rows") %||% integer(0)
    ))
  }
  dt <- copy(x)
  if (!"bt_type" %in% names(dt)) dt[, bt_type := type_label]
  if (!"tp_source" %in% names(dt)) dt[, tp_source := source]
  attr(dt, "bt_type") <- type_label
  attr(dt, "source") <- source
  if (is.null(attr(dt, "failed_parent_rows"))) {
    attr(dt, "failed_parent_rows") <- integer(0)
  }
  dt
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

#' 写出单个途径 CSV
export_tp_dt_csv <- function(
    dt,
    pathway_id,
    out_dir = "results/benchmark/v5/patoon",
    timestamp = format(Sys.time(), "%Y%m%d")
) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  fn <- file.path(out_dir, sprintf("patoon_%s_%s.csv", pathway_id, timestamp))
  data.table::fwrite(dt, fn)
  message("Wrote: ", normalizePath(fn, winslash = "/", mustWork = FALSE))
  invisible(fn)
}

#' 从 patRoon 对象或 data.table 统一转换并写 CSV
export_patoon_pathway <- function(
    res,
    pathway_id,
    type_label,
    source = c("biotransformer", "cts", "library"),
    out_dir = "results/benchmark/v5/patoon",
    timestamp = format(Sys.time(), "%Y%m%d"),
    failed_parent_rows = NULL
) {
  source <- match.arg(source)
  if (inherits(res, "data.table")) {
    dt <- as_unified_tp_dt(res, type_label, source)
  } else {
    dt <- tp_result_to_unified_dt(
      res, type_label, source,
      failed_parent_rows = failed_parent_rows %||% integer(0)
    )
  }
  export_tp_dt_csv(dt, pathway_id, out_dir, timestamp)
}

#' 批量：命名列表 → 全部 CSV
#' @param tp_list 命名列表，如 list(bt_ecbased = TPs_BT1, ...)
#' @param spec data.frame 列：name, type_label, source, pathway_id（可选，默认用 name）
export_all_patoon_pathways <- function(
    tp_list,
    spec = NULL,
    out_dir = "results/benchmark/v5/patoon",
    timestamp = format(Sys.time(), "%Y%m%d")
) {
  nms <- names(tp_list)
  if (is.null(nms)) nms <- paste0("pathway_", seq_along(tp_list))

  paths <- character(length(tp_list))
  for (i in seq_along(tp_list)) {
    nm <- nms[i]
    x <- tp_list[[i]]
    if (!is.null(spec)) {
      row <- spec[spec$name == nm, , drop = FALSE]
      if (nrow(row) == 0) row <- spec[i, , drop = FALSE]
      type_label <- row$type_label[[1]]
      source <- row$source[[1]]
      pathway_id <- if ("pathway_id" %in% names(row)) row$pathway_id[[1]] else nm
    } else {
      type_label <- sub("^TPs_(BT|cts)_", "", nm)
      source <- if (grepl("^TPs_cts", nm)) "cts" else "biotransformer"
      pathway_id <- tolower(gsub("^TPs_", "", nm))
    }
    paths[i] <- export_patoon_pathway(
      x, pathway_id, type_label, source,
      out_dir = out_dir, timestamp = timestamp
    )
  }
  invisible(paths)
}

# ── 在 R 里跑完 TPs_BT1 … TPs_cts_PHr 后，执行下面一段即可写出全部 CSV ──

export_patoon_test_all <- function(
    parents,
    out_dir = "results/benchmark/v5/patoon",
    timestamp = format(Sys.time(), "%Y%m%d")
) {
  stopifnot(exists("TPs_BT1"), exists("TPs_BT6"))

  spec <- data.frame(
    name = c(
      "TPs_BT1", "TPs_BT2", "TPs_BT3",
      "TPs_BT4", "TPs_BT5", "TPs_BT6",
      "TPs_cts_AR", "TPs_cts_HY", "TPs_cts_PHu", "TPs_cts_PHr"
    ),
    type_label = c(
      "ecbased", "cyp450", "hgut",
      "superbio", "allHuman", "env",
      "abiotic_reduction", "hydrolysis",
      "photolysis_unranked", "photolysis_ranked"
    ),
    source = c(
      rep("biotransformer", 6),
      rep("cts", 4)
    ),
    pathway_id = c(
      "bt_ecbased", "bt_cyp450", "bt_hgut",
      "bt_superbio", "bt_allhuman", "bt_env",
      "cts_abiotic_reduction", "cts_hydrolysis",
      "cts_photolysis_unranked", "cts_photolysis_ranked"
    ),
    stringsAsFactors = FALSE
  )

  tp_list <- list(
    TPs_BT1 = TPs_BT1, TPs_BT2 = TPs_BT2, TPs_BT3 = TPs_BT3,
    TPs_BT4 = TPs_BT4, TPs_BT5 = TPs_BT5, TPs_BT6 = TPs_BT6,
    TPs_cts_AR = TPs_cts_AR, TPs_cts_HY = TPs_cts_HY,
    TPs_cts_PHu = TPs_cts_PHu, TPs_cts_PHr = TPs_cts_PHr
  )

  if (exists("TPs_Lib")) {
    tp_list$TPs_Lib <- TPs_Lib
    spec <- rbind(
      spec,
      data.frame(
        name = "TPs_Lib",
        type_label = "library",
        source = "library",
        pathway_id = "library_default",
        stringsAsFactors = FALSE
      )
    )
  }

  paths <- export_all_patoon_pathways(tp_list, spec, out_dir, timestamp)

  # 失败行索引（仅 rowwise 有）
  fail_dir <- file.path(out_dir, "meta")
  dir.create(fail_dir, recursive = TRUE, showWarnings = FALSE)
  for (nm in c("TPs_BT4", "TPs_BT5", "TPs_BT6")) {
    if (!exists(nm)) next
    x <- get(nm)
    fr <- attr(x, "failed_parent_rows")
    if (length(fr)) {
      ff <- file.path(fail_dir, paste0(sub("^TPs_", "", nm), "_failed_rows.txt"))
      writeLines(as.character(fr), ff)
      message("Failed rows: ", ff)
    }
  }

  invisible(paths)
}

# 用法（工作目录设为 SC_CLM 仓库根，或改 out_dir 为绝对路径）：
# source("scratch/patoon_tp_unify.R")
# export_patoon_test_all(parents)

# 或单条：
# export_patoon_pathway(TPs_BT1, "bt_ecbased", "ecbased", "biotransformer")
# export_patoon_pathway(TPs_BT6, "bt_env", "env", "biotransformer")
# export_patoon_pathway(TPs_cts_AR, "cts_abiotic_reduction", "abiotic_reduction", "cts")
