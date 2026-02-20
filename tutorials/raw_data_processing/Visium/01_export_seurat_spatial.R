#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(Seurat)
  library(ggplot2)
})

option_list <- list(
  make_option(c("-i", "--input"), type="character", default=NULL, help="Path to input Seurat .Robj or .rds file"),
  make_option(c("-o", "--outdir"), type="character", default="exports", help="Output directory [default=exports]"),
  make_option(c("-s", "--sample"), type="character", default="sample", help="Sample prefix for output files [default=sample]")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

if (is.null(opt$input)) {
  print_help(opt_parser)
  stop("Input file must be provided.", call.=FALSE)
}

dir.create(opt$outdir, showWarnings = FALSE, recursive = TRUE)

cat("\n=== Processing:", opt$sample, "===\n")

# 加载函数 (保留你原本优秀的临时环境加载逻辑)
load_seurat_from_robj <- function(path) {
  env <- new.env(parent = emptyenv())
  ok_load <- FALSE
  try({ loaded_names <- load(path, envir = env); ok_load <- TRUE }, silent = TRUE)
  if (ok_load && length(loaded_names) > 0) {
    candidates <- mget(loaded_names, envir = env)
    is_seurat <- vapply(candidates, function(x) inherits(x, "Seurat"), logical(1))
    if (any(is_seurat)) return(candidates[[which(is_seurat)[1]]])
    return(candidates[[1]])
  }
  return(readRDS(path))
}

obj <- load_seurat_from_robj(opt$input)

if (!inherits(obj, "Seurat")) stop("Loaded object is not Seurat.")

# 提取空间坐标
imgs <- Images(obj)
if (length(imgs) == 0) stop("No spatial images found in this Seurat object.")
img_use <- imgs[1]

coords <- GetTissueCoordinates(obj[[img_use]])
positions <- data.frame(
  Barcode = rownames(coords),
  x = as.numeric(coords$imagecol),
  y = as.numeric(coords$imagerow)
)
pos_path <- file.path(opt$outdir, paste0(opt$sample, "_positions_spatial.csv"))
write.csv(positions, pos_path, row.names = FALSE)
cat("Saved spatial coordinates to:", pos_path, "\n")

# 提取 Metadata
meta <- obj@meta.data
meta$Barcode <- rownames(meta)
meta <- meta[, c("Barcode", setdiff(colnames(meta), "Barcode"))]
meta_path <- file.path(opt$outdir, paste0(opt$sample, "_metadata.csv"))
write.csv(meta, meta_path, row.names = FALSE)
cat("Saved metadata to:", meta_path, "\n")

cat("Done!\n")