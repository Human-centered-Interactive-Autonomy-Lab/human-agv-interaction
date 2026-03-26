# ============================================================
# Subjective Trust Clustering (MFPCA + k selection + clustering)
# - Feature set: stats vs diff
# - NPC selection: elbow on eigenvalues
# - k selection: WSS elbow + silhouette, then k_final = min(both)
# - Clustering: kmeans / hierarchical / both
# - SAVES:
#     * cluster_assignments_subjective_{model}.csv  (ONLY)
#     * subjective_{model}_result.rds              (model objects)
#     * ALL figures to fig_dir with consistent names
#     * PLUS (added, like behavioral script):
#         - raw curves + dashed fitted curves (PID subset)
#         - mean functions per variable
#         - PC loading/functions per variable
# ============================================================

# =========================
# Libraries
# =========================
library(funData)
library(MFPCA)
library(ggplot2)
library(dplyr)
library(tidyr)
library(dendextend)
library(here)
library(cluster)

# Added (for arranging + colors like your behavioral script)
library(gridExtra)
library(scales)

# =========================
# User-config
# =========================
cfg <- list(
  input_path = here::here("data", "processed", "subjective_clustering_data.csv"),
  
  feature_set = "stats", # "stats" or "diff"
  base_vars = c("Trust", "Comfort", "Safe", "Expect"),
  always_include = "Trust",
  cluster_method = "both", # "kmeans" or "hierarchical" or "both"
  
  npc_max = 8,
  npc_min = 2,
  k_max = 10,
  k_default = 2,
  
  # k selection strategy:
  #   "wss"        -> WSS elbow only
  #   "silhouette" -> silhouette only
  #   "min_both"   -> min(WSS elbow, silhouette recommendation)
  k_strategy = "min_both",
  
  # when cluster_method="both", which silhouette recommendation to use:
  #   "kmeans" | "hierarchical" | "min"
  silhouette_method = "min",
  
  seed = 123,
  plot_pid_labels = TRUE,
  
  save_rds = TRUE,
  save_csv = TRUE,
  save_figures = TRUE,
  
  # figures directory (portable)
  fig_dir = here::here("clustering", "subjective_trust", "figures", "clustring")
)

# =========================
# Helpers
# =========================
ensure_dir <- function(p) if (!dir.exists(p)) dir.create(p, recursive = TRUE, showWarnings = FALSE)

save_base_plot <- function(file, width = 8, height = 5, res = 150, expr) {
  png(filename = file, width = width, height = height, units = "in", res = res)
  on.exit(dev.off(), add = TRUE)
  force(expr)
}

save_ggplot <- function(p, file, width = 8, height = 5, dpi = 150) {
  ggsave(filename = file, plot = p, width = width, height = height, dpi = dpi)
}

# Added: save arranged grobs (gridExtra::arrangeGrob output)
save_grob <- function(grob, file, width = 14, height = 5, dpi = 150) {
  ggsave(filename = file, plot = grob, width = width, height = height, dpi = dpi)
}

make_funData <- function(interaction_data, variable, argvals = 1:32) {
  wide <- interaction_data %>%
    dplyr::select(PID, Interaction_No, !!sym(variable)) %>%
    mutate(Interaction_No = as.character(Interaction_No)) %>%
    pivot_wider(
      names_from  = Interaction_No,
      values_from = all_of(variable),
      values_fill = NA
    ) %>%
    arrange(PID)
  
  wide <- wide %>% dplyr::select(PID, all_of(as.character(argvals)))
  X <- as.matrix(wide[, -1, drop = FALSE])
  
  fd <- funData(argvals = argvals, X = X)
  names(fd) <- paste0("PID_", wide$PID)
  fd
}

get_object_vars <- function(cfg) {
  
  feature_set <- match.arg(cfg$feature_set, c("stats", "diff"))
  
  suffixes <- switch(
    feature_set,
    stats = c("Mean", "Slope", "STD", "Range", "Mode", "Median", "Skewness"),
    diff  = c("Diff_Before", "Diff_After")
  )
  
  # Derived variables from base_vars × suffixes
  derived_vars <- as.vector(
    outer(cfg$base_vars, suffixes, paste, sep = "_")
  )
  
  # Final variable set:
  #   Trust (raw) + derived features
  unique(c(cfg$always_include, derived_vars))
}

# --- Elbow heuristic for NPC (eigenvalues) ---
choose_npc_elbow <- function(eigvals, npc_min = 2, npc_max = 12) {
  npc_max <- min(npc_max, length(eigvals))
  idx <- seq_len(npc_max)
  
  y <- eigvals[idx]
  x <- idx
  
  x1 <- x[1]; y1 <- y[1]
  xN <- x[length(x)]; yN <- y[length(y)]
  
  denom <- sqrt((yN - y1)^2 + (xN - x1)^2)
  if (denom == 0) return(max(npc_min, 2))
  
  dist_to_line <- abs((yN - y1)*x - (xN - x1)*y + xN*y1 - yN*x1) / denom
  dist_to_line[1:(npc_min - 1)] <- -Inf
  
  best <- which.max(dist_to_line)
  max(npc_min, min(best, npc_max))
}

# --- Elbow heuristic for k (WSS curve) ---
choose_k_elbow_wss <- function(wss_vec, k_min = 2) {
  k_max <- length(wss_vec)
  x <- seq_len(k_max)
  y <- wss_vec
  
  x1 <- x[1]; y1 <- y[1]
  xN <- x[k_max]; yN <- y[k_max]
  
  denom <- sqrt((yN - y1)^2 + (xN - x1)^2)
  if (denom == 0) return(max(k_min, 2))
  
  dist_to_line <- abs((yN - y1)*x - (xN - x1)*y + xN*y1 - yN*x1) / denom
  dist_to_line[1:(k_min - 1)] <- -Inf
  
  which.max(dist_to_line)
}

# --- Silhouette helpers ---
silhouette_avg <- function(dist_obj, clusters_int) {
  if (length(unique(clusters_int)) < 2) return(NA_real_)
  sil <- cluster::silhouette(as.integer(clusters_int), dist_obj)
  mean(sil[, "sil_width"])
}

compute_silhouette_over_k <- function(scores_mat, k_max = 10, seed = 123, nstart = 20) {
  dist_obj <- dist(scores_mat, method = "euclidean")
  
  sil_kmeans <- sapply(2:k_max, function(k) {
    set.seed(seed)
    km <- kmeans(scores_mat, centers = k, nstart = nstart)
    silhouette_avg(dist_obj, km$cluster)
  })
  
  sil_hc <- sapply(2:k_max, function(k) {
    hc <- hclust(dist_obj, method = "ward.D2")
    cl <- cutree(hc, k = k)
    silhouette_avg(dist_obj, cl)
  })
  
  list(
    dist_obj = dist_obj,
    k = 2:k_max,
    sil_kmeans = sil_kmeans,
    sil_hc = sil_hc
  )
}

# Naming tag for figures (not for CSVs)
build_fig_tag <- function(cfg, npc, k) {
  paste0(
    "subjective",
    "_feat-", cfg$feature_set,
    "_npc-", npc,
    "_k-", k,
    "_kstrategy-", cfg$k_strategy,
    "_silmeth-", cfg$silhouette_method
  )
}

# ============================================================
# Added plotting utilities (to match behavioral script outputs)
# ============================================================

# pick a sparse set of observation indices to plot (e.g., every 25th PID)
choose_obs_indices <- function(objdata, step = 25) {
  fun_names <- names(objdata[[1]])
  keep_names <- fun_names[seq(1, length(fun_names), by = step)]
  which(fun_names %in% keep_names)
}

# raw curves + dashed fitted curves overlay (per variable)
plot_raw_vs_fit <- function(objdata, mfpca_fit, var_index, obs, var_name) {
  color_vec <- hue_pal()(length(obs))
  
  g <- autoplot(objdata[[var_index]], obs = obs) +
    labs(
      x = "Interaction No.",
      y = var_name,
      col = "PID"
    ) +
    scale_y_continuous(
      limits = range(
        objdata[[var_index]]@X[obs, ],
        mfpca_fit$fit[[var_index]]@X[obs, ],
        na.rm = TRUE
      )
    ) +
    autolayer(
      mfpca_fit$fit[[var_index]],
      obs = obs,
      lty = 2,
      col = rep(color_vec, each = nObsPoints(mfpca_fit$fit[[var_index]]))
    ) +
    theme_bw(base_size = 12)
  
  g
}

# mean functions per variable
plot_mean_functions <- function(mfpca_fit, objdata) {
  g_mean <- autoplot(mfpca_fit$meanFunction, lwd = 1.2)
  lapply(seq_along(objdata), function(i) {
    g_mean[[i]] +
      scale_y_continuous(limits = range(objdata[[i]]@X, na.rm = TRUE)) +
      labs(x = "Interaction No.", y = names(objdata)[i]) +
      theme_bw(base_size = 12)
  })
}

# PC loading/functions per variable
plot_pc_functions_per_var <- function(mfpca_fit, var_index, var_name, n_pcs) {
  pc_colors <- c("#B80C0C", "#0C0CB8", "#129412", "#7A2E8E", "#0B7A75", "#C77700")
  pc_colors <- pc_colors[seq_len(min(n_pcs, length(pc_colors)))]
  
  autoplot(mfpca_fit$functions[[var_index]]) +
    geom_hline(yintercept = 0, col = "grey70", linetype = "dashed") +
    labs(title = var_name, x = "Interaction No.", y = "PC Loading") +
    scale_color_manual(values = pc_colors, name = "Princ. Comp.") +
    theme_bw(base_size = 12)
}

# =========================
# Main pipeline
# =========================
run_subjective_clustering <- function(cfg) {
  ensure_dir(cfg$fig_dir)
  
  # --- Load
  interaction_data <- read.csv(cfg$input_path)
  interaction_data$PID <- as.factor(interaction_data$PID)
  
  message("Loaded rows: ", nrow(interaction_data))
  message("Unique PIDs: ", length(unique(interaction_data$PID)))
  message("Unique Interaction_No: ", length(unique(interaction_data$Interaction_No)))
  
  # Where to save CSV/RDS (same folder as input CSV)
  output_dir <- dirname(cfg$input_path)
  
  # --- Build multiFunData
  object_vars_all <- get_object_vars(cfg)
  object_vars <- intersect(object_vars_all, names(interaction_data))
  
  missing_vars <- setdiff(object_vars_all, object_vars)
  if (length(missing_vars) > 0) {
    message("Skipping missing variables:")
    message(paste(missing_vars, collapse = ", "))
  }
  
  message("Number of variables used in multiFunData: ", length(object_vars))
  
  objdata <- multiFunData(
    setNames(
      lapply(object_vars, function(v) make_funData(interaction_data, v)),
      object_vars
    )
  )
  
  # --- Fit MFPCA with npc_max, then select NPC by elbow on eigenvalues
  npc_fit <- cfg$npc_max
  uniExpansions <- lapply(seq_along(objdata), function(i) list(type = "uFPCA", npc = npc_fit))
  
  mfpca_fit <- MFPCA(
    objdata,
    M = npc_fit,
    uniExpansions = uniExpansions,
    fit = TRUE
  )
  
  # Choose npc
  eigvals <- mfpca_fit$values
  npc <- choose_npc_elbow(eigvals, npc_min = cfg$npc_min, npc_max = cfg$npc_max)
  message("Chosen npc (elbow on eigenvalues): ", npc)
  
  # --- Scores matrix (use npc)
  scores <- mfpca_fit$scores
  scores_to_use <- scores[, 1:npc, drop = FALSE]
  colnames(scores_to_use) <- paste0("PC", seq_len(ncol(scores_to_use)))
  rownames(scores_to_use) <- rownames(scores)
  
  # =========================
  # Figures: scree + scoreplot (existing)
  # =========================
  # (we don't know k yet, so tag uses TBD)
  fig_tag0 <- build_fig_tag(cfg, npc, "TBD")
  
  if (cfg$save_figures) {
    save_base_plot(
      file = file.path(cfg$fig_dir, paste0(fig_tag0, "_01_scree_lines.png")),
      expr = screeplot(mfpca_fit, main = "Screeplot (lines)")
    )
    save_base_plot(
      file = file.path(cfg$fig_dir, paste0(fig_tag0, "_02_scree_bar.png")),
      expr = screeplot(mfpca_fit, type = "barplot", main = "Screeplot (bar)")
    )
    save_base_plot(
      file = file.path(cfg$fig_dir, paste0(fig_tag0, "_03_scoreplot.png")),
      expr = scoreplot(mfpca_fit, cex = 0.8, main = "MFPCA Scoreplot", col = rgb(0, 0, 0, alpha = 0.7))
    )
  } else {
    screeplot(mfpca_fit, main = "Screeplot (lines)")
    screeplot(mfpca_fit, type = "barplot", main = "Screeplot (bar)")
    scoreplot(mfpca_fit, cex = 0.8, main = "MFPCA Scoreplot", col = rgb(0, 0, 0, alpha = 0.7))
  }
  
  # ============================================================
  # Added figures (like behavioral script)
  #   - raw vs fitted (PID subset)
  #   - mean functions
  #   - PC loading/functions
  # ============================================================
  if (cfg$save_figures) {
    # A) Raw vs fitted curves for a subset of PIDs
    obs <- choose_obs_indices(objdata, step = 25)
    obs <- obs[obs <= nObs(objdata)]
    
    var_names_used <- names(objdata)
    
    raw_fit_plots <- lapply(seq_along(objdata), function(i) {
      plot_raw_vs_fit(
        objdata = objdata,
        mfpca_fit = mfpca_fit,
        var_index = i,
        obs = obs,
        var_name = var_names_used[i]
      )
    })
    
    grob_raw_fit <- gridExtra::arrangeGrob(grobs = raw_fit_plots, nrow = 1)
    save_grob(
      grob_raw_fit,
      file.path(cfg$fig_dir, paste0(fig_tag0, "_00_raw_vs_fit.png")),
      width = max(8, 4 * length(raw_fit_plots)),
      height = 4,
      dpi = 150
    )
    
    # B) Mean functions per variable
    mean_list <- plot_mean_functions(mfpca_fit, objdata)
    grob_mean <- gridExtra::arrangeGrob(grobs = mean_list, nrow = 1)
    save_grob(
      grob_mean,
      file.path(cfg$fig_dir, paste0(fig_tag0, "_00_mean_functions.png")),
      width = max(8, 4 * length(mean_list)),
      height = 4,
      dpi = 150
    )
    
    # C) PC loading/functions per variable
    pc_list <- lapply(seq_along(objdata), function(i) {
      plot_pc_functions_per_var(
        mfpca_fit = mfpca_fit,
        var_index = i,
        var_name = var_names_used[i],
        n_pcs = npc
      )
    })
    grob_pc <- gridExtra::arrangeGrob(grobs = pc_list, nrow = 1)
    save_grob(
      grob_pc,
      file.path(cfg$fig_dir, paste0(fig_tag0, "_00_pc_loadings.png")),
      width = max(8, 4 * length(pc_list)),
      height = 4,
      dpi = 150
    )
  }
  
  # =========================
  # 1) WSS elbow recommendation (existing)
  # =========================
  set.seed(cfg$seed)
  wss <- sapply(1:cfg$k_max, function(kk) kmeans(scores_to_use, centers = kk, nstart = 20)$tot.withinss)
  k_elbow <- choose_k_elbow_wss(wss, k_min = 2)
  if (!is.finite(k_elbow)) k_elbow <- cfg$k_default
  message("Recommended k (WSS elbow): ", k_elbow)
  
  if (cfg$save_figures) {
    save_base_plot(
      file = file.path(cfg$fig_dir, paste0(fig_tag0, "_04_elbow_wss.png")),
      expr = {
        plot(1:cfg$k_max, wss, type = "b", xlab = "k", ylab = "Total within SS",
             main = "Elbow Method for k (WSS)")
        abline(v = k_elbow, lty = 2)
      }
    )
  } else {
    plot(1:cfg$k_max, wss, type = "b", xlab = "k", ylab = "Total within SS",
         main = "Elbow Method for k (WSS)")
    abline(v = k_elbow, lty = 2)
  }
  
  # =========================
  # 2) Silhouette recommendation (existing)
  # =========================
  sil_res <- compute_silhouette_over_k(
    scores_mat = scores_to_use,
    k_max = cfg$k_max,
    seed = cfg$seed,
    nstart = 20
  )
  
  best_k_kmeans <- sil_res$k[which.max(sil_res$sil_kmeans)]
  best_k_hc     <- sil_res$k[which.max(sil_res$sil_hc)]
  
  message(sprintf("Recommended k (silhouette, kmeans): %d (avg = %.3f)",
                  best_k_kmeans, max(sil_res$sil_kmeans, na.rm = TRUE)))
  message(sprintf("Recommended k (silhouette, hierarchical): %d (avg = %.3f)",
                  best_k_hc, max(sil_res$sil_hc, na.rm = TRUE)))
  
  if (cfg$save_figures) {
    save_base_plot(
      file = file.path(cfg$fig_dir, paste0(fig_tag0, "_05_silhouette_kmeans.png")),
      expr = {
        plot(sil_res$k, sil_res$sil_kmeans, type = "b",
             xlab = "k", ylab = "Average silhouette",
             main = "Silhouette vs k (K-means)")
        abline(v = best_k_kmeans, lty = 2)
      }
    )
    save_base_plot(
      file = file.path(cfg$fig_dir, paste0(fig_tag0, "_06_silhouette_hierarchical.png")),
      expr = {
        plot(sil_res$k, sil_res$sil_hc, type = "b",
             xlab = "k", ylab = "Average silhouette",
             main = "Silhouette vs k (Hierarchical, Ward.D2)")
        abline(v = best_k_hc, lty = 2)
      }
    )
  } else {
    plot(sil_res$k, sil_res$sil_kmeans, type = "b",
         xlab = "k", ylab = "Average silhouette",
         main = "Silhouette vs k (K-means)")
    abline(v = best_k_kmeans, lty = 2)
    
    plot(sil_res$k, sil_res$sil_hc, type = "b",
         xlab = "k", ylab = "Average silhouette",
         main = "Silhouette vs k (Hierarchical, Ward.D2)")
    abline(v = best_k_hc, lty = 2)
  }
  
  k_sil <- switch(
    cfg$silhouette_method,
    kmeans = best_k_kmeans,
    hierarchical = best_k_hc,
    min = min(best_k_kmeans, best_k_hc)
  )
  message("Recommended k (silhouette, chosen rule): ", k_sil)
  
  # =========================
  # 3) Final k selection (existing)
  # =========================
  k_final <- switch(
    cfg$k_strategy,
    wss = k_elbow,
    silhouette = k_sil,
    min_both = min(k_elbow, k_sil)
  )
  k_final <- max(2, min(k_final, cfg$k_max))
  message("FINAL k used for clustering: ", k_final)
  
  k <- k_final
  fig_tag <- build_fig_tag(cfg, npc, k)
  
  # =========================
  # Base scores df (existing)
  # =========================
  scores_df <- as.data.frame(scores_to_use)
  scores_df$PID <- as.numeric(gsub("PID_", "", rownames(scores_to_use)))
  
  # Precompute dist for final silhouette
  dist_obj_final <- dist(scores_to_use, method = "euclidean")
  
  # =========================
  # KMEANS (existing)
  # =========================
  results <- list()
  
  if (cfg$cluster_method %in% c("kmeans", "both")) {
    set.seed(cfg$seed)
    km <- kmeans(scores_to_use, centers = k, nstart = 20)
    
    df_km <- scores_df
    df_km$Cluster <- as.factor(km$cluster)
    
    sil_km_avg <- silhouette_avg(dist_obj_final, km$cluster)
    message(sprintf("Final silhouette (kmeans, k=%d): %.3f", k, sil_km_avg))
    
    p_km <- ggplot(df_km, aes(x = PC1, y = PC2, color = Cluster)) +
      geom_point(size = 3) +
      labs(title = "K-means Clusters (MFPCA Scores)", x = "PC1", y = "PC2") +
      theme_minimal()
    
    if (cfg$plot_pid_labels) {
      p_km <- p_km + geom_text(aes(label = PID), vjust = -0.5, size = 3, show.legend = FALSE)
    }
    
    if (cfg$save_figures) {
      save_ggplot(p_km, file.path(cfg$fig_dir, paste0(fig_tag, "_07_scatter_kmeans.png")))
    } else {
      print(p_km)
    }
    
    # SAVE ONLY ONE CSV FOR KMEANS
    if (cfg$save_csv) {
      write.csv(df_km, file.path(output_dir, "cluster_assignments_subjective_kmeans.csv"), row.names = FALSE)
    }
    
    # SAVE MODEL OBJECT
    if (cfg$save_rds) {
      saveRDS(km, file.path(output_dir, "subjective_kmeans_result.rds"))
    }
    
    results$kmeans <- list(model = km, df = df_km, silhouette = sil_km_avg)
  }
  
  # =========================
  # HIERARCHICAL (existing)
  # =========================
  if (cfg$cluster_method %in% c("hierarchical", "both")) {
    hc <- hclust(dist_obj_final, method = "ward.D2")
    
    if (cfg$save_figures) {
      save_base_plot(
        file = file.path(cfg$fig_dir, paste0(fig_tag, "_08_dendrogram_plain.png")),
        expr = plot(hc, labels = FALSE, main = "Hierarchical Clustering Dendrogram", xlab = "", sub = "")
      )
    } else {
      plot(hc, labels = FALSE, main = "Hierarchical Clustering Dendrogram", xlab = "", sub = "")
    }
    
    hc_clusters <- cutree(hc, k = k)
    
    df_hc <- scores_df
    df_hc$Cluster <- as.factor(hc_clusters)
    
    sil_hc_avg <- silhouette_avg(dist_obj_final, hc_clusters)
    message(sprintf("Final silhouette (hierarchical, k=%d): %.3f", k, sil_hc_avg))
    
    # colored dendrogram
    dend <- as.dendrogram(hc)
    dend <- color_branches(dend, k = k)
    dend <- set(dend, "labels_cex", 0.7)
    
    if (cfg$save_figures) {
      save_base_plot(
        file = file.path(cfg$fig_dir, paste0(fig_tag, "_09_dendrogram_colored.png")),
        expr = plot(dend, main = "Colored Dendrogram")
      )
    } else {
      plot(dend, main = "Colored Dendrogram")
    }
    
    p_hc <- ggplot(df_hc, aes(x = PC1, y = PC2, color = Cluster)) +
      geom_point(size = 3) +
      labs(title = "Hierarchical Clusters (MFPCA Scores)", x = "PC1", y = "PC2") +
      theme_minimal()
    
    if (cfg$plot_pid_labels) {
      p_hc <- p_hc + geom_text(aes(label = PID), vjust = -0.5, size = 3, show.legend = FALSE)
    }
    
    if (cfg$save_figures) {
      save_ggplot(p_hc, file.path(cfg$fig_dir, paste0(fig_tag, "_10_scatter_hierarchical.png")))
    } else {
      print(p_hc)
    }
    
    # SAVE ONLY ONE CSV FOR HIERARCHICAL
    if (cfg$save_csv) {
      write.csv(df_hc, file.path(output_dir, "cluster_assignments_subjective_hierarchical.csv"), row.names = FALSE)
    }
    
    # SAVE MODEL OBJECT
    if (cfg$save_rds) {
      saveRDS(hc, file.path(output_dir, "subjective_hierarchical_result.rds"))
    }
    
    results$hierarchical <- list(model = hc, clusters = hc_clusters, df = df_hc, silhouette = sil_hc_avg)
  }
  
  invisible(list(
    cfg = cfg,
    npc = npc,
    k = k,
    k_elbow = k_elbow,
    k_sil_kmeans = best_k_kmeans,
    k_sil_hierarchical = best_k_hc,
    k_sil_chosen = k_sil,
    mfpca = mfpca_fit,
    results = results,
    fig_dir = cfg$fig_dir,
    output_dir = output_dir
  ))
}

# =========================
# Run
# =========================
res <- run_subjective_clustering(cfg)
